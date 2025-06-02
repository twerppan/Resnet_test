import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import networkx as nx
from collections import defaultdict, deque
import pandas as pd

# ------------ 用户给定的工具函数 ------------
def find_critical_edges_dag(edges):
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    nodes = set()
    for u, v in edges:
        out_degree[u] += 1
        in_degree[v] += 1
        nodes.add(u)
        nodes.add(v)
    sources = [n for n in nodes if in_degree[n] == 0]
    sinks   = [n for n in nodes if out_degree[n] == 0]
    if len(sources)!=1 or len(sinks)!=1:
        raise ValueError("图必须包含且仅包含一个源和一个汇")
    source, sink = sources[0], sinks[0]
    adj = defaultdict(list)
    rev = defaultdict(list)
    for u,v in edges:
        adj[u].append(v)
        rev[v].append(u)
    topo, q, indeg = [], deque([source]), dict(in_degree)
    while q:
        u = q.popleft(); topo.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v]==0: q.append(v)
    src_paths = defaultdict(int); src_paths[source]=1
    for u in topo:
        for v in adj[u]: src_paths[v]+=src_paths[u]
    dest_paths = defaultdict(int); dest_paths[sink]=1
    for u in reversed(topo):
        for v in rev[u]: dest_paths[v]+=dest_paths[u]
    total = src_paths[sink]
    critical=[]
    for u,v in edges:
        if src_paths[u]*dest_paths[v]==total:
            critical.append((u,v))
    return critical

# 计算 FLOPs 工具
def flops_module(mod, out_shape):
    elems = int(torch.tensor(out_shape).prod())
    if isinstance(mod, nn.Conv2d):
        Cin, Cout = mod.in_channels, mod.out_channels
        Kh, Kw = mod.kernel_size
        H, W = out_shape[2], out_shape[3]
        g = mod.groups
        return 2 * (Cin//g)*Kh*Kw * Cout * H * W
    if isinstance(mod, nn.Linear):
        return 2 * mod.in_features * mod.out_features
    if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return 2 * elems
    if isinstance(mod, (nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
        return elems
    return 0

def flops_function(fn, out_shape):
    elems = int(torch.tensor(out_shape).prod())
    if fn in (F.relu,): return elems
    if fn in (torch.add,): return elems
    return 0

# -------------- 模型分析与表格生成 --------------
def profile_model(model: nn.Module, input_shape=(1,3,224,224)):
    fx_net = symbolic_trace(model)
    ShapeProp(fx_net).propagate(torch.randn(input_shape))
    G = nx.DiGraph()
    for n in fx_net.graph.nodes:
        G.add_node(n.name)
    for n in fx_net.graph.nodes:
        for inp in n.all_input_nodes:
            G.add_edge(inp.name, n.name)
    edges = list(G.edges())
    cuts  = find_critical_edges_dag(edges)
    rows=[]; nodes=list(fx_net.graph.nodes)
    for i,node in enumerate(nodes):
        tm = node.meta.get('tensor_meta')
        shape = tuple(tm.shape) if tm else None
        if node.op=='call_module':
            mod = fx_net.get_submodule(node.target)
            f = flops_module(mod, shape) if shape else 0
        elif node.op=='call_function':
            f = flops_function(node.target, shape) if shape else 0
        else:
            f = 0
        nxt = (node.name, nodes[i+1].name) if i+1<len(nodes) else None
        rows.append({
            'idx': i,
            'node': node.name,
            'shape': shape,
            'size': int(torch.tensor(shape).prod()) if shape else None,
            'flops (MFLOPs)': f/1e6,
            'edge_to_next': nxt,
            'is_cut': nxt in cuts
        })
    df = pd.DataFrame(rows)
    return df

# -------------- 设备定义 --------------
# 示例：5个设备，参数可按需修改
devices = [
    {'id':0,'mem_limit':2.0,'speed':10.0,'bw':10.0},
    {'id':1,'mem_limit':4.0,'speed':8.0, 'bw':10.0},
    {'id':2,'mem_limit':6.0,'speed':6.0, 'bw':10.0},
    {'id':3,'mem_limit':4.0,'speed':12.0,'bw':10.0},
    {'id':4,'mem_limit':6.0,'speed':11.0,'bw':10.0},
]

# 识别瓶颈子段：滑窗 + 阈值
def find_bottleneck_segments(df, window=4, thresh_ratio=1.3):#窗口值3，阈值为1.3
    flops = df['flops (MFLOPs)'].values
    avg = flops.mean()
    segs=[]
    for i in range(len(flops)-window+1):
        s = sum(flops[i:i+window])
        if s > avg * thresh_ratio:
            segs.append((i, i+window))
    merged=[]
    for st, ed in sorted(segs):
        if not merged or st > merged[-1][1]:
            merged.append([st, ed])
        else:
            merged[-1][1] = max(merged[-1][1], ed)
    return merged

# -------------- 切分并分配 --------------
def partition_and_assign(df, devices):
    segs = find_bottleneck_segments(df)#调用函数寻找瓶颈片段
    blocks = []
    prev_end = 0# 跟踪已处理层的末尾位置
    # 设备按算力降序  将计算资源按降序分配给设备
    dev_sorted = sorted(devices, key=lambda d: d['speed'], reverse=True)
    dev_iter = iter(dev_sorted)     ## 创建设备迭代器顺序匹配
    for (st, ed) in segs:
        # 流水线段
        if prev_end < st:
            dev = next(dev_iter)
            blocks.append({'type':'pipeline','layers':list(range(prev_end, st)),'devices':[dev['id']]})
        # 张量并行段
        blk = {'type':'tensor','layers':list(range(st, ed))}
        # 默认选2卡
        blk['devices']=[d['id'] for d in dev_sorted[:2]]
        blocks.append(blk)
        prev_end = ed
    # 尾部流水线段
    if prev_end < len(df):
        dev = next(dev_iter)
        blocks.append({'type':'pipeline','layers':list(range(prev_end, len(df))),'devices':[dev['id']]})
    return blocks

# 计算每块与每设备指标
def compute_metrics(df, blocks, devices):
    # 初始化设备统计
    stats = {d['id']:{'flops':0,'compute_time':0,'comm_time':0,'mem':0} for d in devices}
    # 逐块计算
    for i, blk in enumerate(blocks):
        # 计算该块 FLOPs 与输出 size
        flops_total = df.loc[blk['layers'], 'flops (MFLOPs)'].sum()
        last_shape = df.loc[blk['layers'][-1], 'shape']
        out_elems = int(torch.tensor(last_shape).prod())
        out_bytes = out_elems * 4
        if blk['type']=='pipeline':
            dev_id = blk['devices'][0]
            dev = next(d for d in devices if d['id']==dev_id)
            t_comp = flops_total / dev['speed']
            t_comm = 0
            # 与下块通信
            if i < len(blocks)-1:
                t_comm = out_bytes / (dev['bw']*1e9) * 1000
            stats[dev_id]['flops'] += flops_total
            stats[dev_id]['compute_time'] += t_comp
            stats[dev_id]['comm_time'] += t_comm
            stats[dev_id]['mem'] += out_bytes/(1024**3)
        else:  # tensor
            dev_ids = blk['devices']
            speeds = [d['speed'] for d in devices if d['id'] in dev_ids]
            sum_speed = sum(speeds)
            t_comp = flops_total / sum_speed
            t_comm = 0
            if i < len(blocks)-1:
                bw_min = min(d['bw'] for d in devices if d['id'] in dev_ids)
                t_comm = 2 * out_bytes * (len(dev_ids)-1)/len(dev_ids) / (bw_min*1e9) *1000
            # 分摊到每卡
            for dev_id in dev_ids:
                stats[dev_id]['flops'] += flops_total/len(dev_ids)
                stats[dev_id]['compute_time'] += t_comp
                stats[dev_id]['comm_time'] += t_comm
                stats[dev_id]['mem'] += out_bytes/(len(dev_ids))/(1024**3)
    return stats

# -------------- 主流程 --------------
def auto_partition(model):
    df = profile_model(model)
    print("Layer profiling:")
    print(df.to_markdown(index=False))
    blocks = partition_and_assign(df, devices)
    print("\nPartition result:")
    for blk in blocks:
        kind = '张量' if blk['type']=='tensor' else '流水线'
        names = [df.iloc[i]['node'] for i in blk['layers']]
        print(f"Block {names}: {kind} on devices {blk['devices']}")
    # 计算并打印设备统计
    stats = compute_metrics(df, blocks, devices)
    print("\nDevice resource usage:")
    for dev_id, st in stats.items():
        print(f"Device{dev_id}: FLOPs={st['flops']:.2f}MF, "
              f"Compute={st['compute_time']:.2f}ms, "
              f"Comm={st['comm_time']:.2f}ms, "
              f"Mem={st['mem']:.3f}GB")
    return df, blocks, stats

# -------------- 使用示例 --------------
if __name__=='__main__':
    class CustomNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3,16,3,padding=1)
            self.bn1   = nn.BatchNorm2d(16)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(16,16,3,padding=1)
            self.bn2   = nn.BatchNorm2d(16)
            self.relu2 = nn.ReLU()
            self.pool  = nn.MaxPool2d(2)
            self.conv3 = nn.Conv2d(16,16,3,padding=1)
            self.bn3   = nn.BatchNorm2d(16)
            self.relu3 = nn.ReLU()
            self.conv4 = nn.Conv2d(16,16,3,padding=1)
            self.bn4   = nn.BatchNorm2d(16)
            self.br1_conv = nn.Conv2d(16,8,1)
            self.br1_bn   = nn.BatchNorm2d(8)
            self.br2_conv = nn.Conv2d(16,8,1)
            self.br2_bn   = nn.BatchNorm2d(8)
            self.global_pool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(16,10)
        def forward(self,x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.pool(x)
            identity = x
            out = self.relu3(self.bn3(self.conv3(x)))
            out = self.bn4(self.conv4(out))
            x = F.relu(out+identity)
            b1 = self.br1_bn(self.br1_conv(x))
            b2 = self.br2_bn(self.br2_conv(x))
            x = torch.cat([b1,b2],dim=1)
            x = self.global_pool(x)
            x = torch.flatten(x,1)
            x = self.fc(x)
            return x
    df, blocks, stats = auto_partition(CustomNet())