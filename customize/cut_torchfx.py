import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace, Graph, GraphModule

# ============================
# （1）定义原始网络
# ============================
class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一阶段
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool  = nn.MaxPool2d(2)
        # 残差块
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        # 并行分支
        self.br1_conv = nn.Conv2d(16, 8, 1)
        self.br1_bn   = nn.BatchNorm2d(8)
        self.br2_conv = nn.Conv2d(16, 8, 1)
        self.br2_bn   = nn.BatchNorm2d(8)
        # 全局池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc          = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        identity = x
        out = self.relu3(self.bn3(self.conv3(x)))
        out = self.bn4(self.conv4(out))
        x = out + identity           # ← 切点：add op
        x = self.relu4(x)
        b1 = self.br1_bn(self.br1_conv(x))
        b2 = self.br2_bn(self.br2_conv(x))
        x  = torch.cat([b1, b2], dim=1)
        x  = self.global_pool(x)
        x  = torch.flatten(x, 1)
        x  = self.fc(x)
        return x

# ============================
# （2）Trace 出 FX GraphModule
# ============================
model = CustomNet()
example_input = torch.randn(1, 3, 224, 224)
gm = symbolic_trace(model)

# 打印所有节点，确认结构（可选）
print("所有节点：")
for node in gm.graph.nodes:
    print(f"  {node.op:12} | {node.name:15} | {node.target}")

# 自动定位第一个 add 节点
cut_node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and getattr(n.target, "__name__", "") == "add")
cut_name = cut_node.name
print(f"将会在节点 '{cut_name}' 处切分")  # 通常就是 "add"

# ============================
# （3）定义通用切分函数 split_fx
# ============================
def split_fx(gm: GraphModule, cut_name: str):
    nodes = list(gm.graph.nodes)
    # 找到切分节点的索引
    idx = next(i for i, n in enumerate(nodes) if n.name == cut_name)

    # --- 构建前半段 Graph ---
    g1 = Graph()
    env1 = {}
    for n in nodes[:idx+1]:  # 包含切分节点
        env1[n] = g1.node_copy(n, lambda x: env1[x])
    g1.output(env1[nodes[idx]])
    part1 = GraphModule(gm, g1)

    # --- 构建后半段 Graph ---
    g2 = Graph()
    env2 = {}
    # 第一个新输入占位符：即 part1 的输出
    inp = g2.placeholder('x_res')
    env2[nodes[idx]] = inp
    for n in nodes[idx+1:]:
        env2[n] = g2.node_copy(n, lambda x: env2[x])
    g2.output(env2[nodes[-1]])
    part2 = GraphModule(gm, g2)

    return part1, part2

# 切分
part1_fx, part2_fx = split_fx(gm, cut_name)

# ============================
# （4）验证：先跑 Part1，再跑 Part2
# ============================
with torch.no_grad():
    z   = part1_fx(example_input)    # 前半段输出：residual sum
    out = part2_fx(z)                # 后半段输出：最终分类结果

print("part1_fx 输出 shape:", z.shape)   # e.g. [1,16,112,112]
print("part2_fx 输出 shape:", out.shape) # [1,10]
