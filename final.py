# 定义模型层及其计算量
layers = [
    {"flops": f, "shape": sh}
    for (f, sh) in [
        (100,(512,512)), (120,(512,512)), (130,(512,512)),
        (200,(1024,1024)), (50,(512,512)),
        (60,(512,512)), (80,(512,512)), (500,(2048,2048)),
        (90,(512,512)), (100,(512,512)), (110,(512,512)),
        (250,(1024,1024)), (40,(512,512)), (70,(512,512)),
        (300,(1024,1024)), (60,(512,512)), (70,(512,512))
    ]
]
# 定义设备参数：内存(GB)、计算能力(Flops/ms)、带宽(GB/s)
devices = [
    {"id":0,"mem":2.0,"speed":10.0,"bw":10.0},
    {"id":1,"mem":4.0,"speed":8.0,"bw":10.0},
    {"id":2,"mem":6.0,"speed":6.0,"bw":10.0},
    {"id":3,"mem":4.0,"speed":12.0,"bw":10.0},
    {"id":4,"mem":6.0,"speed":11.0,"bw":10.0},
]
# 识别出的瓶颈段示例：将模型切分为3块
blocks = [
    {"type":"pipeline", "layers":list(range(1,6)),    "devices":[3]},   # 层1-5
    {"type":"tensor",   "layers":list(range(6,12)),   "devices":[4,0]}, # 层6-11
    {"type":"tensor",   "layers":list(range(12,18)),  "devices":[1,2]}, # 层12-17
]
# 计算并输出每个块和设备的负载
print("分块及切分方式：")
for block in blocks:
    flops_total = sum(layers[i-1]["flops"] for i in block["layers"])
    out_shape = layers[block["layers"][-1]-1]["shape"]
    out_bytes = out_shape[0]*out_shape[1]*4
    if block["type"] == "pipeline":
        dev_id = block["devices"][0]
        dev = devices[dev_id]
        time = flops_total / dev["speed"]
        mem_used = sum(layers[i-1]["shape"][0]*layers[i-1]["shape"][1]*4 for i in block["layers"]) / (1024**3)
        comm = out_bytes/(dev["bw"]*1e9)*1000 if blocks.index(block)<len(blocks)-1 else 0
        print(f" Block{block['layers']}: 流水线 on Device{dev_id}, FLOPs={flops_total}M, 时间={time:.2f}ms, "
              f"内存≈{mem_used:.3f}GB, 通信={comm:.3f}ms")
    else:
        dev_ids = block["devices"]
        speed_sum = sum(devices[d]["speed"] for d in dev_ids)
        time = flops_total / speed_sum
        comm = 2 * out_bytes * (len(dev_ids)-1) / len(dev_ids) / (min(devices[d]["bw"] for d in dev_ids)*1e9) * 1000 \
               if blocks.index(block)<len(blocks)-1 else 0
        mem_used = sum(layers[i-1]["shape"][0]*layers[i-1]["shape"][1]*4 for i in block["layers"]) / (1024**3) / len(dev_ids)
        print(f" Block{block['layers']}: 张量 on Dev{dev_ids}, FLOPs={flops_total}M, 时间={time:.2f}ms, "
              f"内存/卡≈{mem_used:.3f}GB, 通信={comm:.3f}ms")
print("\n每个设备负载：")
for dev in devices:
    dev_id = dev["id"]
    flops_load = 0
    mem_load = 0
    for block in blocks:
        if block["type"] == "pipeline" and dev_id in block["devices"]:
            flops_load += sum(layers[i-1]["flops"] for i in block["layers"])
            mem_load += sum(layers[i-1]["shape"][0]*layers[i-1]["shape"][1]*4 for i in block["layers"])
        if block["type"] == "tensor" and dev_id in block["devices"]:
            flops_load += sum(layers[i-1]["flops"] for i in block["layers"]) / len(block["devices"])
            mem_load += sum(layers[i-1]["shape"][0]*layers[i-1]["shape"][1]*4 for i in block["layers"]) / len(block["devices"])
    time = flops_load / dev["speed"]
    mem_load = mem_load / (1024**3)
    print(f" Device{dev_id}: FLOPs={flops_load:.0f}M, 时间={time:.2f}ms, 内存≈{mem_load:.3f}GB")

