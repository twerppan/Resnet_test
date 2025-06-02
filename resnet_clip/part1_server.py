import torch
import socket
import io
import time
from clip import ResNet50_Part1
# 定义模型的第一部分

# 加载模型和数据
model_part1 = ResNet50_Part1()
start_load = time.perf_counter()
model_part1.load_state_dict(torch.load('resnet50_part1.pth',weights_only=False))  # 加载预训练的模型参数
end_load = time.perf_counter()
print(f"本地加载模型第一部分时间: {end_load - start_load:.6f}秒")
model_part1.eval()

input_data = torch.randn(1, 3, 224, 224)  # 示例输入数据

# 前向传播计算
with torch.no_grad():
    output_part1 = model_part1(input_data)
    print(output_part1.shape)
with socket.socket() as s:
    s.connect(('10.10.66.189', 2222))
    buffer = io.BytesIO()
    torch.save(output_part1, buffer)
    buffer.seek(0)
    data = buffer.getvalue()

    # 发送路径长度 + 路径内容
    path = "/home/pycharm_project/output_part1.pt"
    path_bytes = path.encode("utf-8")
    header = len(path_bytes).to_bytes(4, byteorder="big")
    s.sendall(header + path_bytes)

    # 发送张量数据
    start_send = time.perf_counter()
    s.sendall(data)
    end_send = time.perf_counter()
    send_time = end_send - start_send
    print(f"本地发送数据到nano时间: {send_time:.6f}秒")
    ack = s.recv(1024)
    if ack == b'received':
        # 计算传输带宽
        data_size = len(data)  # 数据大小（字节）
        bandwidth = data_size / send_time  # 字节每秒
        print(f"本地与nano传输带宽: {bandwidth / 1024 / 1024:.6f} MB/s")  # 转换为 MB/s
        print("本地发送数据到nano成功")
    else:
        print("接收端未确认接收完成")