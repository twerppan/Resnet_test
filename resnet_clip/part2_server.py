import torch
import socket
import os
import io

import time
from clip import ResNet50_Part2

def process_and_forward(data_buffer):
    data_buffer.seek(0)
    output_part1 = torch.load(data_buffer)

    model_part2 = ResNet50_Part2()  # 实例化模型
    start_load = time.perf_counter()
    model_part2.load_state_dict(torch.load('resnet50_part2.pth'))
    end_load = time.perf_counter()
    print(f"nano加载模型第二部分时间: {end_load - start_load:.6f}秒")
    model_part2.eval()
    with torch.no_grad():
        output_part2 = model_part2(output_part1)  # 调用 forward 方法
        print(output_part2.shape)
        torch.save(output_part2, "output_part1.pt")
    # 转发到设备3
    with socket.socket() as s:
        s.connect(('10.10.65.79', 8000))
        buffer = io.BytesIO()
        torch.save(output_part2, buffer)

        buffer.seek(0)
        data = buffer.getvalue()
        # 发送元数据
        path = "/home/pycharm_project_5/output_part1.pt"
        path_bytes = path.encode("utf-8")
        header = len(path_bytes).to_bytes(4, "big")
        s.sendall(header + path_bytes)

        start_send = time.perf_counter()
        s.sendall(data)
        end_send = time.perf_counter()
        send_time = end_send - start_send
        print(f"nano发送数据到树莓派时间: {send_time:.6f}秒")
        # 计算传输带宽
        data_size = len(data)  # 数据大小（字节）
        # 等待接收端确认接收完成

        ack = conn.recv(1024)
        if ack == b'received':
            # 计算传输带宽
            bandwidth = data_size / send_time  # 字节每秒
            print(f"传输带宽: {bandwidth / 1024 / 1024:.6f} MB/s")  # 转换为 MB/s

        print("nano发送数据到树莓派5成功")


with socket.socket() as s:
    s.bind(('0.0.0.0', 2222))
    s.listen()

    while True:
        conn, addr = s.accept()
        try:
            # 接收元数据
            header = conn.recv(4)
            path_length = int.from_bytes(header, "big")
            save_path = conn.recv(path_length).decode().strip()

            # 安全验证
            allowed_dir = "/home/pycharm_project"
            abs_path = os.path.abspath(save_path)
            if not abs_path.startswith(allowed_dir):
                conn.close()
                raise ValueError(f"非法路径: {abs_path}")

            # 内存接收数据
            data_buffer = io.BytesIO()
            while True:
                chunk = conn.recv(4096)
                if not chunk: break
                data_buffer.write(chunk)

            print(f"收到来自{addr}的数据，大小: {data_buffer.getbuffer().nbytes}字节")
            # 发送确认消息
            conn.sendall(b'received')
            # 触发处理流程
            process_and_forward(data_buffer)
            conn.close()
        except Exception as e:
            print(f"处理错误: {str(e)}")
        finally:
            conn.close()
