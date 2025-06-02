import os
import io
import socket
import torch
import time
from clip import ResNet50_Part3

# 定义模型的第三部分
#
def process_and_forward(data_buffer):
    data_buffer.seek(0)
    output_part2 = torch.load(data_buffer,weights_only=True)

    model_part3 = ResNet50_Part3()  # 实例化模型
    start_load = time.perf_counter()
    model_part3.load_state_dict(torch.load('resnet50_part3.pth',weights_only=True))
    end_load = time.perf_counter()
    print(f"树莓派加载模型第三部分时间: {end_load - start_load:.6f}秒")
    model_part3.eval()
    with torch.no_grad():
        output_part3 = model_part3(output_part2)  # 调用 forward 方法
        print(output_part3.shape)
        torch.save(output_part3, "output_part2.pt")
        print("树莓派5成功接收数据")
# 修改后的设备3处理逻辑
with socket.socket() as s:
    s.bind(('0.0.0.0', 8000))
    s.listen()

    while True:
        conn, addr = s.accept()
        try:
            header = conn.recv(4)
            path_length = int.from_bytes(header, "big")
            save_path = conn.recv(path_length).decode().strip()
            # 安全验证
            allowed_dir = "/home/pycharm_project_5"
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
            # 发送确认消息
            conn.sendall(b'received')
            print(f"收到来自{addr}的数据，大小: {data_buffer.getbuffer().nbytes}字节")
            process_and_forward(data_buffer)
            conn.close()

        except Exception as e:
            print(f"处理错误: {str(e)}")
        finally:
            conn.close()