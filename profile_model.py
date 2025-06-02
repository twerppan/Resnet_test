import torch
import torch.profiler as profiler
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter
# 检查设备支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 AlexNet 模型
model = resnet50()
model = model.to(device)
model.eval()  # 切换到评估模式
writer = SummaryWriter(log_dir='./log/logResnet50')
# 定义输入数据
input_data = torch.randn(1, 3, 224, 224).to(device)  # 1张3通道的224x224图像
with torch.no_grad():
    writer.add_graph(model, input_data)
writer.close()
# 配置并使用 Profiler
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
    schedule=torch.profiler.schedule(
        wait=1,  # 前1步不采样
        warmup=1,  # 第2步作为热身，不计入结果
        active=3,  # 采集后面3步的性能数据
        repeat=2),  # 重复2轮
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/logResnet50'),  # 保存日志以供 TensorBoard 可视化
    record_shapes=True,  # 记录输入张量的形状
    profile_memory=True,  # 分析内存分配
    with_flops=True,
    with_stack=True  # 记录操作的调用堆栈信息
) as prof:

    for step in range(10):
        with torch.no_grad():
            output = model(input_data)
        prof.step()  # 更新 profiler 的步骤

# 打印 Profiler 生成的报告
print(prof.key_averages(group_by_input_shape=True).table(sort_by='cpu_time_total'))