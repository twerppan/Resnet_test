import torch
import torch.nn as nn
import torchvision.models as models

# 加载预训练的 AlexNet 模型
model = models.alexnet(pretrained=True)
print(model)
# 假设我们有一个输入张量，形状为 (1, 3, 224, 224)（batch_size=1, channels=3, height=224, width=224）
input_tensor = torch.randn(1, 3, 224, 224)

# 完整张量的前向传播
output_full = model(input_tensor)
print("完整张量的输出形状:", output_full.shape)

# 假设我们想将张量沿高度方向切割成两个部分
cut_index = 112  # 在高度中间位置切割
tensor_part1 = input_tensor[:, :, :cut_index, :]
tensor_part2 = input_tensor[:, :, cut_index:, :]

print("切割后的张量1形状:", tensor_part1.shape)
print("切割后的张量2形状:", tensor_part2.shape)

# 修改网络以处理切割后的张量
# 我们需要修改 AlexNet 的第一个卷积层，因为它期望输入高度和宽度为 224
# 这里我们直接使用原网络，但在实际应用中可能需要调整网络结构或使用自定义网络

# 前向传播切割后的张量
output_part1 = model(tensor_part1)
output_part2 = model(tensor_part2)

print("切割后张量1的输出形状:", output_part1.shape)
print("切割后张量2的输出形状:", output_part2.shape)

# 合并结果
# 在实际应用中，可能需要根据具体任务调整合并方式
# 这里我们简单地将两个输出在特征维度上拼接
combined_output = torch.cat([output_part1, output_part2], dim=1)

print("合并后的输出形状:", combined_output.shape)

# 如果需要，可以添加额外的全连接层来处理合并后的特征
class CombinedModel(nn.Module):
    def __init__(self, base_model):
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(combined_output.shape[1], 1000)  # 假设输出类别数为 1000

    def forward(self, x1, x2):
        out1 = self.base_model(x1)
        out2 = self.base_model(x2)
        combined = torch.cat([out1, out2], dim=1)
        out = self.fc(combined)
        return out

combined_model = CombinedModel(model)
final_output = combined_model(tensor_part1, tensor_part2)
print("最终输出形状:", final_output.shape)