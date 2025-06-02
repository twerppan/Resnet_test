import torch
from torchvision.models import resnet50

# 下载 ResNet50 的预训练权重
model = resnet50(weights='IMAGENET1K_V2')

print(f'model: {model}')

traced_model = torch.fx.symbolic_trace(model)
print(f'traced_model: {traced_model}')
graph = traced_model.graph
print(f'graph:\n{graph}')