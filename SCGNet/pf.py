import torch
from thop import profile
from models.baseline import baseline

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

model = baseline(in_channels=3, num_classes=7).to(dev)

input1 = torch.randn(1, 3, 512, 512).to(dev)  # 模型输入的形状, batch_size=1
input2 = torch.randn(1, 3, 512, 512).to(dev)

# 计算模型的 FLOPs 和参数数量
flops, params = profile(model, inputs=(input1, input2))

# 打印结果
print(f"FLOPs: {flops / 1e9:.2f} G, Parameters: {params / 1e6:.2f} M")