import torch
import torchvision

# 加载你的已训练好的PyTorch模型
model = torch.load("/home/zhf/src/ai/mindworks/apps/llama2/models/llama-2-7b-chat.ggmlv3.q8_0.bin")

# 创建一个虚拟的输入张量
dummy_input = torch.randn(1, 3, 224, 224)

# 将模型转换为ONNX格式
onnx_path = "./llama-2-7b-chat.ggmlv3.q8_0.onnx"
torch.onnx.export(model, dummy_input, onnx_path)