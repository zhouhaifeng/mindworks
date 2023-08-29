import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset,DataLoader,SequentialSampler

class model(nn.Module):
	def __init__(self,input_dim,hidden_dim,output_dim):
		super(model,self).__init__()
		self.layer1 = nn.LSTM(input_dim,hidden_dim,output_dim,batch_first = True)
		self.layer2 = nn.Linear(hidden_dim,output_dim)
	def forward(self,inputs):
		layer1_output,layer1_hidden = self.layer1(inputs)
		layer2_output = self.layer2(layer1_output)
		layer2_output = layer2_output[:,-1,:]#取出一个batch中每个句子最后一个单词的输出向量即该句子的语义向量！！！！！！！!！
		return layer2_output

#建立小模型
model_student = model(input_dim = 2,hidden_dim = 8,output_dim = 4)

#建立大模型（此处仍然使用LSTM代替，可以使用训练好的BERT等复杂模型）
model_teacher = model(input_dim = 2,hidden_dim = 16,output_dim = 4)

#设置输入数据，此处只使用随机生成的数据代替
inputs = torch.randn(4,6,2)
true_label = torch.tensor([0,1,0,0])

#生成dataset
dataset = TensorDataset(inputs,true_label)

#生成dataloader
sampler = SequentialSampler(inputs)
dataloader = DataLoader(dataset = dataset,sampler = sampler,batch_size = 2)

loss_fun = CrossEntropyLoss()
criterion  = nn.KLDivLoss()#KL散度
optimizer = torch.optim.SGD(model_student.parameters(),lr = 0.1,momentum = 0.9)#优化器，优化器中只传入了学生模型的参数，因此此处只对学生模型进行参数更新，正好实现了教师模型参数不更新的目的

for step,batch in enumerate(dataloader):
	inputs = batch[0]
	labels = batch[1]
	
	#分别使用学生模型和教师模型对输入数据进行计算
	output_student = model_student(inputs)
	output_teacher = model_teacher(inputs)
	
	#计算学生模型和真实标签之间的交叉熵损失函数值
	loss_hard = loss_fun(output_student,labels)
	
	#计算学生模型预测结果和教师模型预测结果之间的KL散度
	loss_soft = criterion(output_student,output_teacher)
	
	loss = 0.9*loss_soft + 0.1*loss_hard
	print(loss)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
