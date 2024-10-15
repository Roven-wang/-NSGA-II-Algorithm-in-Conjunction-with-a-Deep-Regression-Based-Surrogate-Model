from DNN_NSE import DNN1
from DNN_BR2 import DNN2
from DNN_D import DNN3
import torch

# 加载模型NSE
model1 = DNN1().to('cuda:0')
model1.load_state_dict(torch.load(r'H:\python\54parameter\DNN-NSE.pth'))
model1.eval()  # 设置为评估模式


# 加载模型BR2
model2 = DNN2().to('cuda:0')
model2.load_state_dict(torch.load(r"H:\python\54parameter\DNN-BR2.pth"))
model2.eval()

# 加载模型D
model3 = DNN3().to('cuda:0')
model3.load_state_dict(torch.load(r"H:\python\54parameter\DNN-D.pth"))
model3.eval()

def f1(x):
    # 将特征向量转换为张量
    input_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to('cuda:0')
    # 使用模型进行预测
    with torch.no_grad():
        output = model1(input_tensor)
    value = output.item()
    if value < 1:
        return -value

def f2(x):
    # 将特征向量转换为张量
    input_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to('cuda:0')
    # 使用模型进行预测
    with torch.no_grad():
        output = model2(input_tensor)
    value = output.item()
    if value > 0 and value < 1:
        return -value


def f3(x):
    # 将特征向量转换为张量
    input_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to('cuda:0')
    # 使用模型进行预测
    with torch.no_grad():
        output = model3(input_tensor)
    value = output.item()
    if value > 0 and value < 1:
        return -value
