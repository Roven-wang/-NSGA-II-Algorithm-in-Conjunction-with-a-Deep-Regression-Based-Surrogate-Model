import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置英文字体为Times New Roman或其他支持英文的字体
font_path = r'C:\Windows\Fonts\times.ttf'  # Times New Roman的字体文件路径
font = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font.get_name()

# 从Excel文件中读取数据
file_path = r'F:\desktop\小论文作图\DNN相关图\results_data3.xlsx'  # 请替换为你的文件路径
sheet_name = 'Training Set'  # 请替换为你的工作表名称
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设Excel文件中有两列数据，分别为'NSE'和'BR2'
x = data['Actual']
y = data['Predicted']

# 计算相关系数R²、RMSE和MAE
R2 = np.corrcoef(x, y)[0, 1] ** 2
RMSE = np.sqrt(np.mean((x - y) ** 2))
MAE = np.mean(np.abs(x - y))

# 绘制散点图
plt.figure(figsize=(8, 7))
plt.scatter(x, y, color='#257AB6', s=45, marker='o', edgecolors='none')  # 去掉轮廓颜色
#257AB6蓝色#E47B26橙色
# 绘制y=x的斜直线
plt.plot([x.min(), x.max()], [x.min(), x.max()], color='red', linestyle='-', linewidth=2)

# 标注相关系数R²、RMSE和MAE
plt.text(0.65, 0.25, f'r²: {R2:.2f}\nRMSE: {RMSE:.2f}\nMAE: {MAE:.2f}',
         transform=plt.gca().transAxes, fontsize=30, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.xlabel('Actual', fontsize=32, labelpad=10)  # 调大字体
plt.ylabel('Predicted', fontsize=32, labelpad=10)  # 调大字体
plt.grid(False)  # 使用虚线网格

plt.tick_params(axis='x', pad=3, labelsize=30)
plt.tick_params(axis='y', pad=3, labelsize=30)
plt.tight_layout(pad=1.0, w_pad=0.8, h_pad=0.8)

plt.savefig(r'F:\desktop\Train3.png', dpi=1000)
plt.show()