import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.font_manager import FontProperties

# 设置英文字体为Times New Roman或其他支持英文的字体
font_path = r'C:\Windows\Fonts\times.ttf'  # Times New Roman的字体文件路径
font = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font.get_name()

# 读取Excel文件
file_path = r'F:\desktop\小论文作图\DNN相关图\loss_data3.xlsx'
data = pd.read_excel(file_path)

# 提取数据
x = data.iloc[:, 0].values
y1 = data.iloc[:, 1].values
y2 = data.iloc[:, 2].values

# 生成平滑数据
x_new = np.linspace(x.min(), x.max(), 300)
spl1 = make_interp_spline(x, y1, k=3)
y1_smooth = spl1(x_new)
spl2 = make_interp_spline(x, y2, k=3)
y2_smooth = spl2(x_new)

# 创建图形
plt.figure(figsize=(8, 7))

# 绘制平滑折线图
plt.plot(x_new, y1_smooth, label='Training Loss', color='#257AB6', linestyle='-' , linewidth=3)
plt.plot(x_new, y2_smooth, label='Validation Loss', color='#E47B26', linestyle='-' , linewidth=3)

# 设置坐标轴及标注的字体大小
plt.xlabel('Epochs', fontsize=30)
plt.ylabel('Loss', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# 不显示网格线
plt.grid(False)

# 添加图例
plt.legend(fontsize=32, frameon=False)
plt.tight_layout(pad=1.0, w_pad=0.8, h_pad=0.8)

plt.savefig(r'F:\desktop\折线3.png', dpi=1000)
# 显示图形
plt.show()
