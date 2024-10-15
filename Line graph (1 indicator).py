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
file_path = r'F:\desktop\折线图.xlsx'
data = pd.read_excel(file_path)

# 提取数据
x = pd.to_datetime(data.iloc[:, 0].values)  # 转换为时间序列
y1 = data.iloc[:, 1].values

# 将时间序列转换为数字
x_num = x.astype(np.int64) / 10**9  # 转换为秒

# 生成平滑数据
x_new_num = np.linspace(x_num.min(), x_num.max(), 300)
spl1 = make_interp_spline(x_num, y1, k=3)
y1_smooth = spl1(x_new_num)

# 将平滑后的数字时间转换回时间序列
x_new = pd.to_datetime(x_new_num * 10**9)

# 创建图形
plt.figure(figsize=(11, 2))

# 绘制平滑折线图
plt.plot(x_new, y1_smooth, color='#252B80', linestyle='-', linewidth=2)

# 填充折线下方的区域
plt.fill_between(x_new, y1_smooth, color='#252B8015', alpha=0.3)

# 设置坐标轴及标注的字体大小
#plt.xlabel('Time Series', fontsize=20)
#plt.ylabel('Flowout (m$^3$/s)', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# 不显示网格线
plt.grid(False)
# 设置纵坐标范围
#plt.ylim(0, 200)
# 调整布局
plt.tight_layout(pad=1.0, w_pad=0.8, h_pad=0.8)

# 保存图形
plt.savefig(r'F:\desktop\折线10.png', dpi=600)

# 显示图形
plt.show()
