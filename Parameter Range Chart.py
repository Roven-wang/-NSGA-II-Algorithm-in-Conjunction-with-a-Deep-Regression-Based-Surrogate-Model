import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties

# 设置英文字体为Times New Roman或其他支持英文的字体
font_path = r'C:\Windows\Fonts\times.ttf'  # Times New Roman的字体文件路径
font = FontProperties(fname=font_path, size=20)
plt.rcParams['font.family'] = font.get_name()

# 读取Excel文件
file_path = r'F:\desktop\参数范围图2.xlsx'
df = pd.read_excel(file_path)

# 获取所有参数和指标
parameters = df['参数'].unique()
indicators = df['指标'].unique()

# 自定义颜色集
custom_colors = ['#7D1315', '#EB1D22', '#EF5E21', '#FBCD11', '#BDD638', '#81C998', '#48C6EB', '#3C6DB4', '#3752A4', '#252B80']

# 创建图表
fig, ax = plt.subplots(figsize=(12, 11))
fig.patch.set_facecolor('white')  # 设置背景颜色为白色
ax.set_facecolor('white')  # 设置背景颜色为白色

# 为每个指标分配一种颜色
colors = plt.cm.tab10(range(len(indicators)))

# 绘制每个参数的范围和最佳值
for i, (indicator, color) in enumerate(zip(indicators, custom_colors)):
    ind_data = df[df['指标'] == indicator]

    for j, parameter in enumerate(parameters):
        param_data = ind_data[ind_data['参数'] == parameter]

        if not param_data.empty:
            # 获取上下限和最佳值
            upper = param_data['上限'].values[0]
            lower = param_data['下限'].values[0]
            best = param_data['最佳值'].values[0]

            # 在图表上绘制线段和点
            ax.plot([lower, upper], [j + i * 0.05, j + i * 0.05], color=color, linewidth=25)  # 调整距离
            ax.plot(best, j + i * 0.05, '|', color='#E4E4E4', markersize=25, markeredgewidth=5)  # 调整距离

# 设置轴标签和标题
ax.set_yticks([])  # 删除纵坐标标注
ax.set_xlabel('r__CH_S1.sub', fontsize=32, labelpad=12)


# 添加图例
'''legend_entries = [mlines.Line2D([], [], color=color, marker='o', markersize=5, label=indicator) for indicator, color in zip(indicators, colors)]
ax.legend(handles=legend_entries, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')'''

# 确保横坐标标注完整
'''ax.set_xlim(df['下限'].min(), df['上限'].max())'''
ax.set_xlim(-2, 3)

# 调整坐标轴上数字的大小
ax.tick_params(axis='x', labelsize=32)

# 显示图表
#plt.tight_layout()
# 保存图像
#plt.savefig(r'F:\desktop\参数范围图15.png', dpi=1000)
plt.show()


# 创建图例图表
fig_legend, ax_legend = plt.subplots(figsize=(16, 6))
fig_legend.patch.set_facecolor('white')  # 设置背景颜色为白色
ax_legend.axis('off')  # 不显示坐标轴

# 添加图例
legend_entries = [mlines.Line2D([], [], color=color, marker='|', markersize=5, linewidth=25,label=indicator) for indicator, color in zip(indicators, custom_colors)]
legend = ax_legend.legend(handles=legend_entries, loc='lower left', ncol=3, fontsize=45)

# 删除图例的边框
legend.get_frame().set_edgecolor('white')

# 保存图例到桌面
legend_output_path = r'F:\desktop\图例.png'
plt.tight_layout()
plt.savefig(legend_output_path, dpi=1000)
plt.show()