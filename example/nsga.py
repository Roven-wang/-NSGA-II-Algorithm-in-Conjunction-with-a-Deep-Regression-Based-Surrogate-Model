from nsga2.nsga2.problem import Problem
from nsga2.nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from model import f1,f2,f3
from tqdm import tqdm
from nsga2.nsga2.utils import run_evolution

problem = Problem(num_of_variables=43, objectives=[f1, f2, f3], variables_range=[(0,1)], same_range=True, expand=False)
evo = Evolution(problem, mutation_param=20)
func = [i.objectives for i in evo.evolve()]

#绘图部分
# 设置英文字体为Times New Roman
font_path = r'C:\Windows\Fonts\times.ttf'  # Times New Roman的字体文件路径
font = FontProperties(fname=font_path)

# 绘制三维图
function1 = [i[0] for i in func]
function2 = [i[1] for i in func]
function3 = [i[2] for i in func]

fig = plt.figure(figsize=(8, 8))

# 设置全局字体为特定字体
plt.rcParams['font.family'] = font.get_name()
# 绘制第一个子图，三维图
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.scatter3D(function1, function2, function3, marker='o')
plt.tick_params(axis='both', which='major', labelsize=10)
ax1.set_xlabel('NSE', fontsize=12)
ax1.set_ylabel('BR$^2$', fontsize=12)
ax1.set_zlabel('D', fontsize=12)
# 设置视角
ax1.view_init(elev=30, azim=45)  # 设置观察角度为30度和方位角度为45度
# 绘制第二个子图，二维图
plt.subplot(2, 2, 2)
function1 = [i[0] for i in func]
function2 = [i[1] for i in func]
plt.xlabel('NSE', fontsize=12)
plt.ylabel('BR$^2$', fontsize=12)
plt.scatter(function1, function2)
plt.tick_params(axis='both', which='major', labelsize=10)
# 绘制第三个子图，二维图
plt.subplot(2, 2, 3)
function2 = [i[1] for i in func]
function3 = [i[2] for i in func]
plt.xlabel('BR$^2$', fontsize=12)
plt.ylabel('D', fontsize=12)
plt.scatter(function2, function3)
plt.tick_params(axis='both', which='major', labelsize=10)
# 绘制第四个子图，二维图
plt.subplot(2, 2, 4)
function1 = [i[0] for i in func]
function3 = [i[2] for i in func]
plt.xlabel('NSE', fontsize=12)
plt.ylabel('D', fontsize=12)
plt.scatter(function1, function3)
plt.tick_params(axis='both', which='major', labelsize=10)
# 调整子图之间的间距
plt.tight_layout()
# 显示图形
plt.show()
# 保存图像
plt.savefig('F:/desktop/NSGA2.png', dpi=600)

# 获取帕累托前沿的第一层
pareto_front = evo.evolve()

# 存储变量值和目标值的列表
variables_values = []
objectives_values = []

# 遍历帕累托前沿的第一层上的每个个体
for individual in pareto_front:
    # 获取个体的变量值和目标值
    variables_values.append(individual.features)
    objectives_values.append(individual.objectives)

# 输出变量值和目标值
for i in range(len(variables_values)):
    print("Individual", i+1)
    print("Variables:", variables_values[i])
    print("Objectives:", objectives_values[i])
    print()


# 创建一个DataFrame对象，存储变量值和目标值
df = pd.DataFrame({"Variables": variables_values, "Objectives": objectives_values})

# 将数据写入Excel文件
excel_writer = pd.ExcelWriter("pareto_front.xlsx", engine='openpyxl')
df.to_excel(excel_writer, index=False)
excel_writer.save()

print("Excel文件已保存")

num_of_generations=6
num_of_individuals=500
# 运行演化过程并得到 DataFrame
df = run_evolution(evo, num_of_generations, num_of_individuals, f1, f2, f3)

# 将 DataFrame 中的数据写入 Excel 文件
excel_writer = pd.ExcelWriter("evolution_results.xlsx", engine='openpyxl')
df.to_excel(excel_writer, index=False)
excel_writer.save()

print("Excel文件已保存")

