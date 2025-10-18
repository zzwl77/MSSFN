import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# 生成模拟数据
np.random.seed(0)
x = np.random.uniform(10, 100, 50)  # MMSE得分
y = 0.5 * x + np.random.normal(0, 10, 50)  # 模型预测值

# 计算线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
line = slope * x + intercept

# 创建散点图
plt.figure(figsize=(6, 6))
sns.scatterplot(x, y)

# 添加线性回归线
plt.plot(x, line, 'b', label=f'R = {r_value:.2f}, P = {p_value:.1e}')

# 计算并添加置信区间
sns.regplot(x=x, y=y, scatter=False, ci=95)

# 添加文本和标签
plt.title('Correlation between MMSE and Model Prediction', fontsize=14)
plt.xlabel('MMSE', fontsize=12)
plt.ylabel('Model Prediction', fontsize=12)
plt.legend(facecolor='white')

# 设置图例位置和大小
plt.legend(loc='upper left', fontsize=10)

# 添加网格线
plt.grid(True)

# 显示图形
plt.show()