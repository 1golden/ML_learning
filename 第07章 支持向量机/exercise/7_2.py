import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 数据点
X = np.array([
    [1, 2], [2, 3], [3, 3],  # 正例点
    [2, 1], [3, 2]  # 负例点
])
y = np.array([1, 1, 1, -1, -1])  # 标签

# 创建SVM模型并训练
model = SVC(kernel='linear', C=1e5)  # 使用线性核函数 # C-惩罚参数，用于控制软间隔的大小。一个很大的C值表示惩罚误分类的成本很高，从而趋向于硬间隔
model.fit(X, y)

# 获取分离超平面的参数
w = model.coef_[0]
b = model.intercept_[0]
print(f'权重向量: {w}, 偏置: {b}')

# 支持向量
support_vectors = model.support_vectors_

# 创建一个绘图
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='k', s=100,
            label='Support Vectors')

# 绘制分离超平面
x_plot = np.linspace(0, 4, 100)
y_plot = -(w[0] * x_plot + b) / w[1]
plt.plot(x_plot, y_plot, 'k-', label='Separating Hyperplane')

k = - w[0] / w[1]  # 斜率

# 绘制间隔边界
b = model.support_vectors_[0]
y_margin1 = k * x_plot - k * b[0] + b[1]
b = model.support_vectors_[-1]
y_margin2 = k * x_plot - k * b[0] + b[1]
plt.plot(x_plot, y_margin1, 'k--', label='Margin')
plt.plot(x_plot, y_margin2, 'k--')

# 图例和标签
plt.legend()
plt.xlabel('$x^{(1)}$')
plt.ylabel('$x^{(2)}$')
plt.title('SVM with Maximum Margin')
plt.show()
