import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Problem A
# 读取数据集
df = pd.read_csv('dataset/Advertising.csv')

# 绘制 TV、Radio 和 Newspaper 与销售额的散点图
plt.figure(figsize=(10, 8))
plt.scatter(df['TV'], df['Sales'], color='blue', label='TV')
plt.scatter(df['Radio'], df['Sales'], color='green', label='Radio')
plt.scatter(df['Newspaper'], df['Sales'], color='red', label='Newspaper')
plt.xlabel('Advertising Cost')
plt.ylabel('Sales')
plt.title('Advertising Cost——Sales')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()

# 绘制三个子图
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
axs[0].scatter(df['TV'], df['Sales'], color='blue', label='TV')
axs[0].set_title('TV——Sales')
axs[1].scatter(df['Radio'], df['Sales'], color='green', label='Radio')
axs[1].set_title('Radio——Sales')
axs[2].scatter(df['Newspaper'], df['Sales'], color='red', label='Newspaper')
axs[2].set_title('Newspaper——Sales')
for ax in axs:
    ax.set_xlabel('Advertising Cost')
    ax.set_ylabel('Sales')
    ax.grid(True, which='both', linestyle='--')
fig.tight_layout()
plt.show()

# 分割数据集
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 封装Pipeline管道，便于灵活调整多项式回归模型参数
def kk(degree):
    return Pipeline(
        [('poly', PolynomialFeatures(degree=degree)),
         ('std_scaler', StandardScaler()),
         ('lin_reg', LinearRegression())])


# 建立和评估不同多项式次数的线性回归模型
models = []
for degree in range(1, 10):
    model = kk(degree)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    models.append(model)
    # 绘制预测结果与实际销售额的图
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(y_test)), y_pred, color='green', label=f'Predicted (Degree {degree})')
    plt.plot(range(len(y_test)), y_test, color='red', label='Actual')
    plt.title(f'Sales Prediction (Degree {degree})')
    plt.xlabel('Sample Index')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()


# Problem B
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 读取数据集，指定没有表头
df = pd.read_csv('dataset/iris.csv', header=None)

# 数据预处理：分割训练集和测试集
X = df.iloc[:, :-1]  # 选择除了最后一列以外的所有列作为特征
y = df.iloc[:, -1]   # 选择最后一列作为标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 多项式次数从1到9的训练和评估
best_score = 0
best_degree = []
for degree in range(1, 10):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # 逻辑回归
    model = LogisticRegression(solver='liblinear', multi_class='ovr')
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    score = accuracy_score(y_test, y_pred)
    print(f'Degree: {degree}, Accuracy: {score}')

    # 保存最优预测时的准确率和多项式次数
    if score >= best_score:
        best_score = score
        best_degree.append(degree)

print(f'多项式次数为: {best_degree} 时预测效果最优，准确率为： {best_score}')
