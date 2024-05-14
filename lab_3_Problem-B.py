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
