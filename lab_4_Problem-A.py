import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# 读取数据集
df = "dataset/实验4-iris.csv"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(df, names=column_names)

# 数据标准化
scaler = StandardScaler()
iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(
    iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

# 类别编码
label_encoder = LabelEncoder()
iris_data['class'] = label_encoder.fit_transform(iris_data['class'])

# 划分数据集
X = iris_data.drop('class', axis=1)
y = iris_data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 决策树分类器 - Gini
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)
gini_accuracy = accuracy_score(y_test, y_pred_gini)
print(f"Gini系数决策树测试集准确率: {gini_accuracy}")

# 决策树分类器 - entropy
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)
entropy_accuracy = accuracy_score(y_test, y_pred_entropy)
print(f"entropy决策树测试集准确率: {entropy_accuracy}")

# 参数网格
param_grid = {
    'max_depth': range(1, 6),
    'max_features': range(1, 5)
}

# Gini
grid_search_gini = GridSearchCV(dt_gini, param_grid, cv=10)
grid_search_gini.fit(X_train, y_train)
print(f"Gini系数决策树最优参数: {grid_search_gini.best_params_}")

# entropy
grid_search_entropy = GridSearchCV(dt_entropy, param_grid, cv=10)
grid_search_entropy.fit(X_train, y_train)
print(f"entropy决策树最优参数: {grid_search_entropy.best_params_}")

# 若通过网格搜索找到了最优参数
best_params_gini = grid_search_gini.best_params_
best_params_entropy = grid_search_entropy.best_params_

# 使用最优参数创建新的决策树模型
dt_gini_optimized = DecisionTreeClassifier(criterion='gini', **best_params_gini, random_state=42)
dt_entropy_optimized = DecisionTreeClassifier(criterion='entropy', **best_params_entropy, random_state=42)

# 评估模型
gini_scores = cross_val_score(dt_gini_optimized, X, y, cv=10)
entropy_scores = cross_val_score(dt_entropy_optimized, X, y, cv=10)

# 输出平均准确率
print(f"Gini系数决策树平均准确率: {gini_scores.mean()}")
print(f"entropy决策树平均准确率: {entropy_scores.mean()}")
