import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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

# 随机森林分类器
rf_entropy = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=42)
rf_entropy.fit(X_train, y_train)
y_pred_entropy = rf_entropy.predict(X_test)
entropy_accuracy = accuracy_score(y_test, y_pred_entropy)
print(f"随机森林（信息熵）测试集准确率: {entropy_accuracy}")

# 参数网格
param_grid = {
    'max_depth': range(1, 6),
    'n_estimators': range(1, 21)
}

# 网格搜索
grid_search_rf = GridSearchCV(rf_entropy, param_grid, cv=10)
grid_search_rf.fit(X_train, y_train)

# 输出最优参数
print(f"随机森林最优参数: {grid_search_rf.best_params_}")
