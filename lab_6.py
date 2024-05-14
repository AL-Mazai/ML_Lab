import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 载入数据
df = pd.read_csv('dataset/实验6-data/KaggleCredit2.csv')

features = df.columns[1:]
# 绘制各特征的频次直方图或条形图
for feature in features:
    plt.figure(figsize=(10, 6))

    unique_values = df[feature].nunique()

    if unique_values <= 2:  # 处理二元数据
        value_counts = df[feature].value_counts().sort_index()
        plt.bar(value_counts.index, value_counts.values, tick_label=value_counts.index)
        plt.title(f'Bar Chart of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    else:  # 处理连续数据
        # 计算数据范围
        data_min = df[feature].min()
        data_max = df[feature].max()

        # 直接使用原始数据绘图，不过滤极端值
        plt.hist(df[feature], bins=50, edgecolor='k')

        # 设置横坐标的范围和刻度
        plt.xlim([data_min, data_max])
        num_ticks = 10  # 设置刻度数量
        tick_step = (data_max - data_min) / num_ticks if (data_max - data_min) > 0 else 1
        plt.xticks(np.arange(data_min, data_max + tick_step, tick_step), rotation=45)

        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

    plt.grid(True)  # 开启网格线，方便查看
    plt.tight_layout()
    plt.show()

# 查看数据基本情况
print(df.info())
print(df.describe())
# 检查缺失值
print(df.isnull().sum())
# 删除缺失较多的行
df = df.dropna()

# 准备数据
X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']
# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 逻辑回归模型
log_reg = LogisticRegression()
log_reg.fit(X_scaled, y)
# 获取特征权重
feature_importance = abs(log_reg.coef_[0])
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)
# 找到权重最大的特征
top_feature = feature_importance_df.iloc[0]['Feature']
print(f"最显著的特征是: {top_feature}")


# 使用最显著的特征
X_top_feature = df[[top_feature]].values
# 数据切分，测试集占比30%
X_train, X_test, y_train, y_test = train_test_split(X_top_feature, y, test_size=0.3, random_state=0)
# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 构建SVM模型
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
# 进行预测
y_pred = svm_model.predict(X_test_scaled)

# 结果评估
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

