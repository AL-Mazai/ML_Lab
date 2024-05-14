import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# 1. 数据读取与预处理
data = pd.read_csv('dataset/Advertising.csv')
features = data[['TV', 'Radio', 'Newspaper']]
target = data['Sales']

# 数据标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# 建立回归模型
# 回归树模型
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, y_train)
tree_pred = tree_regressor.predict(X_test)

# xgboost模型
xgb_regressor = xgb.XGBRegressor(random_state=42)
xgb_regressor.fit(X_train, y_train)
xgb_pred = xgb_regressor.predict(X_test)

# 线性回归模型
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
linear_pred = linear_regressor.predict(X_test)

# 评估模型性能
tree_mse = mean_squared_error(y_test, tree_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
print(f"XGBoost MSE: {xgb_mse}")
print(f"回归树的MSE: {tree_mse}")
print(f"线性回归的MSE: {linear_mse}")

# 交叉验证找到最优的n_estimators
param_grid = {'n_estimators': range(100, 1001)}
grid_search = GridSearchCV(xgb_regressor, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_n_estimators = grid_search.best_params_['n_estimators']
print(f"最优的n_estimators: {best_n_estimators}")

# 使用最优的n_estimators来训练XGBoost模型
xgb_best_regressor = xgb.XGBRegressor(n_estimators=best_n_estimators, random_state=42)
xgb_best_regressor.fit(X_train, y_train)
xgb_pred = xgb_best_regressor.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_pred)
print(f"XGBoost MSE: {xgb_mse}")
