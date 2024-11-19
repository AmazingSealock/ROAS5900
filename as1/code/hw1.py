import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载数据
# 假设数据是空格分隔
columns = ["index", "length_of_stay", "age", "med_school", "avg_daily_census", "other1", "other2", "other3", "other4", "other5", "other6", "other7"]
data = pd.read_csv("SENIC.txt", sep="\s+", header=None, names=columns)

# 提取需要的变量
df = data[["length_of_stay", "age", "med_school", "avg_daily_census"]]

# 检查数据
print(df.info())

# 2. 划分训练集和测试集
X = df[["age", "med_school", "avg_daily_census"]]
y = df["length_of_stay"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X_train, y_train)

# 4. 预测
y_pred = model.predict(X_test)

# 5. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# 输出结果
print("回归系数:", model.coef_)
print("截距:", model.intercept_)
print("均方误差 (MSE):", mse)
print("R² 得分:", r2)

# 6. 输出模型系数对应解释
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print(coefficients)
