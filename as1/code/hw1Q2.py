import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. 加载数据
columns = ["market_value", "lacquer_coats", "batch", "replicate"]
data = pd.read_csv("CH25PR17.txt", sep="\s+", header=None, names=columns)

# 显示数据基本信息
# print(data.head())
# print(data.info())

# 2. 构建Hassediagram
# 可视化实验设计
sns.catplot(data=data, x="lacquer_coats", y="market_value", hue="batch", kind="point", ci=None)
plt.title("Hassediagram: Effect of Coats and Batches on Market Value")
plt.xlabel("Number of Lacquer Coats")
plt.ylabel("Market Value")
plt.show()

# 3. 方差分析 (ANOVA)
# 定义公式: market_value ~ C(lacquer_coats) + C(batch)
model = ols("market_value ~ C(lacquer_coats) + C(batch)", data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# 输出方差分析结果
print("ANOVA Table:")
print(anova_table)

# 4. 模型残差分析
# 残差提取
residuals = model.resid

# (a) 正态性检验
sns.histplot(residuals, kde=True)
plt.title("Residuals Histogram")
plt.xlabel("Residuals")
plt.show()

# QQ图
sm.qqplot(residuals, line="45")
plt.title("QQ Plot of Residuals")
plt.show()

# (b) 同方差性检验
fitted_values = model.fittedvalues
sns.scatterplot(x=fitted_values, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# (c) 独立性检验
dw_stat = sm.stats.durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {dw_stat}")

# 5. 输出模型结果
print("Model Summary:")
print(model.summary())
