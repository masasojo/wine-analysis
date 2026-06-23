import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# データ読み込み（パスは自分の環境に合わせる）
df = pd.read_csv("data/wine.csv")

# 説明変数（例：alcohol と density）
X = df[["alcohol", "density"]]

# 目的変数
y = df["quality"]

# モデル作成
model = LinearRegression()
model.fit(X, y)

# 予測
y_pred = model.predict(X)

# 結果の図を作成
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Linear Regression: Actual vs Predicted Quality")
plt.grid(True)

# 図を保存（wine-analysis 直下）
plt.savefig("linear_regression_result.png")
plt.close()

print("係数:", model.coef_)
print("切片:", model.intercept_)


