import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()


# Step 1: Load dataset
df = pd.read_csv("CarPrice_Assignment.csv")

# Step 2: Check the data
print(df.head())
print(df.info())

df

# Step 3: Features & Target
X = df.drop(columns=["price"])  # Independent variables
y = df["price"]                  # Target variable

# Step 4: Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Step 5: Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols)
])

# Step 6: Ridge Regression
ridge_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", Ridge(alpha=1.0))
])
ridge_pipeline.fit(X, y)
ridge_pred = ridge_pipeline.predict(X)

# Step 7: Lasso Regression
lasso_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", Lasso(alpha=0.1,max_iter=10000))
])
lasso_pipeline.fit(X, y)
lasso_pred = lasso_pipeline.predict(X)


# Step 8: Result
print("=== Ridge Regression ===")
print("MSE:", mean_squared_error(y, ridge_pred))
print("R² Score:", r2_score(y, ridge_pred))

print("\n=== Lasso Regression ===")
print("MSE:", mean_squared_error(y, lasso_pred))
print("R² Score:", r2_score(y, lasso_pred))

# Step 9: Plot predictions
plt.figure(figsize=(8, 5))
plt.scatter(y, ridge_pred, alpha=0.5, label="Ridge")
plt.scatter(y, lasso_pred, alpha=0.5, label="Lasso")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Car Price Prediction (Ridge vs Lasso)")
plt.legend()
plt.show()