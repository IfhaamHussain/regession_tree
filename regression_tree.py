# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv(r"C:\Users\HP\Documents\archive\heart.csv")  

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dt)))
print("Decision Tree R2 Score:", r2_score(y_test, y_pred_dt))

# Visualize Tree
import os
os.makedirs("../images", exist_ok=True)
plt.figure(figsize=(15,10))
plot_tree(dt, filled=True)
plt.savefig("../images/decision_tree.png")
plt.show()
plt.savefig("../images/decision_tree.png")
plt.show()

# Overfitting Control
dt_limited = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_limited.fit(X_train, y_train)

y_pred_limited = dt_limited.predict(X_test)
print("Limited Tree RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_limited)))
# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))
y_pred_rf = rf.predict(X_test)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))

# Feature Importance
importances = rf.feature_importances_
feature_importance = pd.Series(importances, index=X.columns)

feature_importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
# Cross Validation
cv_scores = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42), X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
print("CV RMSE Scores:", rmse_scores)
print("Average CV RMSE:", np.mean(rmse_scores))
cv_scores = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42), X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
print("CV RMSE Scores:", rmse_scores)
print("Average CV RMSE:", np.mean(rmse_scores))