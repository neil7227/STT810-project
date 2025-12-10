import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/mnt/user-data/uploads/base_cleaned_data.csv')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('OPS+')
if '背號' in numeric_cols:
    numeric_cols.remove('背號')

X = df[numeric_cols]
y = df['OPS+']

X = X.replace([np.inf, -np.inf], np.nan)
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

print("=" * 60)
print("Ridge Regression Model Results")
print("=" * 60)
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Cross-Validation R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print("=" * 60)

feature_coef = pd.DataFrame({
    'Feature': numeric_cols,
    'Coefficient': model.coef_
})
feature_coef = feature_coef.sort_values('Coefficient', key=abs, ascending=False)
print("\nFeature Coefficients:")
print(feature_coef.to_string(index=False))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual OPS+')
plt.ylabel('Predicted OPS+')
plt.title(f'Ridge Regression: Predictions vs Actual (R²={r2:.4f})')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted OPS+')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/ridge_regression_results.png', dpi=300, bbox_inches='tight')
print("\nPlot saved: ridge_regression_results.png")

prediction_results = pd.DataFrame({
    'Actual_OPS+': y_test.values,
    'Predicted_OPS+': y_pred,
    'Residual': y_test.values - y_pred
})
prediction_results.to_csv('/mnt/user-data/outputs/ridge_predictions.csv', index=False)
print("Predictions saved: ridge_predictions.csv")
