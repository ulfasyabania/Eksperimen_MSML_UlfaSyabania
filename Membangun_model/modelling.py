import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
DATA_PATH = "CaliforniaHousing_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

# Pisahkan fitur dan target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split ulang untuk memastikan reproducibility (jika ingin, bisa skip jika sudah split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow manual logging
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CaliforniaHousing_RF")

with mlflow.start_run():
    # Model dan training
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Manual logging
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log model
    mlflow.sklearn.log_model(rf, "model")

    print(f"Logged to MLflow. RMSE: {rmse:.4f}, R2: {r2:.4f}")
