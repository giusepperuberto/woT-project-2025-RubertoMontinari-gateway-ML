from turtle import st

import daily as daily
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error



# === 1. Carica il CSV ===
df = pd.read_csv("file_ripulito.csv", parse_dates=["datetime"])

# === 2. Estrai solo la data (giorno) ===
df["date"] = df["datetime"].dt.date

# === 3. Calcola energia consumata per ogni minuto ===
# EnergyConsumption è potenza in Watt istantanea → converti in kWh
df["energia_kwh"] = df["EnergyConsumption"] / 1000  # ogni ora

# === 4. Aggrega per giorno ===
daily = df.groupby("date").agg({
    "energia_kwh": "sum",              # target
    "Temperature": ["mean", "min", "max"],
    "Humidity": ["mean", "min", "max"],
    "day_of_week": "first",            # categoriale
    "is_weekend": "first",
    "month": "first",
    "week_of_year": "first"
}).reset_index()

# === 5. Rinomina colonne ===
daily.columns = [
    "date", "consumo_kwh",
    "temp_mean", "temp_min", "temp_max",
    "hum_mean", "hum_min", "hum_max",
    "day_of_week", "is_weekend", "month", "week_of_year"
]

# === 6. Costruisci dataset ML ===
X = daily.drop(columns=["date", "consumo_kwh"])
y = daily["consumo_kwh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Dimensioni Training Set: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Dimensioni Test Set: X_test={X_test.shape}, y_test={y_test.shape}")

# Funzione di valutazione
def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"\n📊 {name}")
    print(f" R² = {r2: .4f}")
    print(f" MSE = {mse: .2f}")
    print(f" RMSE = {rmse: .2f}")
    print(f" MAPE = {mape: .2f}")

    return r2, mse, rmse

# 3. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_r2, rf_mse, rf_rmse = evaluate("Random Forest", y_test, rf_preds)

# 4. XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_r2, xgb_mse, xgb_rmse = evaluate("XGBoost", y_test, xgb_preds)


# 5. CatBoost
cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=5,
                              verbose=0, random_state=42)
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_test)
cat_r2, cat_mse, cat_rmse = evaluate("CatBoost", y_test, cat_preds)

# 7. Seleziona il migliore
models_results = {
    "RandomForest": (rf_r2, rf),
    "XGBoost": (xgb_r2, xgb_model),
    "CatBoost": (cat_r2, cat_model)
}

best_name, (best_r2, best_model) = max(models_results.items(), key=lambda x: x[1][0])

# 8. Salva il modello e il test set
joblib.dump(best_model, "rf_energy_model.joblib")
joblib.dump((X_test, y_test), "test_data.joblib")

print(f"\n✅ Miglior modello: {best_name} (salvato come 'rf_energy_model.joblib')")




###########################
