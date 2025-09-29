import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Importa le pagine esterne
from data_preprocessing import clean_dataset
from forecast_dashboard import forecast_dashboard_page
from dati_home import home_data_page
from streamlit_historical_data import historical_data_page
from predict_from_api import predict_from_api_page



# --- Pulizia e caricamento dataset ---
clean_dataset()
data = pd.read_csv('file_ripulito.csv')

# --- Feature engineering ---
data['Weekend'] = data['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
numeric_features = ['Temperature', 'Humidity', 'hour', 'day_of_week', 'month', 'day_of_year', 'week_of_year', 'is_weekend']
target = 'EnergyConsumption'

# --- Carica modello e dati test ---
model = None
X_test = None
try:
    model = joblib.load("rf_energy_model.joblib")
except FileNotFoundError:
    st.error("Modello non trovato. Assicurati che 'rf_energy_model.joblib' sia stato creato.")
    st.stop()

# Carica dati test
try:
    X_test, y_test = joblib.load("test_data.joblib")
except FileNotFoundError:
    st.error("Dati di test non trovati. Assicurati che 'test_data.joblib' sia presente.")
    st.stop()



X = data[numeric_features]
y = data[target]


y_pred = model.predict(X_test)


# Sidebar menu
st.sidebar.title("Menu")
page = st.sidebar.radio("Vai a:", ["Analisi Dati", "Predizione Manuale", "Predizione da Meteo API",
                                   "Dashboard Previsioni",
                                   "Dati Abitazione"])

if model is None and page in ["Analisi Dati", "Predizione Manuale"]:
    st.error("Modello non trovato. Assicurati che 'rf_energy_model.joblib' sia stato creato.")
    st.stop()

# === Pagina: Analisi Dati ===
if page == "Analisi Dati":
    st.title("Analisi Esplorativa dei Dati")

    # Boxplot Weekday vs Weekend
    st.subheader("Consumo Energetico: Weekday vs Weekend")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x='Weekend', y='EnergyConsumption', data=data, ax=ax1)
    st.pyplot(fig1)

    # Scatter Temperature vs Energy Consumption
    st.subheader("Relazione tra Temperatura e Consumo Energetico")
    fig2, ax2 = plt.subplots()
    # sns.scatterplot(x='Temperature', y='EnergyConsumption', data=data, ax=ax2)
    sns.scatterplot(x='Temperature', y='EnergyConsumption', data=data, ax=ax2, alpha=0.3)
    sns.regplot(x='Temperature', y='EnergyConsumption', data=data, ax=ax2, scatter=False, color='red')
    st.pyplot(fig2)

    # Umidità vs Consumo Energetico
    st.subheader("Relazione tra Umidità e Consumo Energetico")
    fig3, ax3 = plt.subplots()
    # sns.scatterplot(x="Humidity", y="EnergyConsumption", data=data, ax=ax3)
    sns.scatterplot(x="Humidity", y="EnergyConsumption", data=data, ax=ax3, alpha=0.3)
    sns.regplot(x='Humidity', y='EnergyConsumption', data=data, ax=ax3, scatter=False, color='red')
    ax3.set_title("Umidità vs Consumo Energetico")
    st.pyplot(fig3)

    # Predizione sullo stesso dataset
    y_pred = model.predict(X_test)

    # Importanza delle variabili
    st.subheader("Importanza delle Variabili")
    importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)


    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=importance, ax=ax_imp)
    st.pyplot(fig_imp)

    # Valori Reali vs Predetti
    st.subheader("Valori Reali vs Predetti")
    fig4, ax4 = plt.subplots()
    # sns.scatterplot(x=y, y=y_pred, ax=ax4)
    sns.scatterplot(x=y_test, y=y_pred, ax=ax4)
    ax4.set_xlabel("Reale")
    ax4.set_ylabel("Predetto")
    st.pyplot(fig4)

    # Metriche
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("###Metriche del Modello")
    rmse = mse ** 0.5
    st.write(f"**R² (R-squared):** {r2:.4f}")
    st.write(f"**MSE (Mean Squared Error):** {mse:.2f}")
    st.write(f"**RMSE (Mean Squared Error):** {rmse:.2f}")

    # Distribuzione Reale vs Predetto
    st.subheader("Distribuzione: Reale vs Predetto")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
    # sns.histplot(y, label='Reale', kde=True, ax=ax_dist)
    # sns.histplot(y_pred, label='Predetto', kde=True, ax=ax_dist)
    sns.histplot(y_test, label='Reale', kde=True, ax=ax_dist)
    sns.histplot(y_pred, label='Predetto', kde=True, ax=ax_dist)
    ax_dist.legend()
    st.pyplot(fig_dist)



    # Andamento orario
    st.subheader("Consumo Energetico nelle Ore del Giorno")
    fig_hour, ax_hour = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='hour', y='EnergyConsumption', data=data, ax=ax_hour, ci=None)
    ax_hour.set_title("Consumo Energetico vs Ora")
    st.pyplot(fig_hour)

    # Confronto linea Reale vs Predetto
    st.subheader("Confronto: Reale vs Predetto")
    fig_cmp, ax_cmp = plt.subplots(figsize=(12, 6))
    # ax_cmp.plot(y.values[:100], label='Reale', marker='o')  # Limita a 100 per performance
    # ax_cmp.plot(y_pred[:100], label='Predetto', marker='x')
    ax_cmp.plot(y_test.values[:100], label='Reale', marker='o')
    ax_cmp.plot(y_pred[:100], label='Predetto', marker='x')
    ax_cmp.set_xlabel("Indice")
    ax_cmp.set_ylabel("Consumo Energetico")
    ax_cmp.legend()
    st.pyplot(fig_cmp)

    # Errore assoluto (opzionale)
    N = min(len(y_test), len(y_pred), 100)
    error = abs(y_test.values[:N] - y_pred[:N])
    fig_err, ax_err = plt.subplots(figsize=(12, 3))
    ax_err.bar(range(N), error, color='gray')
    ax_err.set_title("Errore Assoluto (|Reale - Predetto|)")
    st.pyplot(fig_err)


# === Pagina: Predizione Manuale ===
elif page == "Predizione Manuale":
    st.title("Predizione Consumo Energetico Manuale")

    st.subheader("Inserisci i dati medi giornalieri per la previsione:")

    # Inserimento dei valori medi giornalieri
    temp_mean = st.number_input("Temperatura media (°C)", value=22.0)
    temp_min = st.number_input("Temperatura minima (°C)", value=18.0)
    temp_max = st.number_input("Temperatura massima (°C)", value=26.0)

    hum_mean = st.number_input("Umidità media (%)", value=50.0)
    hum_min = st.number_input("Umidità minima (%)", value=40.0)
    hum_max = st.number_input("Umidità massima (%)", value=60.0)

    selected_date = st.date_input("Data per la previsione")

    #  Calcolo delle feature temporali dalla data
    selected_date = pd.to_datetime(selected_date)
    day_of_week = selected_date.dayofweek
    is_weekend = int(day_of_week in [5, 6])
    month = selected_date.month
    week_of_year = selected_date.isocalendar().week

    # Costruzione dataframe di input
    input_data = pd.DataFrame([[
        temp_mean, temp_min, temp_max,
        hum_mean, hum_min, hum_max,
        day_of_week, is_weekend,
        month, week_of_year
    ]], columns=[
        'temp_mean', 'temp_min', 'temp_max',
        'hum_mean', 'hum_min', 'hum_max',
        'day_of_week', 'is_weekend',
        'month', 'week_of_year'
    ])

    if st.button("Predici Consumo Giornaliero"):
        prediction = model.predict(input_data)
        st.success(f"📅 Consumo energetico previsto per il {selected_date.date()}: **{prediction[0]:.2f} kWh**")
#####################################################################################################################

# === Altre Pagine ===
elif page == "Predizione da Meteo API":
    predict_from_api_page()

elif page == "Dashboard Previsioni":
    forecast_dashboard_page()

elif page == "Dati Abitazione":
    home_data_page()

###########################################################################################

