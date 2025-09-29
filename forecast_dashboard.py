import shap
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import io
import joblib
import matplotlib.pyplot as plt

# --- Funzione per prendere dati storici ---
def get_historical_data(start_date, end_date, entity_id):
    url = f"https://gdfhome.duckdns.org/api/history/period/{start_date}?filter_entity_id={entity_id}&end_time={end_date}"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
                         ".eyJpc3MiOiI1MDdmNjliNjA0MmY0M2M1OTk0NjVmZDRiZmZlMWZjNyIsImlhdCI6MTcyNDc1MTMxMywiZXhwIjoyMD"
                         "QwMTExMzEzfQ.S61REKbuTL1l4yP-iIQRDuZvyCmGWDJoL7-FQXAl7xg "
    }
    params = {
        "filter_entity_id": entity_id,
        "end_time": end_date
    }

    response = requests.get(url, headers=headers, params=params, verify=False)
    response.raise_for_status()
    data = response.json()

    flat_data = []
    for entity in data:
        for entry in entity:
            flat_data.append({
                "timestamp": entry["last_changed"],
                "value": entry["state"],
                "entity_id": entry["entity_id"]
            })

    df = pd.DataFrame(flat_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df['timestamp'] = df['timestamp'].dt.floor('S')
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

# --- Funzione aggregazione ---
def analizza_dati_comuni(df_consumo, df_umidita, df_temperatura):
    df_consumo['rounded_timestamp'] = df_consumo['timestamp'].dt.floor('min')
    df_umidita['rounded_timestamp'] = df_umidita['timestamp'].dt.floor('min')
    df_temperatura['rounded_timestamp'] = df_temperatura['timestamp'].dt.floor('min')

    set_common = set(df_consumo['rounded_timestamp']) & set(df_umidita['rounded_timestamp']) & set(df_temperatura['rounded_timestamp'])
    results = []
    for ts in sorted(set_common):
        avg = {
            'timestamp_minuto': ts,
            'consumo_medio': df_consumo[df_consumo['rounded_timestamp'] == ts]['value'].mean(),
            'umidita_media': df_umidita[df_umidita['rounded_timestamp'] == ts]['value'].mean(),
            'temperatura_media': df_temperatura[df_temperatura['rounded_timestamp'] == ts]['value'].mean()
        }
        results.append(avg)
    return pd.DataFrame(results)

# --- Dashboard ---
def forecast_dashboard_page():
    st.title("Previsione Consumo Giornaliero")

    # --- Selezione giorni direttamente nella pagina ---
    num_giorni = st.slider("Seleziona quanti giorni visualizzare (massimo 7)", min_value=1, max_value=7, value=3)
    # Seleziona quanti giorni futuri predire (1-3)
    num_giorni_futuri = st.slider("Seleziona quanti giorni futuri predire", min_value=1, max_value=3, value=1)

    #API_KEY = "3001a620158131c941d011ba003258b3"
    API_KEY="rzwVU2IQ4Lhvk5TBEpfrMAMuMTgOPDYT"
    CITY = "Lecce"
    MODEL_PATH = "rf_energy_model.joblib"

    ENTITIES = {
        "Consumo abitazione (W)": "sensor.quadro_primo_terra_channel_1_power",
        "Temperatura interna (°C)": "sensor.termostat_temperature",
        "Umidità interna (%)": "sensor.termostat_humidity"
    }

    try:
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=num_giorni + 1)
        start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Scarica dati storici
        df_consumo = get_historical_data(start_iso, end_iso, ENTITIES["Consumo abitazione (W)"])
        df_temp = get_historical_data(start_iso, end_iso, ENTITIES["Temperatura interna (°C)"])
        df_umid = get_historical_data(start_iso, end_iso, ENTITIES["Umidità interna (%)"])

        # Calcolo energia
        df_consumo = df_consumo.sort_values('timestamp').reset_index(drop=True)
        df_consumo['delta_t_ore'] = df_consumo['timestamp'].diff().dt.total_seconds() / 3600
        df_consumo['energia_Wh'] = df_consumo['value'].shift() * df_consumo['delta_t_ore']
        df_consumo = df_consumo.dropna(subset=['energia_Wh'])
        df_consumo['data'] = df_consumo['timestamp'].dt.date
        energia_per_giorno = df_consumo.groupby('data')['energia_Wh'].sum().reset_index()
        energia_per_giorno['consumo_kWh'] = energia_per_giorno['energia_Wh'] / 1000
        energia_per_giorno = energia_per_giorno[['data', 'consumo_kWh']]

        # Aggrega per minuto e unisci
        df_agg = analizza_dati_comuni(df_consumo, df_umid, df_temp)
        df_agg = df_agg.dropna()
        df_agg['energia_kwh'] = df_agg['consumo_medio'] / 1000 / 60
        df_agg['data'] = df_agg['timestamp_minuto'].dt.date

        df_daily = df_agg.groupby("data").agg({
            "energia_kwh": "sum",
            "temperatura_media": ["mean", "min", "max"],
            "umidita_media": ["mean", "min", "max"]
        }).reset_index()

        df_daily.columns = ["date", "consumo_kwh",
                            "temp_mean", "temp_min", "temp_max",
                            "hum_mean", "hum_min", "hum_max"]

        df_daily['day_of_week'] = pd.to_datetime(df_daily['date']).dt.dayofweek
        df_daily['is_weekend'] = df_daily['day_of_week'].isin([5, 6]).astype(int)
        df_daily['month'] = pd.to_datetime(df_daily['date']).dt.month
        df_daily['week_of_year'] = pd.to_datetime(df_daily['date']).dt.isocalendar().week.astype(int)

        giorni_disponibili = df_daily['date'].sort_values().unique()
        giorni_recenti = giorni_disponibili[-num_giorni:]

        df_recenti = df_daily[df_daily['date'].isin(giorni_recenti)].copy()

        # Carica modello e predici
        modello = joblib.load(MODEL_PATH)

        feature_cols = ['temp_mean', 'temp_min', 'temp_max',
                        'hum_mean', 'hum_min', 'hum_max',
                        'day_of_week', 'is_weekend', 'month', 'week_of_year']

        y_pred_recenti = modello.predict(df_recenti[feature_cols])
        df_recenti['consumo_predetto'] = y_pred_recenti

        # Calcola le date future da predire
        oggi = datetime.utcnow().date()
        future_dates = [oggi + timedelta(days=i) for i in range(1, num_giorni_futuri + 1)]

        # Predizioni future
        predizioni_future = []

        for f_date in future_dates:
            # Coordinate per Lecce (modifica se necessario)
            CITY_LAT = 40.35
            CITY_LON = 18.17

            meteo_df = get_weather_data(CITY_LAT, CITY_LON, f_date, f_date, API_KEY)
#            meteo_df = get_weather_data(CITY, f_date, f_date, API_KEY)

            if meteo_df.empty:
                st.warning(f"Nessun dato meteo disponibile per {f_date}. Previsione non effettuata per questo giorno.")
                continue

            meteo_giorno = meteo_df.groupby('Date').agg({
                'Temperature': ['mean', 'min', 'max'],
                'Humidity': ['mean', 'min', 'max']
            }).reset_index()

            meteo_giorno.columns = ['date', 'temp_mean', 'temp_min', 'temp_max', 'hum_mean', 'hum_min', 'hum_max']
            meteo_giorno['day_of_week'] = pd.to_datetime(meteo_giorno['date']).dt.dayofweek
            meteo_giorno['is_weekend'] = meteo_giorno['day_of_week'].isin([5, 6]).astype(int)
            meteo_giorno['month'] = pd.to_datetime(meteo_giorno['date']).dt.month
            meteo_giorno['week_of_year'] = pd.to_datetime(meteo_giorno['date']).dt.isocalendar().week.astype(int)

            y_pred = modello.predict(meteo_giorno[feature_cols])[0]
            predizioni_future.append((f_date, y_pred))

        ##############################

        # === Grafico comparativo ===
        st.subheader(f"Consumo Giornaliero (ultimi {num_giorni} giorni selezionati): Reale vs Predetto")

        energia_per_giorno_ultimi = energia_per_giorno[energia_per_giorno['data'].isin(giorni_recenti)]

        # Aggiungi giorni futuri e predizioni
        future_dates_list = [d for d, _ in predizioni_future]
        future_preds_list = [pred for _, pred in predizioni_future]

        df_bar = pd.DataFrame({
            'Data': list(energia_per_giorno_ultimi['data']) + future_dates_list,
            'Consumo Reale (kWh)': list(energia_per_giorno_ultimi['consumo_kWh']) + [None] * len(future_dates_list),
            'Consumo Predetto (kWh)': list(df_recenti['consumo_predetto']) + future_preds_list
        })
########################
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(df_bar))

        bars_real = ax.bar(x, df_bar['Consumo Reale (kWh)'], width=0.4, label='Reale', align='center')
        bars_pred = ax.bar([i + 0.4 for i in x], df_bar['Consumo Predetto (kWh)'], width=0.4, label='Predetto', align='center')

        for bar in bars_real:
            height = bar.get_height()
            if not pd.isna(height):
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

        for bar in bars_pred:
            height = bar.get_height()
            if not pd.isna(height):
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels([str(d) for d in df_bar['Data']])
        ax.set_ylabel("Consumo (kWh)")
        ax.set_title("Confronto Consumo Reale vs Predetto (Giornaliero)")
        ax.legend()
        st.pyplot(fig)

        # Tabella
        st.subheader(f"Tabella Consumo Giornaliero (ultimi {num_giorni} giorni selezionati)")
        st.dataframe(df_bar)

        # Valori istantanei
        oggi = datetime.utcnow().date()
        giorni_fa = oggi - timedelta(days=num_giorni)
        df_consumo_recenti = df_consumo[df_consumo['timestamp'].dt.date >= giorni_fa]
        df_consumo_recenti = df_consumo_recenti.sort_values(by='timestamp', ascending=False)

        st.subheader(f"Valori istantanei del consumo (ultimi {num_giorni} giorni)")
        st.write("Questi sono tutti i valori di potenza istantanea (Watt) recuperati dall'API:")
        st.dataframe(df_consumo_recenti[['timestamp', 'value']].rename(columns={
            'timestamp': 'Timestamp',
            'value': 'Potenza (W)'
        }))

    except Exception as e:
        st.error(f"Errore: {e}")


#'https://api.tomorrow.io/v4/weather/forecast?location=42.3478,-71.0466&apikey=rzwVU2IQ4Lhvk5TBEpfrMAMuMTgOPDYT'
def get_weather_data(city_lat, city_lon, start_date, end_date, api_key):
    url = "https://api.tomorrow.io/v4/weather/forecast"
    params = {
        "location": f"{city_lat},{city_lon}",
        "timesteps": "1h",
        "apikey": api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "timelines" not in data or "hourly" not in data["timelines"]:
            st.warning(f"️ Risposta inattesa dall'API meteo Tomorrow.io:\n{data}")
            return pd.DataFrame()

        records = []
        for entry in data["timelines"]["hourly"]:
            dt = pd.to_datetime(entry["time"])
            if start_date <= dt.date() <= end_date:
                values = entry["values"]
                records.append({
                    "Datetime": dt,
                    "Date": dt.date(),
                    "Hour": dt.hour,
                    "Temperature": values.get("temperature"),
                    "Humidity": values.get("humidity")
                })

        return pd.DataFrame(records)

    except requests.exceptions.RequestException as e:
        st.error(f"Errore nella richiesta all'API meteo: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Errore imprevisto durante il parsing della risposta meteo: {e}")
        return pd.DataFrame()

