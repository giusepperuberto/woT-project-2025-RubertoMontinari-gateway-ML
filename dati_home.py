import streamlit as st
import requests
import pandas as pd

BASE_URL = "https://gdfhome.duckdns.org/api/states/"
HEADERS = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1MDdmNjliNjA0MmY0M2M1OTk0NjVmZDRiZmZl"
                     "MWZjNyIsImlhdCI6MTcyNDc1MTMxMywiZXhwIjoyMDQwMTExMzEzfQ.S61REKbuTL1l4yP-iIQRDuZvyCmGWDJoL7"
                     "-FQXAl7xg",
    "Content-Type": "application/json"
}

ENTITIES = {
    "Temperatura": "sensor.termostat_temperature",
    "Umidità": "sensor.termostat_humidity",
    "Consumo Energetico": "sensor.quadro_primo_terra_channel_1_power"
}


def fetch_current_data(entity_id):
    try:
        url = BASE_URL + entity_id
        response = requests.get(url, headers=HEADERS, verify=False)
        response.raise_for_status()
        data = response.json()

        raw_value = data["state"]

        # Gestione valori non numerici (unavailable, unknown, ecc.)
        try:
            value = float(raw_value)
        except ValueError:
            value = None

        return {
            "entity_id": data["entity_id"],
            "value": value,
            "unit": data["attributes"].get("unit_of_measurement", "")
        }
    except Exception as e:
        st.error(f"Errore durante il recupero per {entity_id}: {e}")
        return None


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
    # Converte valori numerici, mette NaN se "unavailable"
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def home_data_page():
    st.title("Dati Istantanei dalla Casa (Home Assistant)")

    data_collected = []

    for name, entity_id in ENTITIES.items():
        data = fetch_current_data(entity_id)
        if data:
            data_collected.append(data)

    if data_collected:
        df = pd.DataFrame(data_collected)
        df = df.rename(columns={"entity_id": "Parametro", "value": "Valore", "unit": "Unità"})
        st.dataframe(df, use_container_width=True)

        st.title("Scarica Dati Storici da API")

        ENTITY_OPTIONS = {
            "Consumo abitazione (W)": "sensor.quadro_primo_terra_channel_1_power",
            "Temperatura interna (°C)": "sensor.termostat_temperature",
            "Umidità interna (%)": "sensor.termostat_humidity"
        }

        selected_entity_label = st.selectbox("Scegli il parametro da analizzare:", list(ENTITY_OPTIONS.keys()))
        entity_id = ENTITY_OPTIONS[selected_entity_label]

        start_date = st.date_input("Data inizio")
        end_date = st.date_input("Data fine")

        if st.button("Scarica dati"):
            if not entity_id:
                st.error("Inserisci un ID entità valido.")
            else:
                try:
                    df = get_historical_data(
                        start_date=start_date.strftime("%Y-%m-%dT00:00:00Z"),
                        end_date=end_date.strftime("%Y-%m-%dT23:59:59Z"),
                        entity_id=entity_id
                    )

                    if df.empty:
                        st.warning("Nessun dato trovato per il periodo selezionato.")
                    else:
                        st.success("Dati caricati con successo!")
                        st.dataframe(df)
                        st.write(df[:1])  # Visualizza il primo elemento ricevuto

                        # Salvataggio CSV
                        csv = df.to_csv(index=False)
                        st.download_button("Scarica CSV", data=csv, file_name="storico_dati.csv", mime="text/csv")

                except Exception as e:
                    st.error(f"Errore durante il recupero: {e}")
