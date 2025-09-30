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
        return {
            "entity_id": data["entity_id"],
            "value": float(data["state"]),
            "unit": data["attributes"].get("unit_of_measurement", "")
        }
    except Exception as e:
        st.error(f"Errore durante il recupero per {entity_id}: {e}")
        return None

def realtime_data_page():
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

