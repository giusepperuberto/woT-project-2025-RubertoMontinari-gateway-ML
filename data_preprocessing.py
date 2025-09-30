
import os
import pandas as pd

def clean_dataset(use_isolation_forest=False):
    if os.path.exists('file_ripulito.csv'):
        print("✅ file_ripulito.csv già esistente. Nessuna azione eseguita.")
        return


        # === 3. Carica e pulisci il terzo dataset ===
    third_dataset_path = os.path.join("data", "energy_weather_raw_data.csv")
    df3 = pd.read_csv(third_dataset_path, parse_dates=['date'])

    df3 = df3.rename(columns={'date': 'datetime'})
    df3 = df3[['datetime', 'active_power', 'temp', 'humidity']]  # Seleziona solo le colonne utili
    df3 = df3.rename(columns={
        'active_power': 'EnergyConsumption',
        'temp': 'Temperature',
        'humidity': 'Humidity'
    })

    df3 = df3.dropna()
    df3 = df3[(df3['Temperature'].between(-30, 50)) &
              (df3['Humidity'].between(0, 100)) &
              (df3['EnergyConsumption'] >= 0)]

    df3.set_index('datetime', inplace=True)
    df3 = df3.resample('1h').mean()  # Granularità a 1 minuto
    df3.interpolate(method='time', inplace=True)
    df3.ffill(inplace=True)
    df3.bfill(inplace=True)
    df3.dropna(inplace=True)

    import numpy as np

    # === 4. Carica e pulisci il quarto dataset (da timestamp UNIX) ===
    fourth_dataset_path = os.path.join("data", "HomeC.csv")
    # Leggo con gestione tipo e delimitatori inconsistenti
    df4 = pd.read_csv(fourth_dataset_path, low_memory=False)

    # Rimuovo eventuali righe con valori non validi nella colonna 'time'
    df4 = df4[pd.to_numeric(df4['time'], errors='coerce').notnull()]
    df4['time'] = df4['time'].astype(float).astype(int)

    # Converto timestamp in datetime
    df4['datetime'] = pd.to_datetime(df4['time'], unit='s')

    # Rinomino colonne utili
    df4 = df4.rename(columns={
        'House overall [kW]': 'EnergyConsumption',
        'temperature': 'Temperature',
        'humidity': 'Humidity'
    })

    print("Dimensione df4:", df4.shape)

    # Seleziono solo le colonne rilevanti
    df4 = df4[['datetime', 'EnergyConsumption', 'Temperature', 'Humidity']]
    df4['EnergyConsumption'] *= 1000

    # Pulizia valori
    df4 = df4.dropna()
    df4 = df4[(df4['Temperature'].between(-30, 50)) &
              (df4['Humidity'].between(0, 100)) &
              (df4['EnergyConsumption'] >= 0)]

    print("Dimensione df4:", df4.shape)

    df4.set_index('datetime', inplace=True)
    df4 = df4.resample('1h').mean()
    df4.interpolate(method='time', inplace=True)
    df4.ffill(inplace=True)
    df4.bfill(inplace=True)
    df4.dropna(inplace=True)


    print("Dimensione df4:", df4.shape)

    # === 5. Carica e pulisci il quinto dataset (simile a HomeC.csv) ===
    fifth_dataset_path = os.path.join("data", "Smart Home Dataset.csv")
    df5 = pd.read_csv(fifth_dataset_path, low_memory=False)

    # Rimuove righe con timestamp non validi
    df5 = df5[pd.to_numeric(df5['time'], errors='coerce').notnull()]
    df5['time'] = df5['time'].astype(float).astype(int)
    df5['datetime'] = pd.to_datetime(df5['time'], unit='s')

    # Seleziona e rinomina solo le colonne utili
    df5 = df5.rename(columns={
        'House overall [kW]': 'EnergyConsumption',
        'temperature': 'Temperature',
        'humidity': 'Humidity'
    })
    df5 = df5[['datetime', 'EnergyConsumption', 'Temperature', 'Humidity']]
    df5['Humidity'] = df5['Humidity'] * 100
    df5['EnergyConsumption'] = df5['EnergyConsumption'] * 1000
    print("Dimensione df5:", df5.shape)

    # Pulisce i dati
    df5 = df5.dropna()
    df5 = df5[(df5['Temperature'].between(-30, 50)) &
              (df5['Humidity'].between(0, 100)) &
              (df5['EnergyConsumption'] >= 0)]

    df5.set_index('datetime', inplace=True)

    # Resample a livello orario
    df5 = df5.resample('1h').mean()
    df5.interpolate(method='time', inplace=True)
    df5.ffill(inplace=True)
    df5.bfill(inplace=True)
    df5.dropna(inplace=True)
    # Controlla come sono i dati a video:
    print(df5.head())

    # Percorso del file Excel contenente i dati simili a quelli che hai mostrato
    #excel_path = r"C:\Users\giuse\Desktop\dataset\dataset_prof\df_api_historical.xlsx"
    excel_path = os.path.join("data", "df_api_historical.xlsx")
    # Carica il file Excel (assumendo che i dati siano nel primo foglio o specifica il nome del foglio)
    df_api = pd.read_excel(excel_path)


    # Se le colonne hanno nomi simili a quelli mostrati, rinominale
    df_api = df_api.rename(columns={
        'last_changed': 'datetime',
        'power': 'EnergyConsumption',
        'temperature_indoor': 'Temperature',
        'humidity_indoor': 'Humidity'
    })

    # Se i numeri sono con virgola come separatore decimale (es. 80,65882353),
    # pandas in genere li legge come stringhe: converti con replace e astype(float):
    for col in ['EnergyConsumption', 'Temperature', 'Humidity']:
        # Se la colonna non è ancora numerica, converti:
        if df_api[col].dtype == 'object':
            df_api[col] = df_api[col].astype(str).str.replace(',', '.').astype(float)

    # Converti la colonna datetime
    df_api['datetime'] = pd.to_datetime(df_api['datetime'])

    # Imposta datetime come indice
    df_api.set_index('datetime', inplace=True)

    # Pulisci i dati come fatto con df4 e df5
    df_api = df_api.dropna()
    df_api = df_api[(df_api['Temperature'].between(-30, 50)) &
                    (df_api['Humidity'].between(0, 100)) &
                    (df_api['EnergyConsumption'] >= 0)]

    # Resample a livello orario
    df_api = df_api.resample('1h').mean()
    df_api.interpolate(method='time', inplace=True)
    df_api.ffill(inplace=True)
    df_api.bfill(inplace=True)
    df_api.dropna(inplace=True)


    ###################
    #prova integrazione
    # --- Caricamento e pulizia dati multi-casa (da csv o excel) ---
    #file_path = r'C:\Users\giuse\Downloads\Smart Home Energy Consumption Optimization.csv'  # cambia percorso
    file_path = os.path.join("data", "Smart Home Energy Consumption Optimization.csv")
    
    df = pd.read_csv(file_path)  # o pd.read_excel se excel

    # Converte timestamp in datetime
    df['datetime'] = pd.to_datetime(df['timestamp'])


    # Conversione numeri con virgola come separatore decimale, se serve (esempio generico)
    for col in ['power_watt', 'indoor_temp', 'humidity']:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    # Rimuove righe con dati anomali o NaN
    df = df.dropna(subset=['power_watt', 'indoor_temp', 'humidity'])

    df = df[(df['indoor_temp'].between(-30, 50)) &
            (df['humidity'].between(0, 100)) &
            (df['power_watt'] >= 0)]

    # Raggruppa per casa e ora: somma consumo, media temp e umidità
    df_grouped_home = df.groupby(['home_id', 'datetime']).agg({
        'power_watt': 'sum',
        'indoor_temp': 'mean',
        'humidity': 'mean'
    }).reset_index()

    # Ora aggrega su tutte le case per ogni ora: media consumo, temp, umidità
    df_agg = df_grouped_home.groupby('datetime').agg({
        'power_watt': 'mean',  # oppure 'sum' se vuoi totale consumo su tutte le case
        'indoor_temp': 'mean',
        'humidity': 'mean'
    }).reset_index()

    # Rinomina colonne per coerenza
    df_agg = df_agg.rename(columns={
        'power_watt': 'EnergyConsumption',
        'indoor_temp': 'Temperature',
        'humidity': 'Humidity'
    })

    # Imposta indice datetime
    df_agg.set_index('datetime', inplace=True)

    # Resample e interpolazione (se vuoi uniformare a orari precisi e riempire eventuali buchi)
    df_agg = df_agg.resample('1h').mean()
    df_agg.interpolate(method='time', inplace=True)
    df_agg.ffill(inplace=True)
    df_agg.bfill(inplace=True)
    df_agg.dropna(inplace=True)

    print("Prima della concatenazione:", df3.shape, df4.shape, df5.shape, df_api.shape, df_agg.shape)



    # === 1. Caricamento dati consumo ===
    #df_energy = pd.read_csv(r"C:\Users\giuse\Desktop\dataset\dataset_esterni\utilizzabili\HomeB-meter1_2015.csv")  # metti il tuo path
    df_energy = os.path.join("data", "HomeB-meter1_2015.csv")
    df_energy['datetime'] = pd.to_datetime(df_energy['Date & Time'])
    df_energy['datetime'] = df_energy['datetime'].dt.floor('h')  # arrotonda all'ora
    df_energy_hourly = df_energy.groupby('datetime')['use [kW]'].mean().reset_index()

    # === 2. Caricamento dati meteo ===
    #df_weather = pd.read_csv(r"C:\Users\giuse\Desktop\dataset\dataset_esterni\utilizzabili\homeB2015.csv")  # metti il tuo path
    df_weather = os.path.join("data", "homeB2015.csv")
    df_weather['datetime'] = pd.to_datetime(df_weather['time'], unit='s')
    df_weather['datetime'] = df_weather['datetime'].dt.floor('h')  # arrotonda all'ora
    df_weather_hourly = df_weather.groupby('datetime')[['temperature', 'humidity']].mean().reset_index()

    # === 3. Merge su datetime orario ===
    df_merged = pd.merge(df_energy_hourly, df_weather_hourly, on='datetime', how='inner')

    # === 4. Rinominare per chiarezza ===
    df_merged = df_merged.rename(columns={
        'use [kW]': 'EnergyConsumption',
        'temperature': 'Temperature',
        'humidity': 'Humidity'
    })

    df_merged['EnergyConsumption'] *= 1000

    # === 5. (Facoltativo) Imposta datetime come indice ===
    df_merged.set_index('datetime', inplace=True)



    # === 1. Caricamento dati consumo ===
    #df_energy_14 = pd.read_csv(r"C:\Users\giuse\Desktop\dataset\dataset_esterni\utilizzabili\HomeB-meter1_2014.csv")  # metti il tuo path
    df_energy_14 = os.path.join("data", "HomeB-meter1_2014.csv")
    df_energy_14['datetime'] = pd.to_datetime(df_energy_14['Date & Time'])
    df_energy_14['datetime'] = df_energy_14['datetime'].dt.floor('h')  # arrotonda all'ora
    df_energy_hourly_14 = df_energy.groupby('datetime')['use [kW]'].mean().reset_index()

    # === 2. Caricamento dati meteo ===
    #df_weather_14 = pd.read_csv(r"C:\Users\giuse\Desktop\dataset\dataset_esterni\utilizzabili\homeB2014.csv")  # metti il tuo path
    df_weather_14 = os.path.join("data", "homeB2014.csv")
    df_weather_14['datetime'] = pd.to_datetime(df_weather_14['time'], unit='s')
    df_weather_14['datetime'] = df_weather_14['datetime'].dt.floor('h')  # arrotonda all'ora
    df_weather_hourly_14 = df_weather_14.groupby('datetime')[['temperature', 'humidity']].mean().reset_index()

    # === 3. Merge su datetime orario ===
    df_merged_14 = pd.merge(df_energy_hourly_14, df_weather_hourly_14, on='datetime', how='inner')

    # === 4. Rinominare per chiarezza ===
    df_merged_14 = df_merged_14.rename(columns={
        'use [kW]': 'EnergyConsumption',
        'temperature': 'Temperature',
        'humidity': 'Humidity'
    })

    df_merged_14['EnergyConsumption'] *= 1000

    # === 5. (Facoltativo) Imposta datetime come indice ===
    df_merged_14.set_index('datetime', inplace=True)

    # === 6. Concateno i due dataset ===
    df = pd.concat([df_agg, df3, df4, df5, df_api, df_merged, df_merged_14]).sort_index()#[df3, df4, df5, df_api, df_agg, df_merged, df_merged_14]).sort_index() #df1, df2,df_api,,
    print("Prima della pulizia indice:", df.shape)
    df = df[~df.index.duplicated(keep='first')]
    print("Dopo pulizia indice:", df.shape)

    # Aggiunta feature temporali
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['is_weekend'] = ((df.index.dayofweek == 5) | (df.index.dayofweek == 6)).astype(int)


    print("Max consumo prima del MAD:", df['EnergyConsumption'].max())

    print("Dimensione df:", df.shape)

    # === 1. Pulizia valori estremi "hard" ===
    #df = df[df['EnergyConsumption'] > 30]
    df = df[df['EnergyConsumption'] < 3000]

    print("Dimensione df dopo pulizia:", df.shape)

    # === 2. Rimozione outlier con MAD ===
    def remove_outliers_mad(df, col, thresh=3.5):
        median = df[col].median()
        diff = (df[col] - median).abs()
        mad = diff.median()
        if mad == 0:
            return df
        modified_z_score = 0.6745 * diff / mad
        return df[modified_z_score <= thresh]

    for col in ['Temperature', 'Humidity']:
        before = df.shape[0]
        df = remove_outliers_mad(df, col, thresh=3.5)
        after = df.shape[0]
        print(f"Rimossi {before - after} outlier da {col} con MAD.")

    # === 3. Smussa picchi/rumore a breve termine ===
    df['EnergyConsumption'] = df['EnergyConsumption'].rolling(window=3, center=True).median().bfill().ffill()
    df['EnergyConsumption'] = df['EnergyConsumption'].ewm(span=10, adjust=False).mean()

    print("Dimensione df dopo rolling:", df.shape)

    df.reset_index(inplace=True)
    df.to_csv('file_ripulito.csv', index=False)
    print("✅ Dataset pulito salvato come 'file_ripulito.csv'")

if __name__ == "__main__":
    clean_dataset(use_isolation_forest=True)  # True o False come preferisci




