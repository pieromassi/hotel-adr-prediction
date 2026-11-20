import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. CONFIGURAZIONE DELLA PAGINA
# --------------------------------------------------
st.set_page_config(
    page_title="Analisi delle prenotazioni alberghiere",
    page_icon="H",
    layout="wide"
)

# --------------------------------------------------
# 2. CARICAMENTO DEI FILE
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "DashBoard_data" / "dashboard_data.csv"
MODEL_PATH = BASE_DIR / "modello_r_f" / "random_forest_model.joblib"
FEATURES_PATH = BASE_DIR / "modello_r_f" / "model_features.json"
METRICS_PATH = BASE_DIR / "modello_r_f" / "model_metrics.json"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

# Controlla che tutti i file necessari esistano
required_files = [
    DATA_PATH,
    MODEL_PATH,
    FEATURES_PATH,
    METRICS_PATH
]
missing_files = [
    str(path) for path in required_files
    if not path.exists()
]
if missing_files:
    st.error("Mancano alcuni file necessari:")
    for path in missing_files:
        st.write(path)
    st.stop()

df = load_data()
model = load_model()
features = load_json(FEATURES_PATH)
visible_features = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_day_of_month", "adults", "children", "meal",
    "reserved_room_type"
]
hidden_features = [f for f in features if f not in visible_features]
metrics = load_json(METRICS_PATH)

# --------------------------------------------------
# 3. BARRA LATERALE
# --------------------------------------------------
st.sidebar.title("Hotel Analytics")
page = st.sidebar.radio(
    "Navigazione",
    [
        "Dashboard",
        "Previsione ADR",
        "Esplora i dati",
        "Modello"
    ]
)
st.sidebar.markdown("---")
st.sidebar.caption("Hotel Booking Demand")

# --------------------------------------------------
# 4. DASHBOARD
# --------------------------------------------------
if page == "Dashboard":
    st.title("Analisi delle prenotazioni alberghiere")
    st.write(
        "Esplora i dati delle prenotazioni alberghiere e stima la "
        "tariffa media giornaliera (ADR)."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Record totali",
            f"{len(df):,}"
        )
    with col2:
        st.metric(
            "ADR medio",
            f"€{df['adr'].mean():.2f}"
        )
    with col3:
        st.metric(
            "ADR mediano",
            f"€{df['adr'].median():.2f}"
        )
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.subheader("Distribuzione dell'ADR")
        st.plotly_chart(
            __import__("plotly.express", fromlist=[""]).histogram(
                df,
                x="adr",
                nbins=40,
                title="Distribuzione delle tariffe giornaliere delle camere"
            ),
            use_container_width=True
        )
    with right:
        st.subheader("ADR medio per hotel")
        if "hotel" in df.columns:
            hotel_adr = (
                df.groupby("hotel", as_index=False)["adr"]
                .mean()
                .sort_values("adr", ascending=False)
            )
            st.bar_chart(
                hotel_adr.set_index("hotel")["adr"]
            )
        else:
            st.info("La colonna hotel non è disponibile.")
    st.subheader("Anteprima del dataset")
    st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# 5. PREVISIONE ADR
# --------------------------------------------------
elif page == "Previsione ADR":
    st.title("Previsione ADR")
    st.write(
        "Inserisci le caratteristiche della prenotazione per stimare "
        "la tariffa media giornaliera della camera."
    )
    input_data = {}
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        for i, feature in enumerate(visible_features):
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                if feature not in df.columns:
                    st.warning(
                        f"Variabile non disponibile nel dataset: {feature}"
                    )
                    continue
                series = df[feature]
                if pd.api.types.is_numeric_dtype(series):
                    median_value = series.median()
                    if pd.isna(median_value):
                        median_value = 0
                    input_data[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        value=int(median_value)
                    )
                else:
                    options = (
                        series.dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    options = sorted(options)
                    if options:
                        input_data[feature] = st.selectbox(
                            feature.replace("_", " ").title(),
                            options
                        )
                    else:
                        input_data[feature] = st.text_input(
                            feature.replace("_", " ").title()
                        )
        submitted = st.form_submit_button(
            "Prevedi ADR",
            use_container_width=True
        )
    if submitted:
        # Valori predefiniti per le variabili nascoste nel form
        for feature in hidden_features:
            if feature not in df.columns:
                continue
            series = df[feature]
            if pd.api.types.is_numeric_dtype(series):
                input_data[feature] = series.median()
            else:
                input_data[feature] = series.mode().iloc[0]

        missing_features = [
            feature for feature in features
            if feature not in input_data
        ]
        if missing_features:
            st.error(
                "Non è stato possibile raccogliere alcune variabili del modello: "
                + ", ".join(missing_features)
            )
        else:
            input_df = pd.DataFrame(
                [input_data],
                columns=features
            )
            try:
                prediction = float(model.predict(input_df)[0])
                st.markdown("---")
                st.subheader("Risultato della previsione")
                st.metric(
                    "ADR previsto",
                    f"€{prediction:.2f}"
                )
                if prediction < 0:
                    st.warning(
                        "Il modello ha restituito un valore negativo. "
                        "Controlla il modello e i dati di addestramento."
                    )
            except Exception as error:
                st.error(
                    f"Previsione non riuscita: {error}"
                )

# --------------------------------------------------
# 6. ESPLORA I DATI
# --------------------------------------------------
elif page == "Esplora i dati":
    st.title("Esplora il dataset")
    filtered_df = df.copy()
    if "hotel" in df.columns:
        hotel_options = sorted(
            df["hotel"].dropna().astype(str).unique()
        )
        selected_hotels = st.multiselect(
            "Seleziona il tipo di hotel",
            hotel_options,
            default=hotel_options
        )
        filtered_df = filtered_df[
            filtered_df["hotel"].astype(str).isin(
                selected_hotels
            )
        ]
    if "adr" in filtered_df.columns:
        min_adr = float(filtered_df["adr"].min())
        max_adr = float(filtered_df["adr"].max())
        if min_adr < max_adr:
            adr_range = st.slider(
                "Filtra l'intervallo di ADR",
                min_value=min_adr,
                max_value=max_adr,
                value=(min_adr, max_adr)
            )
            filtered_df = filtered_df[
                filtered_df["adr"].between(
                    adr_range[0],
                    adr_range[1]
                )
            ]
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Record filtrati",
            f"{len(filtered_df):,}"
        )
    with col2:
        if len(filtered_df) > 0:
            st.metric(
                "ADR medio filtrato",
                f"€{filtered_df['adr'].mean():.2f}"
            )
        else:
            st.metric("ADR medio filtrato", "N/D")
    st.subheader("Dati filtrati")
    st.dataframe(
        filtered_df.head(500),
        use_container_width=True
    )
    st.download_button(
        "Scarica i dati filtrati",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_hotel_data.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# 7. PRESTAZIONI DEL MODELLO
# --------------------------------------------------
elif page == "Modello":
    st.title("Prestazioni del modello")
    st.write(
        "Le seguenti metriche valutano il modello "
        "sul test set, escluso dall'addestramento."
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
                st.metric(
                    "MAPE",
                    f"{metrics['MAPE %']:.3f}"
                )
    with col2:
        st.metric(
            "MAE",
            f"{metrics['MAE']:.3f}"
        )
    with col3:
        st.metric(
            "RMSE",
            f"{metrics['RMSE']:.3f}"
        )
    with col4:
        st.metric(
            "R²",
            f"{metrics['R2']:.3f}"
        )
    st.markdown("---")
    st.subheader("Come interpretare le metriche")
    st.write(
            "**MAPE:** Differenza in percentuale fra la media predetta e quella effettiva del adr "
        )
    st.write(
        "**MAE:** differenza assoluta media tra "
        "l'ADR previsto e quello effettivo."
    )
    st.write(
        "**RMSE:** errore di previsione che penalizza "
        "maggiormente gli errori grandi."
    )
    st.write(
        "**R²:** proporzione della variabilità del target "
        "spiegata dal modello."
    )
    st.info(
        "Valori più bassi di MAE e RMSE sono generalmente migliori. "
        "Un R² più vicino a 1 indica un adattamento migliore."
    )