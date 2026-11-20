# Progetto — Previsione ADR alberghiero

Il progetto stima l’ADR (*Average Daily Rate*), la tariffa media giornaliera di una camera, a partire dalle caratteristiche delle prenotazioni del dataset *Hotel Booking Demand*. La dashboard Streamlit permette di esplorare i dati e stimare l’ADR per una prenotazione.

## Requisiti

- Python 3.14 o superiore (come specificato nella configurazione originale del progetto).
- Mantieni la struttura delle cartelle del repository.

## Installazione

1. Clonare il repository da GitHub oppure scaricarlo come ZIP ed estrarlo.
2. Aprire un terminale nella cartella principale del progetto, quella che contiene `requirements.txt` e la cartella `03_DashBoard`.
3. Installare le dipendenze:

   ```bash
   python -m pip install -r requirements.txt
   ```

   Su Windows, se `python` non è riconosciuto, provare `py -m pip install -r requirements.txt`.

## Avvio della dashboard

Dopo l’installazione puoi avviare direttamente la dashboard se sono presenti questi quattro file, con i percorsi indicati a partire da `03_DashBoard/`.

- `DashBoard_data/dashboard_data.csv`
- `modello_r_f/random_forest_model.joblib`
- `modello_r_f/model_features.json`
- `modello_r_f/model_metrics.json`

Al primo avvio dopo il clone, segui la sezione [Notebook](#notebook) per generare `random_forest_model.joblib` e gli altri artefatti eventualmente mancanti.

Dalla cartella principale del progetto, eseguire:

```bash
python -m streamlit run 03_DashBoard/Hotel_booking_dashboard/App.py
```

Streamlit mostrerà nel terminale l'indirizzo locale della dashboard (generalmente `http://localhost:8501`). Per interromperla, premere `Ctrl+C` nel terminale.

## Notebook

Per riprodurre l’analisi, scarica il CSV del dataset [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data) da Kaggle e salvalo come `Dati/DataFrame.csv`. Crea le cartelle `Dati/`, `03_DashBoard/DashBoard_data/` e `03_DashBoard/modello_r_f/` se non sono già presenti.

I notebook vanno eseguiti nell'ordine:

1. `00_DataExploration/data_cleaning.ipynb` pulisce il dataset e genera `Dati/DataFrame_00.csv`.
2. `01_FeatureSelection/feature_selection.ipynb` seleziona le variabili e genera `Dati/DataFrame_01.csv` e `dashboard_data.csv`.
3. `02_modelli/modelli.ipynb` confronta i modelli di regressione ed esporta la Random Forest e i due file JSON.

Per aprirli, avviare Jupyter dalla cartella principale:

```bash
python -m jupyter notebook
```

## Risultati

Sul test set, ottenuto con uno split casuale del 30%, la Random Forest ha ottenuto un MAE di 8,36 €, un RMSE di 14,64 € e un R² di 0,898. Il Dummy Regressor ha ottenuto un MAE di 34,10 €. La selezione delle variabili è stata condotta sull’intero dataset prima dello split.
