# Project — Hotel ADR Prediction

This project was developed during GiGroup's Junior Data Scientist Academy in collaboration with Mariapia Strazzella, Lorenzo Quintè, and Costanza Pizzi.

The project estimates ADR (*Average Daily Rate*), the average daily room rate, using booking features from the *Hotel Booking Demand* dataset. The Streamlit dashboard lets you explore the data and estimate the ADR for a booking.

## Requirements

- Python 3.14 or later (as specified in the original project configuration).
- Keep the repository's folder structure intact.

## Installation

1. Clone the repository from GitHub or download it as a ZIP file and extract it.
2. Open a terminal in the project root, which contains `requirements.txt` and the `03_DashBoard` folder.
3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

   On Windows, if `python` is not recognized, try `py -m pip install -r requirements.txt`.

## Running the Dashboard

After installation, you can launch the dashboard directly if the following four files are present. The paths are relative to `03_DashBoard/`.

- `DashBoard_data/dashboard_data.csv`
- `modello_r_f/random_forest_model.joblib`
- `modello_r_f/model_features.json`
- `modello_r_f/model_metrics.json`

Before launching the dashboard for the first time after cloning the repository, follow the [Notebook](#notebook) section to generate `random_forest_model.joblib` and any other missing artifacts.

From the project root, run:

```bash
python -m streamlit run 03_DashBoard/Hotel_booking_dashboard/App.py
```

Streamlit will display the dashboard's local address in the terminal (usually `http://localhost:8501`). To stop the dashboard, press `Ctrl+C` in the terminal.

## Notebook

To reproduce the analysis, download the [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data) CSV from Kaggle and save it as `Dati/DataFrame.csv`. Create the `Dati/`, `03_DashBoard/DashBoard_data/`, and `03_DashBoard/modello_r_f/` folders if they do not already exist.

Run the notebooks in this order:

1. `00_DataExploration/data_cleaning.ipynb` cleans the dataset and generates `Dati/DataFrame_00.csv`.
2. `01_FeatureSelection/feature_selection.ipynb` selects the features and generates `Dati/DataFrame_01.csv` and `dashboard_data.csv`.
3. `02_modelli/modelli.ipynb` compares the regression models and exports the Random Forest model and the two JSON files.

To open the notebooks, launch Jupyter from the project root:

```bash
python -m jupyter notebook
```

## Results

On the test set, which consisted of a random 30% of the data, the Random Forest achieved an MAE of €8.36, an RMSE of €14.64, and an R² of 0.898. The Dummy Regressor achieved an MAE of €34.10. Feature selection was performed on the full dataset before the split.
