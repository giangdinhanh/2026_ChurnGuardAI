# ChurnGuard AI — Streamlit Deployment

This version converts the Flask prototype into a Streamlit application so it can be deployed publicly from a GitHub repository using Streamlit Community Cloud.

## Required repository structure

```text
your-repository/
├── app.py
├── model_service.py
├── customer_churn_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

`database.db` is created automatically when the app runs.

## Important: dataset

The model retrains from the original `customer_churn_data.csv` when the Streamlit app starts. Put the exact CSV used for the project beside `app.py`.

Expected columns:

- CustomerID
- Age
- Gender
- Tenure
- MonthlyCharges
- TotalCharges
- ContractType
- InternetService
- TechSupport
- Churn

The app expects these category labels from the project dataset:

- ContractType: `Month-to-Month`, `One-Year`, `Two-Year`
- InternetService: `Fiber Optic`, `DSL`, or missing (treated as `No Internet Service`)
- Gender: `Male`, `Female`
- TechSupport: `Yes`, `No`
- Churn: `Yes`, `No`

## Model

The deployment uses the selected tuned KNN configuration:

```python
{
    "metric": "manhattan",
    "n_neighbors": 11,
    "p": 1,
    "weights": "distance",
}
```

## Run locally

```bash
python -m venv myvenv
```

Windows:

```bash
myvenv\Scripts\activate
```

macOS/Linux:

```bash
source myvenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Push to GitHub

```bash
git add -A
git commit -m "Convert ChurnGuard to Streamlit"
git push origin main
```

## Deploy publicly

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app.
3. Select your GitHub repository.
4. Select branch `main`.
5. Set the entrypoint file to `app.py`.
6. Deploy.
7. Share the generated `https://...streamlit.app` URL with your instructor.

## SQLite note

The app still uses SQLite so the assignment can demonstrate form → database → model → prediction history.

However, the local filesystem on Streamlit Community Cloud is not permanent application storage. Records written after deployment can disappear when the app restarts or redeploys. The prediction feature itself will still work because the model is retrained from the CSV in the repository.


## Dataset validation

The model loader now automatically:
- strips spaces/BOM from column names
- removes accidental repeated header rows inside the CSV
- keeps only valid `Churn` values (`Yes` / `No`)
- converts numeric fields safely
- checks class counts before stratified splitting

Optional local check:

```bash
python check_dataset.py
```
