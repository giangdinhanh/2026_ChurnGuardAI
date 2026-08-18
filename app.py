from pathlib import Path
import sqlite3
import time

import pandas as pd
import streamlit as st

from model_service import ChurnModelService, MODEL_PARAMS

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn_data.csv"
DB_PATH = BASE_DIR / "database.db"

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
        :root {
            --brand: #ab3500;
            --brand2: #ff6b35;
            --ink: #292827;
            --muted: #716f6d;
            --surface: #fcf9f8;
        }
        .stApp { background: var(--surface); color: var(--ink); }
        .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 4rem; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        .hero {
            padding: 3rem 3.2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #fff 0%, #fff4ef 60%, #ffe2d5 100%);
            border: 1px solid rgba(171,53,0,.12);
            box-shadow: 0 18px 45px rgba(62,42,32,.08);
            margin-bottom: 1.5rem;
        }
        .hero-kicker {
            color: #ab3500;
            font-size: .82rem;
            font-weight: 800;
            letter-spacing: .12em;
        }
        .hero h1 { font-size: 3.2rem; margin: .35rem 0 1rem 0; }
        .hero p { font-size: 1.15rem; color: #625d59; max-width: 760px; }
        .card {
            background: rgba(255,255,255,.82);
            padding: 1.25rem 1.35rem;
            border-radius: 18px;
            border: 1px solid rgba(0,0,0,.06);
            box-shadow: 0 8px 24px rgba(0,0,0,.045);
            min-height: 130px;
        }
        .risk-high { padding: 1rem 1.2rem; border-radius: 16px; background:#ffdad6; }
        .risk-medium { padding: 1rem 1.2rem; border-radius: 16px; background:#fff0c2; }
        .risk-low { padding: 1rem 1.2rem; border-radius: 16px; background:#ddf5e5; }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid rgba(0,0,0,.06);
            padding: 16px;
            border-radius: 16px;
        }
        .model-chip {
            display:inline-block; padding:.35rem .7rem; border-radius:999px;
            background:#fff0e9; color:#ab3500; font-weight:700; font-size:.85rem;
        }
        .small-note { color:#777; font-size:.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- SQLite ----------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                tenure INTEGER NOT NULL,
                monthly_charges REAL NOT NULL,
                total_charges REAL NOT NULL,
                contract_type TEXT NOT NULL,
                internet_service TEXT NOT NULL,
                tech_support TEXT NOT NULL,
                prediction TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_prediction(values, result):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                age, gender, tenure, monthly_charges, total_charges,
                contract_type, internet_service, tech_support,
                prediction, churn_probability, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(values["age"]),
                values["gender"],
                int(values["tenure"]),
                float(values["monthly_charges"]),
                float(values["total_charges"]),
                values["contract_type"],
                values["internet_service"],
                values["tech_support"],
                result["prediction"],
                result["probability"],
                result["risk_level"],
            ),
        )
        conn.commit()


def load_history():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC"
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def clear_history():
    with get_db() as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()


init_db()


# ---------- Model ----------
@st.cache_resource(show_spinner=False)
def load_model(data_path: str, modified_time_ns: int, file_size: int):
    """
    Cache the trained model, but automatically rebuild it whenever the CSV changes.
    modified_time_ns and file_size are included in the cache key on purpose.
    """
    return ChurnModelService(Path(data_path))


if DATA_PATH.exists():
    _csv_stat = DATA_PATH.stat()
    model_service = load_model(
        str(DATA_PATH),
        _csv_stat.st_mtime_ns,
        _csv_stat.st_size,
    )
else:
    model_service = ChurnModelService(DATA_PATH)


# ---------- Shared helpers ----------
def page_header(title, subtitle):
    st.markdown(f"# {title}")
    st.caption(subtitle)


def show_model_status():
    if model_service.ready:
        st.success(
            f"Model ready · Tuned KNN · k={MODEL_PARAMS['n_neighbors']} · "
            f"{MODEL_PARAMS['metric'].title()} distance · {MODEL_PARAMS['weights']} weighting"
        )
    else:
        st.error("Model is not ready.")
        st.code(model_service.error or "Unknown model-loading error")
        st.info(
            "Add `customer_churn_data.csv` to the same GitHub folder as `app.py`, "
            "then commit and redeploy."
        )


# ---------- Pages ----------
def home():
    st.markdown(
        """
        <section class="hero">
          <div class="hero-kicker">CHURNGUARD AI · CUSTOMER RETENTION</div>
          <h1>Reduce Churn. Increase Loyalty.</h1>
          <p>
            Explore customer patterns and run a tuned K-Nearest Neighbours model
            to estimate churn probability from eight customer attributes.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", "1,000")
    c2.metric("Churned", "883", "88.3%")
    c3.metric("Retained", "117", "11.7%")
    c4.metric("Tuned KNN Accuracy", "96.5%", "Test set")

    st.write("")
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            '<div class="card"><h3>🎯 Live Prediction</h3>'
            '<p>Enter a customer profile and receive churn probability and risk level.</p></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="card"><h3>📊 Analysis</h3>'
            '<p>Review dataset-level churn patterns and prediction history.</p></div>',
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            '<div class="card"><h3>🧠 Tuned KNN</h3>'
            '<p>Uses k=11, Manhattan distance and distance-based weighting.</p></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.info("Use **Analysis → Predictions** in the sidebar to run the model.")


def overview():
    page_header(
        "Analysis Overview",
        "Dataset summary and prediction activity.",
    )

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        work = df.copy()
        work["InternetService"] = work["InternetService"].fillna("No Internet Service")

        total = len(work)
        churned = int((work["Churn"] == "Yes").sum())
        retained = int((work["Churn"] == "No").sum())

        a, b, c, d = st.columns(4)
        a.metric("Total Customers", f"{total:,}")
        b.metric("Churn Rate", f"{churned / total:.1%}", f"{churned:,} customers")
        c.metric("Retention Rate", f"{retained / total:.1%}", f"{retained:,} customers")
        d.metric("Model", "Tuned KNN", "96.5% test accuracy")

        st.subheader("Churn rate by contract type")
        contract = (
            work.assign(churn_flag=(work["Churn"] == "Yes").astype(int))
            .groupby("ContractType", as_index=False)
            .agg(Customers=("Churn", "size"), Churn_Rate=("churn_flag", "mean"))
        )
        st.bar_chart(contract.set_index("ContractType")["Churn_Rate"])

        left, right = st.columns(2)
        with left:
            st.subheader("Internet service mix")
            service = work["InternetService"].value_counts()
            st.bar_chart(service)
        with right:
            st.subheader("Tech support vs churn")
            support = pd.crosstab(
                work["TechSupport"],
                work["Churn"],
                normalize="index",
            )
            st.bar_chart(support)
    else:
        st.warning("Dataset not found. Add `customer_churn_data.csv` to enable overview charts.")

    history = load_history()
    st.subheader("Prediction activity")
    if history.empty:
        st.info("No predictions have been stored in this runtime yet.")
    else:
        a, b = st.columns(2)
        a.metric("Stored Predictions", len(history))
        b.metric("High-risk Predictions", int((history["risk_level"] == "HIGH").sum()))
        st.dataframe(history, use_container_width=True, hide_index=True)


def predictions():
    page_header(
        "Customer Risk Predictor",
        "Enter one customer profile and run the tuned KNN model.",
    )
    show_model_status()

    defaults = st.session_state.get(
        "prediction_defaults",
        {
            "age": 35,
            "gender": "Male",
            "tenure": 12,
            "monthly_charges": 70.0,
            "total_charges": 840.0,
            "contract_type": "Month-to-Month",
            "internet_service": "Fiber Optic",
            "tech_support": "No",
        },
    )

    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("Customer profile")

        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=18, max_value=120, value=int(defaults["age"]))
        gender = c2.selectbox(
            "Gender",
            ["Male", "Female"],
            index=["Male", "Female"].index(defaults["gender"]),
        )

        c3, c4 = st.columns(2)
        tenure = c3.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=120,
            value=int(defaults["tenure"]),
        )
        contract_type = c4.selectbox(
            "Contract Type",
            ["Month-to-Month", "One-Year", "Two-Year"],
            index=["Month-to-Month", "One-Year", "Two-Year"].index(defaults["contract_type"]),
        )

        c5, c6 = st.columns(2)
        monthly_charges = c5.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=float(defaults["monthly_charges"]),
            step=1.0,
        )
        total_charges = c6.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=float(defaults["total_charges"]),
            step=10.0,
        )

        c7, c8 = st.columns(2)
        internet_service = c7.selectbox(
            "Internet Service",
            ["Fiber Optic", "DSL", "No Internet Service"],
            index=["Fiber Optic", "DSL", "No Internet Service"].index(defaults["internet_service"]),
        )
        tech_support = c8.selectbox(
            "Tech Support",
            ["Yes", "No"],
            index=["Yes", "No"].index(defaults["tech_support"]),
        )

        submitted = st.form_submit_button(
            "Run Prediction",
            type="primary",
            use_container_width=True,
            disabled=not model_service.ready,
        )

    reset_col, clear_col = st.columns(2)
    if reset_col.button("Reset Form", use_container_width=True):
        st.session_state["prediction_defaults"] = {
            "age": 35,
            "gender": "Male",
            "tenure": 12,
            "monthly_charges": 70.0,
            "total_charges": 840.0,
            "contract_type": "Month-to-Month",
            "internet_service": "Fiber Optic",
            "tech_support": "No",
        }
        st.session_state.pop("latest_result", None)
        st.rerun()

    if clear_col.button("Clear Database History", use_container_width=True):
        clear_history()
        st.session_state.pop("latest_result", None)
        st.success("Prediction history cleared.")
        st.rerun()

    if submitted:
        values = {
            "age": age,
            "gender": gender,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
        }

        st.session_state["prediction_defaults"] = values

        status = st.status("Running churn prediction...", expanded=True)
        status.write("1/4 Validating customer input")
        time.sleep(0.25)
        status.write("2/4 Encoding categorical features")
        time.sleep(0.25)
        status.write("3/4 Standardizing features and running tuned KNN")
        result = model_service.predict(values)
        time.sleep(0.25)
        status.write("4/4 Saving prediction to SQLite")
        save_prediction(values, result)
        status.update(label="Prediction complete", state="complete", expanded=False)

        st.session_state["latest_result"] = result

    result = st.session_state.get("latest_result")
    if result:
        st.divider()
        st.subheader("Prediction Result")

        a, b, c = st.columns(3)
        a.metric("Predicted Churn", result["prediction"])
        b.metric("Churn Probability", f'{result["confidence_percent"]:.1f}%')
        c.metric("Risk Level", result["risk_level"])

        risk_class = {
            "HIGH": "risk-high",
            "MEDIUM": "risk-medium",
            "LOW": "risk-low",
        }[result["risk_level"]]
        st.markdown(
            f'<div class="{risk_class}"><strong>{result["risk_level"]} RISK</strong><br>'
            f'Primary profile signals: {result["root_cause"]}.</div>',
            unsafe_allow_html=True,
        )

        probability = result["probability"]
        chart_df = pd.DataFrame(
            {
                "Probability": [probability, 1 - probability],
            },
            index=["Churn", "Retained"],
        )
        st.bar_chart(chart_df)

    st.divider()
    st.subheader("Prediction History")
    history = load_history()
    if history.empty:
        st.caption("No stored predictions yet.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            "Download history as CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="churnguard_prediction_history.csv",
            mime="text/csv",
        )

    st.caption(
        "Streamlit Community Cloud uses an ephemeral local filesystem. "
        "SQLite demonstrates database integration, but cloud-written records "
        "may reset when the app restarts or redeploys."
    )


def model():
    page_header(
        "Model",
        "Final model selection and deployment configuration.",
    )

    st.markdown('<span class="model-chip">SELECTED MODEL</span>', unsafe_allow_html=True)
    st.markdown("## Tuned K-Nearest Neighbours")
    st.write(
        "The deployed predictor uses the tuned KNN configuration already selected "
        "for ChurnGuard. Customer features are cleaned, encoded, aligned to the "
        "training feature columns and standardized before inference."
    )

    a, b, c, d = st.columns(4)
    a.metric("n_neighbors", "11")
    b.metric("Metric", "Manhattan")
    c.metric("Weights", "Distance")
    d.metric("p", "1")

    st.subheader("Prediction pipeline")
    st.code(
        """Customer form
   ↓
Validation
   ↓
Missing-value treatment
   ↓
Binary + one-hot encoding
   ↓
Training-column alignment
   ↓
StandardScaler
   ↓
Tuned KNN
   ↓
Churn probability + risk level
   ↓
SQLite prediction record""",
        language="text",
    )

    show_model_status()


def about():
    page_header(
        "About ChurnGuard AI",
        "A customer-churn decision-support prototype.",
    )
    st.write(
        """
        ChurnGuard AI demonstrates how a web application can connect a trained
        machine-learning workflow with an interactive user interface. It uses
        eight customer features: age, gender, tenure, monthly charges,
        total charges, contract type, internet service and tech support.
        """
    )

    a, b = st.columns(2)
    with a:
        st.markdown("### What the system does")
        st.markdown(
            "- Accepts customer data through an interactive form\n"
            "- Runs the tuned KNN model in Python\n"
            "- Produces churn probability and a risk category\n"
            "- Stores prediction records in SQLite\n"
            "- Displays live charts and prediction history"
        )
    with b:
        st.markdown("### Deployment")
        st.markdown(
            "- Source code stored on GitHub\n"
            "- Public app deployed with Streamlit Community Cloud\n"
            "- Python and scikit-learn run on the hosted app\n"
            "- No Flask server configuration is required"
        )


# ---------- Navigation ----------
pages = {
    "Main": [
        st.Page(home, title="Home", icon="🏠", default=True),
        st.Page(about, title="About", icon="ℹ️"),
    ],
    "Analysis": [
        st.Page(overview, title="Overview", icon="📊"),
        st.Page(predictions, title="Predictions", icon="🎯", url_path="predictions"),
    ],
    "Technical": [
        st.Page(model, title="Model", icon="🧠"),
    ],
}

pg = st.navigation(pages)
pg.run()
