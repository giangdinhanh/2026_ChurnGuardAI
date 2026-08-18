from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

MODEL_PARAMS = {
    "metric": "manhattan",
    "n_neighbors": 11,
    "p": 1,
    "weights": "distance",
}


class ChurnModelService:
    """Train and serve the tuned ChurnGuard KNN model."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.ready = False
        self.error = None
        self.train_rows = 0
        self.test_rows = 0
        self._load_and_train()

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        required_columns = [
            "CustomerID",
            "Age",
            "Gender",
            "Tenure",
            "MonthlyCharges",
            "TotalCharges",
            "ContractType",
            "InternetService",
            "TechSupport",
            "Churn",
        ]

        # Normalize CSV header names.
        df = df.copy()
        df.columns = [
            str(column).replace("\ufeff", "").strip()
            for column in df.columns
        ]

        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{missing}. Found columns: {list(df.columns)}"
            )

        model_df = df[required_columns].copy()

        # Strip whitespace from text fields.
        text_columns = model_df.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in text_columns:
            model_df[column] = (
                model_df[column]
                .astype("string")
                .str.strip()
            )

        # Remove fully empty rows.
        model_df = model_df.dropna(how="all")

        # Remove repeated header rows accidentally embedded in the CSV body.
        repeated_header_mask = pd.Series(True, index=model_df.index)
        for column in required_columns:
            repeated_header_mask &= (
                model_df[column].astype("string").str.strip() == column
            )
        model_df = model_df.loc[~repeated_header_mask].copy()

        # Keep only valid target labels.
        model_df = model_df[
            model_df["Churn"].isin(["Yes", "No"])
        ].copy()

        if model_df.empty:
            raise ValueError(
                "No valid training rows remain after cleaning. "
                "The Churn column must contain Yes/No values."
            )

        # Convert numeric fields safely.
        numeric_columns = [
            "Age",
            "Tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]

        for column in numeric_columns:
            model_df[column] = pd.to_numeric(
                model_df[column],
                errors="coerce",
            )

        # Missing internet service is treated as its own category.
        model_df["InternetService"] = model_df["InternetService"].replace(
            {"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA, "": pd.NA}
        )
        model_df["InternetService"] = model_df["InternetService"].fillna(
            "No Internet Service"
        )

        # Fill numeric missing values.
        for column in numeric_columns:
            median_value = model_df[column].median()
            if pd.isna(median_value):
                raise ValueError(
                    f"Column {column} has no usable numeric values."
                )
            model_df[column] = model_df[column].fillna(median_value)

        # Fill categorical missing values.
        categorical_columns = [
            "Gender",
            "ContractType",
            "InternetService",
            "TechSupport",
        ]

        for column in categorical_columns:
            model_df[column] = model_df[column].replace(
                {"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA, "": pd.NA}
            )
            if model_df[column].isna().any():
                modes = model_df[column].mode(dropna=True)
                if modes.empty:
                    raise ValueError(
                        f"Column {column} has no usable categorical values."
                    )
                model_df[column] = model_df[column].fillna(modes.iloc[0])

        model_df = model_df.drop_duplicates().reset_index(drop=True)

        # Validate class counts before stratified splitting.
        class_counts = model_df["Churn"].value_counts()

        if len(class_counts) < 2:
            raise ValueError(
                "The dataset must contain both Churn='Yes' and Churn='No'. "
                f"Found: {class_counts.to_dict()}"
            )

        if (class_counts < 2).any():
            raise ValueError(
                "Each churn class needs at least 2 valid rows for stratified training. "
                f"Found: {class_counts.to_dict()}"
            )

        return model_df

    @staticmethod
    def _encode(dataframe: pd.DataFrame) -> pd.DataFrame:
        encoded = dataframe.copy()

        encoded["Gender"] = encoded["Gender"].map(
            {"Male": 1, "Female": 0}
        )
        encoded["TechSupport"] = encoded["TechSupport"].map(
            {"Yes": 1, "No": 0}
        )

        if "Churn" in encoded.columns:
            encoded["Churn"] = encoded["Churn"].map(
                {"Yes": 1, "No": 0}
            )

        encoded = pd.get_dummies(
            encoded,
            columns=["ContractType", "InternetService"],
            drop_first=False,
        )

        if "CustomerID" in encoded.columns:
            encoded = encoded.drop(columns=["CustomerID"])

        return encoded

    def _load_and_train(self):
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"{self.data_path.name} was not found. "
                    "Add it to the repository beside app.py."
                )

            df = pd.read_csv(self.data_path)
            model_df = self._clean_dataframe(df)

            train_df, test_df = train_test_split(
                model_df,
                test_size=0.20,
                random_state=RANDOM_STATE,
                stratify=model_df["Churn"],
            )

            train_encoded = self._encode(train_df)
            test_encoded = self._encode(test_df)

            X_train = train_encoded.drop(columns=["Churn"])
            y_train = train_encoded["Churn"]

            X_test = (
                test_encoded
                .drop(columns=["Churn"])
                .reindex(columns=X_train.columns, fill_value=0)
            )

            self.feature_columns = list(X_train.columns)

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.scaler.transform(X_test)

            self.model = KNeighborsClassifier(**MODEL_PARAMS)
            self.model.fit(X_train_scaled, y_train)

            self.train_rows = len(train_df)
            self.test_rows = len(test_df)
            self.ready = True
            self.error = None

        except Exception as exc:
            self.ready = False
            self.error = str(exc)

    def predict(self, values: dict) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "Model is not ready.")

        row = pd.DataFrame(
            [
                {
                    "CustomerID": "WEB-INPUT",
                    "Age": float(values["age"]),
                    "Gender": values["gender"],
                    "Tenure": float(values["tenure"]),
                    "MonthlyCharges": float(values["monthly_charges"]),
                    "TotalCharges": float(values["total_charges"]),
                    "ContractType": values["contract_type"],
                    "InternetService": values["internet_service"],
                    "TechSupport": values["tech_support"],
                }
            ]
        )

        encoded = self._encode(row).reindex(
            columns=self.feature_columns,
            fill_value=0,
        )

        scaled = self.scaler.transform(encoded)
        prediction = int(self.model.predict(scaled)[0])

        probabilities = self.model.predict_proba(scaled)[0]
        churn_class_index = list(self.model.classes_).index(1)
        churn_probability = float(probabilities[churn_class_index])

        if churn_probability >= 0.70:
            risk_level = "HIGH"
        elif churn_probability >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        reasons = []

        if values["contract_type"] == "Month-to-Month":
            reasons.append("month-to-month contract")
        if float(values["monthly_charges"]) >= 80:
            reasons.append("higher monthly charges")
        if float(values["tenure"]) <= 12:
            reasons.append("short customer tenure")
        if values["tech_support"] == "No":
            reasons.append("no tech support")

        root_cause = (
            ", ".join(reasons[:3])
            if reasons
            else "combined customer profile signals"
        )

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": churn_probability,
            "confidence_percent": round(churn_probability * 100, 1),
            "risk_level": risk_level,
            "root_cause": root_cause,
            "model_name": "Tuned K-Nearest Neighbours",
            "model_params": MODEL_PARAMS,
        }
