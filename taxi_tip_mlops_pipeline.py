import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from metaflow import FlowSpec, Parameter, step, current
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import nannyml as nml
import logging
from enum import IntEnum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
class PaymentType(IntEnum):
    CREDIT_CARD = 1
    CASH = 2
    NO_CHARGE = 3
    DISPUTE = 4
    UNKNOWN = 5
    VOIDED_TRIP = 6

EXPECTED_SCHEMA = {
    "ehail_fee": "object",
    "ratecodeid": "float64",
    "store_and_fwd_flag": "object",
    "trip_type": "float64",
    "payment_type": "float64",
    "passenger_count": "float64",
    "congestion_surcharge": "float64",
    "dolocationid": "int64",
    "pulocationid": "int64",
    "lpep_pickup_datetime": "datetime64[us]",
    "lpep_dropoff_datetime": "datetime64[us]",
    "vendorid": "int64",
    "extra": "float64",
    "fare_amount": "float64",
    "trip_distance": "float64",
    "tolls_amount": "float64",
    "tip_amount": "float64",
    "mta_tax": "float64",
    "total_amount": "float64",
    "improvement_surcharge": "float64",
}

RANGE_SPECS = [
    ("trip_distance", 0.0, 200.0),
    ("fare_amount", 0.0, 500.0),
    ("tip_amount", 0.0, 200.0),
    ("tolls_amount", 0.0, 200.0),
    ("total_amount", 0.0, 1000.0),
    ("passenger_count", 0.0, 10.0),
    ("duration_min", 0.0, 360.0),
]

# MLflow configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Green_Taxi_Tip_Demo3"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

class TaxiTipMLOpsFlow(FlowSpec):
    # Parameters
    reference_path = Parameter("reference-path", help="Path to reference dataset", default=None)
    batch_path = Parameter("batch-path", help="Path to batch dataset", default="data/green_tripdata_2020-04.parquet")
    model_name = Parameter("model-name", default="green_taxi_tip_model")
    min_improvement = Parameter("min-improvement", default=0.01) # 1% improvement required

    @step
    def start(self):
        """Step 0: Initialize flow"""
        logger.info(f"Starting flow for batch: {self.batch_path}")
        self.next(self.load_data)

    @step
    def load_data(self):
        """Step A: Load reference and batch datasets"""
        logger.info("Loading datasets...")
        self.batch_df = pd.read_parquet(self.batch_path)
        
        if self.reference_path and os.path.exists(self.reference_path):
            self.ref_df = pd.read_parquet(self.reference_path)
        else:
            logger.info("No reference path provided or file missing. Using batch as reference.")
            self.ref_df = self.batch_df.copy()

        # Standardize column names to lowercase
        self.ref_df.columns = [c.lower() for c in self.ref_df.columns]
        self.batch_df.columns = [c.lower() for c in self.batch_df.columns]
        
        # Basic cleanup: ensure datetime columns are datetime objects
        for df_name in ["ref_df", "batch_df"]:
            df = getattr(self, df_name)
            for col in ["lpep_pickup_datetime", "lpep_dropoff_datetime"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            initial_count = len(df)
            
            # Filter logically invalid rows
            mask = pd.Series(True, index=df.index)
            if "lpep_pickup_datetime" in df.columns and "lpep_dropoff_datetime" in df.columns:
                mask &= df["lpep_dropoff_datetime"] >= df["lpep_pickup_datetime"]
            
            df = df[mask].copy()
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} invalid/out-of-range rows from {df_name}")
            setattr(self, df_name, df)
        
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        """Step B: Integrity gate (Hard rules + NannyML soft gate)."""
        logger.info("Running integrity gate...")
        
        # Layer 1: Hard rules (fail-fast)
        missing_cols = [c for c in EXPECTED_SCHEMA.keys() if c not in self.batch_df.columns]
        
        # Schema type validation
        type_mismatches = []
        if not missing_cols:
            for col, expected_type in EXPECTED_SCHEMA.items():
                actual_type = str(self.batch_df[col].dtype)
                # Allow minor datetime differences (us vs ns)
                if "datetime64" in expected_type and "datetime64" in actual_type:
                    continue
                if expected_type != actual_type:
                    # Check for float vs int compatibility if values are numeric
                    if "float" in expected_type and "int" in actual_type:
                        continue
                    type_mismatches.append(f"{col}: expected {expected_type}, got {actual_type}")

        # Range checks 
        range_violations = []
        range_violations_count = 0
        for col, min_val, max_val in RANGE_SPECS:
            if col in self.batch_df.columns:
                violators_count = 0
                if min_val is not None:
                    violators_count += (self.batch_df[col] < min_val).sum()
                if max_val is not None:
                    violators_count += (self.batch_df[col] > max_val).sum()
                
                if violators_count > 0:
                    range_violations.append(f"{col}: {violators_count} rows")
                    range_violations_count += violators_count

        # Logic to check for dropoff before pickup 
        invalid_times = (self.batch_df["lpep_dropoff_datetime"] < self.batch_df["lpep_pickup_datetime"]).sum() if "lpep_pickup_datetime" in self.batch_df.columns and "lpep_dropoff_datetime" in self.batch_df.columns else 0
        
        # Hard fail only on missing columns or type mismatches
        self.hard_fail = len(missing_cols) > 0 or len(type_mismatches) > 0
        
        # Soft warning on range violations or invalid times
        self.range_warn = range_violations_count > 0 or invalid_times > 0
        self.integrity_warn = self.range_warn # Initial value, will be updated by NannyML
        
        with mlflow.start_run(run_name=f"IntegrityGate_{current.run_id}"):
            mlflow.log_param("batch_path", self.batch_path)
            mlflow.log_metrics({
                "missing_cols_count": len(missing_cols),
                "type_mismatches_count": len(type_mismatches),
                "range_violations_total": range_violations_count,
                "invalid_times_count": int(invalid_times),
                "soft_fail_count": range_violations_count + int(invalid_times)
            })
            
            if self.hard_fail:
                logger.error(f"Hard integrity gate failed!")
                if missing_cols: logger.error(f"Missing: {missing_cols}")
                if type_mismatches: logger.error(f"Type mismatches: {type_mismatches}")
                
                decision = {
                    "action": "reject_batch",
                    "reason": "Hard failure (Schema/Types)",
                    "details": {
                        "missing_cols": missing_cols,
                        "type_mismatches": type_mismatches
                    }
                }
                with open("decision.json", "w") as f:
                    json.dump(decision, f)
                mlflow.log_artifact("decision.json")
                mlflow.set_tag("integrity_status", "failed")
            else:
                # Layer 2: NannyML checks (soft gate)
                logger.info("Running NannyML drift detection...")
                nannyml_report = {
                    "drift_detected": False,
                    "drift_details": {},
                    "range_violations": range_violations,
                    "invalid_times_count": int(invalid_times),
                    "nannyml_status": "skipped"
                }
                
                try:
                    features_to_monitor = ['trip_distance', 'pulocationid', 'dolocationid']
                    calc = nml.UnivariateDriftCalculator(
                        column_names=features_to_monitor,
                        timestamp_column_name='lpep_pickup_datetime',
                        continuous_methods=['jensen_shannon'],
                        categorical_methods=['jensen_shannon']
                    )
                    calc.fit(self.ref_df)
                    results = calc.calculate(self.batch_df)
                    
                    # Robust alert extraction (NannyML columns are (col, method, metric))
                    res_df = results.to_df()
                    drift_details = {}
                    
                    # Alert columns end with 'alert' in the 3rd level of the multi-index
                    # Index 2 corresponds to the 3rd level
                    alert_cols = [c for c in res_df.columns if c[2] == 'alert']
                    for col in alert_cols:
                        if res_df[col].any():
                            feature_name = str(col[0])
                            drift_details[feature_name] = "Drift alert triggered"
                            self.integrity_warn = True
                    
                    nannyml_report.update({
                        "drift_detected": bool(self.integrity_warn),
                        "drift_details": drift_details,
                        "nannyml_status": "success"
                    })
                    
                    if self.integrity_warn:
                        logger.warning("Data drift or range violations detected!")
                            
                except Exception as e:
                    logger.error(f"NannyML failed: {e}")
                    nannyml_report["nannyml_status"] = f"error: {str(e)}"
                    mlflow.set_tag("nannyml_error", str(e))

                # Always log the report artifact
                report_path = "nannyml_report.json"
                with open(report_path, "w") as f:
                    json.dump(nannyml_report, f, indent=4)
                mlflow.log_artifact(report_path)
                
                mlflow.set_tag("integrity_warn", str(self.integrity_warn).lower())
                mlflow.log_metrics({
                    "drift_alerts_count": len(nannyml_report.get("drift_details", {})),
                    "integrity_warn": 1.0 if self.integrity_warn else 0.0
                })

                mlflow.set_tag("integrity_status", "passed")
        
        self.branch = "fail" if self.hard_fail else "pass"
        self.next({"fail": self.end, "pass": self.feature_engineering}, condition="branch")


    def _engineer_df(self, df):
        """Transform the DataFrame into model-ready features
        Features were selected, as they likely to impact the tip amount
        """
        out = df.copy()
        
        # Time features (columns already lowercased in load_data)
        pickup_dt = pd.to_datetime(out["lpep_pickup_datetime"])
        out["lpep_pickup_hour"] = pickup_dt.dt.hour
        out["lpep_pickup_day_of_week"] = pickup_dt.dt.dayofweek
        out["lpep_pickup_month"] = pickup_dt.dt.month
        
        # Duration of the trip in minutes.
        dropoff_dt = pd.to_datetime(out["lpep_dropoff_datetime"])
        out["duration_min"] = (dropoff_dt - pickup_dt).dt.total_seconds() / 60.0
        
        # Filters (e.g. credit card only for better tip reporting)
        if "payment_type" in out.columns:
            out = out[out["payment_type"] == PaymentType.CREDIT_CARD].copy()
        
        X = pd.DataFrame()
        X["lpep_pickup_hour"] = out["lpep_pickup_hour"]
        X["lpep_pickup_day_of_week"] = out["lpep_pickup_day_of_week"]
        X["lpep_pickup_month"] = out["lpep_pickup_month"]
        X["PULocationID"] = out["pulocationid"]
        X["DOLocationID"] = out["dolocationid"]
        X["passenger_count"] = out["passenger_count"]
        X["trip_distance"] = np.log1p(out['trip_distance'].clip(lower=0, upper=100))
        X["duration_min"] = np.log1p(out['duration_min'].clip(lower=0, upper=360))
        
        features = X.columns.tolist()
        
        # Target
        y = out["tip_amount"].values
        
        # Cast to float64 to avoid type issues in MLflow
        X = X.astype("float64").fillna(0)
        
        return X, y, features

    @step
    def feature_engineering(self):
        """Step C: Feature engineering"""
        logger.info("Engineering features...")
        self.X_ref, self.y_ref, self.feature_names = self._engineer_df(self.ref_df)
        self.X_batch, self.y_batch, _ = self._engineer_df(self.batch_df)
        
        with mlflow.start_run(run_name=f"FeatureEngineering_{current.run_id}"):
            feature_spec = {
                "features": self.feature_names,
                "count": len(self.feature_names),
                "dtypes": {col: str(self.X_ref[col].dtype) for col in self.feature_names}
            }
            with open("feature_spec.json", "w") as f:
                json.dump(feature_spec, f, indent=4)
            mlflow.log_artifact("feature_spec.json")
            
        self.next(self.load_champion)

    @step
    def load_champion(self):
        """Step D: Load champion model from registry
        If no model exists, bootstrap by training on the batch and registering as champion
        """
        logger.info(f"Loading champion model: {self.model_name}")
        client = MlflowClient()
        self.champion_version = None
        self.champion_model = None
        self.bootstrapped = False
        
        try:
            # Try to get model by alias
            model_v = client.get_model_version_by_alias(self.model_name, "champion")
            self.champion_version = model_v.version
            model_uri = f"models:/{self.model_name}@champion"
            self.champion_model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded champion version {self.champion_version}")
        except Exception as e:
            logger.info(f"No champion model found: {e}. Starting automated bootstrap...")
            
            # --- BOOTSTRAP IN LOAD_CHAMPION ---
            with mlflow.start_run(run_name=f"Bootstrap_{current.run_id}") as run:
                # Train on X_ref (which is batch if no ref was provided)
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                model.fit(self.X_ref, self.y_ref)
                
                # Log and register
                mlflow.sklearn.log_model(model, "model")
                run_id = run.info.run_id
                model_uri = f"runs:/{run_id}/model"
                mv = mlflow.register_model(model_uri, self.model_name)
                
                # Promote to champion
                client.set_registered_model_alias(self.model_name, "champion", mv.version)
                client.set_model_version_tag(self.model_name, mv.version, "role", "champion")
                client.set_model_version_tag(self.model_name, mv.version, "promotion_reason", "bootstrap")
                client.set_model_version_tag(self.model_name, mv.version, "validation_status", "approved")
                
                logger.info(f"Successfully bootstrapped and registered version {mv.version} as @champion.")
                self.bootstrapped = True
                self.champion_model = model
                self.champion_version = mv.version
                
                # Set baseline tags for demo consistency
                mlflow.set_tags({
                    "retrain_recommended": "false",
                    "promotion_recommended": "false"
                })

                decision = {
                    "action": "bootstrap_and_promote",
                    "reason": "No champion model existed",
                    "retrain_recommended": False,
                    "promotion_recommended": False
                }
                with open("decision.json", "w") as f:
                    json.dump(decision, f, indent=4)
                mlflow.log_artifact("decision.json")

        self.branch_load = "bootstrap" if self.bootstrapped else "evaluate"
        self.next({"bootstrap": self.end, "evaluate": self.model_gate}, condition="branch_load")

    @step
    def model_gate(self):
        """Step E: Model performance gate."""
        logger.info("Evaluating champion model performance...")
        self.retrain_needed = False
        self.rmse_champion = float('inf')
        self.rmse_baseline = float('inf')
        self.rmse_increase_pct = 0.0
        
        with mlflow.start_run(run_name=f"ModelGate_{current.run_id}"):
            if self.champion_model:
                # Evaluate on full batch (new unseen data)
                preds_batch = self.champion_model.predict(self.X_batch)
                self.rmse_champion = float(np.sqrt(mean_squared_error(self.y_batch, preds_batch)))
                
                # Evaluate on reference (historical baseline)
                preds_ref = self.champion_model.predict(self.X_ref)
                self.rmse_baseline = float(np.sqrt(mean_squared_error(self.y_ref, preds_ref)))
                
                # Compute increase percentage
                if self.rmse_baseline > 0:
                    self.rmse_increase_pct = ((self.rmse_champion - self.rmse_baseline) / self.rmse_baseline) * 100
                
                mlflow.log_metric("rmse_champion", self.rmse_champion)
                mlflow.log_metric("rmse_baseline", self.rmse_baseline)
                mlflow.log_metric("rmse_increase_pct", self.rmse_increase_pct)
                
                logger.info(f"Champion RMSE on batch: {self.rmse_champion:.4f}")
                logger.info(f"Champion RMSE baseline (ref): {self.rmse_baseline:.4f}")
                logger.info(f"RMSE increase pct: {self.rmse_increase_pct:.2f}%")
                
                # Logic to decide retraining
                if self.rmse_champion > 2.0 or self.integrity_warn or self.rmse_increase_pct > 10.0:
                    self.retrain_needed = True
                    logger.info("Retraining recommended.")
            else:
                logger.info("No champion exists. Retraining required (Bootstrap).")
                self.retrain_needed = True
            
            mlflow.set_tag("retrain_recommended", str(self.retrain_needed).lower())
            
            decision = {
                "action": "evaluate",
                "rmse_champion": self.rmse_champion,
                "rmse_baseline": self.rmse_baseline,
                "rmse_increase_pct": self.rmse_increase_pct,
                "retrain_recommended": self.retrain_needed
            }
            with open("decision.json", "w") as f:
                json.dump(decision, f)
            mlflow.log_artifact("decision.json")
            
        self.next(self.retrain)

    @step
    def retrain(self):
        """Step F: Retrain model (conditional)."""
        self.candidate_model = None
        self.rmse_candidate = float('inf')
        
        if self.retrain_needed:
            # For demo 3 : failing point 
            # To demo failure/resumption: Uncomment the next line, run the flow, 
            # then comment it back and run 'python taxi_tip_mlops_pipeline.py resume retrain'
            #raise Exception("Simulated failure in retrain step for robustness demo.")

            logger.info("Retraining candidate model on full combined Ref + Batch data...")
            with mlflow.start_run(run_name=f"Retrain_{current.run_id}") as run:
                self.candidate_run_id = run.info.run_id
                
                # Concatenate the full Reference and full Batch data
                X_train_combined = pd.concat([self.X_ref, self.X_batch])
                y_train_combined = np.concatenate([self.y_ref, self.y_batch])
                
                # Model Choice: RandomForest as an example
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                model.fit(X_train_combined, y_train_combined)
                
                self.candidate_model = model
                
                # Evaluate the candidate on the full batch data
                preds = model.predict(self.X_batch)
                self.rmse_candidate = float(np.sqrt(mean_squared_error(self.y_batch, preds)))
                
                mlflow.log_metric("rmse_candidate", self.rmse_candidate)
                # Log model but don't register yet (will do in promotion step)
                mlflow.sklearn.log_model(model, "model")
                logger.info(f"Candidate RMSE on batch: {self.rmse_candidate:.4f}")
        else:
            logger.info("Skipping retraining.")
            
        self.next(self.promotion_gate)

    @step
    def promotion_gate(self):
        """Step G: Candidate acceptance and promotion."""
        logger.info("Running promotion gate...")
        self.promoted = False
        
        with mlflow.start_run(run_name=f"PromotionGate_{current.run_id}"):
            if self.retrain_needed and self.candidate_model:
                # Promotion logic
                is_better = self.rmse_candidate < self.rmse_champion * (1 - self.min_improvement)
                
                # Bootstrap case: if no champion, always promote
                if not self.champion_model or is_better:
                    logger.info("Promoting candidate model!")
                    self.promoted = True
                    
                    # Register and promote
                    client = MlflowClient()
                    
                    # Log and register model
                    model_uri = f"runs:/{self.candidate_run_id}/model"
                    mv = mlflow.register_model(model_uri, self.model_name)
                    
                    # Set tags
                    client.set_model_version_tag(self.model_name, mv.version, "role", "candidate")
                    client.set_model_version_tag(self.model_name, mv.version, "validation_status", "approved")
                    
                    # Flip alias
                    client.set_registered_model_alias(self.model_name, "champion", mv.version)
                    client.set_model_version_tag(self.model_name, mv.version, "role", "champion")
                    client.set_model_version_tag(self.model_name, mv.version, "promotion_reason", "bootstrap" if not self.champion_model else "performance_improvement")
                    
                    if self.champion_version:
                        client.set_model_version_tag(self.model_name, self.champion_version, "role", "previous_champion")
                else:
                    logger.info("Candidate did not beat champion meaningfully.")
            
            mlflow.set_tag("promotion_recommended", str(self.promoted).lower())
            
            decision = {
                "criteria": f"rmse_candidate < rmse_champion * (1 - {self.min_improvement})",
                "rmse_champion": self.rmse_champion,
                "rmse_candidate": self.rmse_candidate,
                "action": "promote" if self.promoted else "maintain",
                "reason": "Improved performance" if self.promoted else "Insufficient improvement"
            }
            with open("decision.json", "w") as f:
                json.dump(decision, f)
            mlflow.log_artifact("decision.json")

        self.next(self.end)

    @step
    def end(self):
        """Step H: End of flow."""
        logger.info("Flow completed.")

if __name__ == "__main__":
    TaxiTipMLOpsFlow()
