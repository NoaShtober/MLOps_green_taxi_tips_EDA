# Green Taxi Tip Prediction - MLOps Pipeline #
An end-to-end MLOps pipeline built with Metaflow and MLflow to manage, monitor, and automate the training of a machine learning model that predicts taxi trip tip amounts for New York City Green Taxis.

The project integrates rigid data quality gates (Schema & Type checks) alongside soft data drift monitoring powered by NannyML, orchestrating a robust champion/candidate model promotion lifecycle.
## peline Architecture ## 
The pipeline is structured as a Directed Acyclic Graph (DAG) using Metaflow, comprising the following operational steps:

                     [ start ]
                         |
                    [ load_data ]
                         |
                 [ integrity_gate ]
                    /          \
         (Hard Fail)            (Pass)
                  /              \
            [ end ]       [ feature_engineering ]
                                   |
                            [ load_champion ]
                               /         \
                 (Bootstrapped)           (Evaluate)
                             /             \
                       [ end ]         [ model_gate ]
                                              |
                                          [ retrain ]
                                              |
                                      [ promotion_gate ]
                                              |
                                           [ end ]
start: Initializes the pipeline run and logs configuration parameters.

load_data: Loads historical reference data and the incoming batch dataset, standardizing schemas and cleaning invalid time ranges.

integrity_gate: Evaluates data health using a two-layer validation strategy (Schema constraints + NannyML drift detection).

Branching: If a hard schema failure is detected, the batch is rejected, and the pipeline fast-fails straight to end.

feature_engineering: Transforms raw tabular data into model-ready features (extracting temporal components, trip durations, and applying log transformations).

load_champion: Connects to the MLflow Model Registry to pull the current @champion model. If no model exists, it triggers an automated bootstrapping flow to train an initial baseline.

model_gate: Evaluates the champion model's performance on the new batch data and calculates metrics against historical baselines to decide if retraining is necessary.

retrain: Conditionally trains a new RandomForestRegressor (Candidate) on the combined historical and newly received batch data if performance degradation or data drift is flagged.

promotion_gate: Assesses whether the candidate model outperforms the active champion by a defined margin. If successful, it updates the registry aliases to promote the new model.

end: Finalizes logs, dumps decision manifests, and terminates the pipeline execution.

## Tech Stack & Frameworks ##
Metaflow: Orchestrates the pipeline steps, state management, and workflow transitions.

MLflow Tracking & Registry: Logs parameters, performance metrics, and generated artifacts, while serving as the central model registry utilizing production tags and aliases (@champion).

NannyML: Measures multi-variate statistical data drift using the Jensen-Shannon distance metric.

Scikit-Learn: Drives core modeling operations via RandomForestRegressor.

## Two-Layer Integrity Gate ##
To guarantee data quality before features ever hit training or inference, the integrity_gate employs two distinct validation mechanisms:

1. Hard Rules (Fail-Fast Gate)
Triggers an immediate pipeline halt and batch rejection if:

Any critical column from the expected schema (EXPECTED_SCHEMA) is missing.

There are strict data type mismatches that could break downstream code.

2. Soft Rules (Warning & Retrain Gate)
Logs warnings and automatically flags the pipeline for a model retraining recommendation (retrain_needed = True) if:

Range Violations: Numerical attributes deviate past realistic boundaries specified in RANGE_SPECS (e.g., negative fares or extreme passenger counts).

Data Drift: NannyML's UnivariateDriftCalculator detects a significant statistical shift in core features: trip_distance, pulocationid, or dolocationid.

## Retraining & Promotion Logic ##
Retraining Criteria:
Retraining is recommended if the current champion's error (RMSE) on the incoming batch exceeds 
2.0
, if a soft integrity warning is raised, or if the model's error increases by more than 
10%
 compared to historical reference baselines.

Promotion Criteria:
A newly trained candidate model will only displace the reigning champion if it shows a noticeable performance gain on the latest data batch. This is governed by the min-improvement parameter (defaulting to 1%):


## Running the Project ##

Prerequisites
Ensure your local MLflow tracking server is running in the background prior to starting a run:

mlflow server --host 127.0.0.1 --port 5000

## Running the Pipeline ## 
Execute a standard run using default parameter configurations:


python taxi_tip_mlops_pipeline.py run
Execute a run with customized target paths for your streaming/batch files and reference baselines:


python taxi_tip_mlops_pipeline.py run \
  --batch-path "data/green_tripdata_2020-04.parquet" \
  --reference-path "data/green_tripdata_2020-01.parquet"
The experiment you will run is called Green_Taxi_Tip_Demo3
Resuming from Failures
If the pipeline fails midway through execution (e.g., a network hiccup or an intentional debugging failure in the retrain step), you can resume the exact state graph right where it broke after fixing the issue:


python taxi_tip_mlops_pipeline.py resume retrain

## Artifacts & Tracking ## 
Every major step serializes and attaches diagnostic JSON summaries directly to the corresponding MLflow tracking run:

decision.json: Stores structural workflow decisions (e.g., batch rejection reasons, model evaluation statistics, and promotion details).

nannyml_report.json: Holds complete breakdown logs of statistical drift alarms and schema range violations.

feature_spec.json: Documents explicit feature column counts and internal data types processed by the pipeline.
