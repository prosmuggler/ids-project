# AI-Powered Intrusion Detection and Anomaly Analysis for IoT/WSN

A centralized intrusion detection system (IDS) that classifies IoT/WSN traffic as benign or as a specific attack (DoS/DDoS, probing, spoofing, flooding) and flags abnormal behavior it has never seen before. The project compares classical ML and deep learning models on detection accuracy, false-positive rate, computational cost and real-time suitability, then deploys the best one on an edge device (Raspberry Pi).

## Table of Contents

- [Objectives](#objectives)
- [Datasets](#datasets)
- [Setup](#setup)
- [Repository Layout](#repository-layout)
- [Preprocessing Rules](#preprocessing-rules)
- [Workflow](#workflow)
- [Metrics](#metrics)
- [Experiment Hygiene](#experiment-hygiene)
- [Safety Notice](#safety-notice)
- [Results](#results)

## Objectives

1. **Data pipeline and feature selection**: build a clean, leakage-free pipeline and select compact feature sets (15–30 features).
2. **ML vs DL benchmark**: compare classifiers for known attacks on both detection quality and cost.
3. **Anomaly detection**: detect unknown (zero-day) attacks and abnormal behavior with anomaly detectors and a hybrid detector.
4. **Edge deployment**: run a lightweight, explainable detector in real time on an edge gateway.

Work through the objectives in order. Each one ends with a deliverable to review with your supervisor before moving on.

## Datasets

| Dataset | Environment | Attacks | Used for | Priority |
| --- | --- | --- | --- | --- |
| [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | 105 real IoT devices | DDoS, DoS, recon, spoofing, web, brute force, Mirai (33 attacks, 7 categories) | Main experiments (Objectives 1, 2, 4) | Required |
| [WSN-DS](https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds) | Simulated LEACH-based WSN | Blackhole, grayhole, flooding, scheduling | WSN-specific results | Required |
| N-BaIoT or TON_IoT | IoT device traffic | Mirai, BASHLITE and others | Anomaly and zero-day experiments (Objective 3) | Required for anomaly work |

Raw files are large (tens of GB). Develop on a **stratified 5–10% sample**, then scale up for final runs. Write a one-page data card per dataset (rows per class, features kept, what was dropped and why).

## Setup

### Software

- Python 3.10+ in a virtual environment (conda or venv)
- **Core:** pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn
- **Models:** xgboost, lightgbm, catboost, torch (PyTorch is the default DL library)
- **Tuning and tracking:** optuna, mlflow (or Weights & Biases)
- **Explainability:** shap, lime
- **Traffic tools:** Wireshark/tshark, CICFlowMeter or nfstream, tcpreplay, hping3, nmap

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn \
            xgboost lightgbm catboost torch optuna mlflow shap lime \
            nfstream onnx onnxruntime pyarrow streamlit
```

### Hardware

- GPU machine, or Google Colab / Kaggle, for training
- 2–6 Raspberry Pi 4 boards (4 GB or more) for edge measurements; Jetson Nano optional
- USB power meter or INA219 sensor for energy measurement
- An isolated LAN or switch for any live attack traffic

## Repository Layout

```
ids-project/
  data/raw/          # downloaded datasets (not committed to git)
  data/processed/    # cleaned parquet files
  src/data/          # loaders, cleaning, splitting, scaling
  src/models/        # classifiers and anomaly detectors
  src/train/         # training and evaluation scripts
  src/deploy/        # edge inference and dashboard
  configs/           # one YAML file per experiment
  results/           # metrics CSVs, plots, logs
  notebooks/         # exploration only
```

## Preprocessing Rules

- **Drop identifier columns** (source/destination IP, MAC address, timestamp, flow ID). Keeping them leaks the label and produces fake ~99.9% accuracy.
- Remove duplicate rows, NaN and infinite values, and constant columns. Log how many rows each step removes.
- Keep three label levels: **binary** (benign vs attack), **category** (DoS, DDoS, recon, spoofing, ...) and **attack_type** (specific attack).
- Split 70/15/15 (stratified) into train/validation/test **before** any scaling or resampling. Fit the scaler and SMOTE on the training split only.
- Where the data allows, also build a time-based split (train on earlier captures, test on later ones).

## Workflow

### Objective 1: Data pipeline and feature selection

- Download CICIoT2023, WSN-DS and N-BaIoT; load a 5–10% stratified sample for development.
- Explore class counts, feature types, missing values and duplicates; plot class distribution on a log scale.
- Implement `src/data/clean.py` (identifier columns, duplicates, NaN/inf, constant columns) and map CICIoT2023's 33 attacks into its 7 categories.
- Build a scikit-learn `Pipeline` with a scaler (MinMax or Standard) fit on the training set only.
- Compare imbalance handling on the training set: class weights, SMOTE/ADASYN, random undersampling.
- Rank features with mutual information, Random Forest importance and RFE; keep top-k for k in {10, 20, 30, all}.
- For WSN-DS, use per-node features (energy consumed, advertisement/join/schedule messages, data sent/received, distance to cluster head).

**Deliverable:** processed Parquet files, a ranked feature list per dataset, and data cards.

### Objective 2: Compare ML and DL classifiers

| Family | Models |
| --- | --- |
| Classical ML | Logistic Regression, Naive Bayes, KNN, Decision Tree, Random Forest, linear SVM (RBF SVM on a subsample only) |
| Boosting | XGBoost, LightGBM, CatBoost |
| Deep learning | MLP, 1D-CNN, LSTM/GRU on flow windows, CNN-LSTM (under 500K parameters) |
| Optional | TabNet, FT-Transformer, GNN on the device communication graph |
| Ensemble | Stacking or soft voting over the top 3 |

- Train all classical and boosting models with defaults, then tune the top 5 with Optuna (50–100 trials, 5-fold stratified CV on the training set).
- Train DL models in PyTorch with early stopping on validation macro-F1; group flows into windows of 10–50 for LSTM/GRU.
- Evaluate on the untouched test set at binary, category and attack-type levels.
- Repeat final runs with 5 seeds (mean ± std), then run a Friedman test with Nemenyi post-hoc and draw a critical-difference diagram.
- Plot confusion matrices for the best 3 models and note which classes get confused (e.g. DoS vs DDoS).

**Deliverable:** results table (models x datasets x label level), cost table, confusion matrices, and a short write-up recommending 2–3 models to carry forward.

### Objective 3: Anomaly detection for unknown attacks

- Train semi-supervised detectors on **benign data only**: Isolation Forest, One-Class SVM (subsample), Local Outlier Factor, dense autoencoder, LSTM autoencoder.
- Set thresholds from benign validation data (95th, 99th, 99.5th percentile of reconstruction error) and report how FPR changes.
- Build the **hybrid detector**:
  1. The anomaly detector flags suspicious flows.
  2. The best Objective 2 classifier labels the flagged flows.
  3. If classifier confidence is below a threshold (e.g. 0.6), report the flow as an **unknown attack**.
- Run **leave-one-attack-out** tests for every category and compare supervised-only, anomaly-only and hybrid systems.
- On N-BaIoT, train one autoencoder per device and test on its Mirai/BASHLITE traffic.
- On WSN-DS, model per-node behavior over rounds to catch blackhole/grayhole nodes (sudden drop in forwarded vs received data).

**Deliverable:** leave-one-attack-out results table and a three-system comparison.

### Objective 4: Real-time, lightweight, explainable deployment

- Compress the 2–3 carried-forward models: INT8 post-training quantization, magnitude pruning (30–70%), knowledge distillation into a small MLP or shallow tree.
- Export to ONNX (or TFLite) and confirm test macro-F1 drops by less than 1 point.
- Build the Raspberry Pi pipeline: **nfstream capture -> features in 1–5 s windows -> ONNX Runtime inference -> alert**.
- Replay test PCAPs with `tcpreplay` at 1x, 10x and 100x to find the highest rate with no dropped flows.
- Measure packet-to-alert latency, CPU %, RAM and energy per 1,000 inferences for each model.
- Add SHAP explanations: a global importance plot and the top 5 contributing features for each alert.
- Build a dashboard (Streamlit or Grafana) showing live alerts, attack type, confidence and explanation.
- *Optional (isolated testbed only):* live attacks with hping3 (SYN flood), nmap (probing), arpspoof (spoofing); WSN simulation in NS-3 or Cooja.

**Deliverable:** a working edge demo (video), a table of accuracy vs latency vs energy per compressed model, and SHAP figures.

## Metrics

- **Detection:** accuracy, precision, recall (detection rate), **macro-F1**, **false-positive rate**, per-class F1, ROC-AUC, PR-AUC.
- **Cost:** training time, inference latency (ms/flow at batch size 1 and 1024), throughput (flows/s), model size (MB), peak RAM (MB).
- **Anomaly detection:** detection rate on unseen attacks, FPR on benign traffic, ROC-AUC, and the gain of the hybrid system over each component.

## Experiment Hygiene

- Put every run's settings in a config file under `configs/` and log it with MLflow. Never hard-code hyperparameters in notebooks.
- Fix random seeds and run each final configuration with 3–5 seeds.
- Save processed data as Parquet.
- Commit code weekly with short, descriptive messages.

## Safety Notice

> **Never run attack tools (hping3, nmap, arpspoof, etc.) on the campus network or the internet.** Use an isolated LAN or switch only.

## Results

_To be filled in as objectives are completed._

| Objective | Deliverable | Status |
| --- | --- | --- |
| 1 | Parquet files, ranked features, data cards | ☐ |
| 2 | Benchmark tables, confusion matrices, model recommendation | ☐ |
| 3 | Leave-one-attack-out table, system comparison | ☐ |
| 4 | Edge demo video, latency/energy table, SHAP figures | ☐ |
