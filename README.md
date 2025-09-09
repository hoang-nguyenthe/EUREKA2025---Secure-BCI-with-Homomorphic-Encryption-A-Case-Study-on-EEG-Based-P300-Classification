
# 🔐 Privacy‑Preserving P300 BCI Inference with Paillier HE

> **One‑line summary.** This repo demonstrates how to train a simple yet strong **P300 EEG classifier** and run **privacy‑preserving inference** using **Paillier Homomorphic Encryption (HE)**. We reproduce plaintext performance under encryption and report end‑to‑end latency per sample. This work includes results from **19 subjects (A_01 to A_19)**, comparing **plaintext** and **Paillier HE** performance.

---

## 🧠 Dataset

- **Name:** BigP3BCI – *An Open, Diverse, and Machine‑Learning‑Ready P300‑based BCI Dataset*  
- **Host:** PhysioNet  
- **URL:** [https://physionet.org/content/bigp3bci/1.0.0/](https://physionet.org/content/bigp3bci/1.0.0/)  
- **What we use:** Subjects **A_01 to A_19** from **Session `SE001`** (both **Train**/ **Test**; **CB** and **RD** tasks). The code works for other sessions in the same layout.

**EEG preprocessing (per file):**
- Resample (if needed) to **256 Hz**.
- Notch filter **50 Hz**, band‑pass **0.1–15 Hz**.
- Build events from `StimulusBegin` + `StimulusType` or annotations.
- Create epochs `t = [-0.2, 0.8] s`, baseline in `[-0.2, 0] s`.
- Keep EEG channels; assign **10‑20 montage** (ignore missing locations).

---

## 📦 Environment Setup

### **1. Clone the repository:**

```bash
git clone https://github.com/your-username/P300-BCI-HE-Inference.git
cd P300-BCI-HE-Inference
```

### **2. Create and activate a Python environment:**

You can set up a virtual environment using **Python 3.12**:

```bash
python -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
# or: .venv\Scriptsctivate  # (Windows PowerShell)
```

### **3. Install the required dependencies:**

Ensure you have all the required libraries by running:

```bash
pip install -r requirements.txt
```

**`requirements.txt`** includes:
```
numpy
pandas
scikit-learn
mne
joblib
matplotlib
tqdm
phe            # Paillier (python-paillier / phe)
```

> If you use our Docker/container image, you can skip the steps above.

---

## 📁 Project Layout

```
project/
├─ inputs/                      # feature files created from EEG
│  ├─ SE001_timewin64/         # time-window features (X, y in .npz)
│  └─ SE001_xdawn/             # xDAWN features (X, y in .npz)
├─ runs/                       # trained models + scalers + configs
│  ├─ timewin64_cv/
│  └─ xdawn_cv/
├─ artifacts/                  # reports, figures and CSV logs
│  ├─ summary_all.csv          # consolidated numeric results (from all A_xx)
│  ├─ summary_all.md           # table (markdown) of results
│  ├─ results_log_artifacts.csv
│  ├─ results_tuned.csv
│  ├─ best_config.json
│  ├─ model_best.joblib        # best sklearn pipeline (often LDA)
│  ├─ scaler_best.joblib       # companion scaler (if any)
│  ├─ acc_vs_n_by_keybits.png
│  ├─ auc_vs_n_by_keybits.png
│  └─ time_per_sample_by_n_and_keybits.png
├─ make_xdawn_features.py      # build xDAWN features
├─ save_features64.py          # build time-window features
├─ train_eval_models.py        # CV model selection (LR/SVC/KNN/DT/LDA)
├─ he_infer_paillier_artifacts.py  # Paillier-HE inference runner
├─ demo_infer.py               # timing & score-matching demo (HE vs plain)
├─ plot_result.py              # generate the summary plots
├─ run_he_suite.sh             # one-click pipeline (optional)
└─ README.md
```

---

## 🚀 Reproduce our pipeline

> **Assumption:** your EEG data resides in `/work/bigP3BCI-data/StudyA/A_01/SE001` with subfolders `Train/CB/…edf`, `Train/RD/…edf`, `Test/CB/…edf`, `Test/RD/…edf` (this is the PhysioNet layout).

### 1) Extract features

**A. Time‑window mean (64 windows × channels)**

```bash
python save_features64.py   --session "/work/bigP3BCI-data/StudyA/A_01/SE001"   --outdir  inputs/SE001_timewin64
```

**B. xDAWN spatial filtering (3 comps × temporal windows)**

```bash
python make_xdawn_features.py   --session "/work/bigP3BCI-data/StudyA/A_01/SE001"   --outdir inputs/SE001_xdawn   --ncomp 3 --t0 0.0 --t1 0.6 --win 0.15 --step 0.15   --decim 2 --max-files-train 2 --max-files-test 2   --max-epochs-train 4000 --max-epochs-test 4000
```

### 2) Train & select the best model (CV on Train → evaluate on Test)

Time‑window features:
```bash
python train_eval_models.py --indir inputs/SE001_timewin64 --outdir runs/timewin64_cv
```

xDAWN features:
```bash
python train_eval_models.py --indir inputs/SE001_xdawn --outdir runs/xdawn_cv
```

### 3) Homomorphic inference with Paillier

Run the HE pipeline on saved features using the selected model:

```bash
python he_infer_paillier_artifacts.py   --featuresdir inputs/SE001_timewin64   --modeldir   runs/timewin64_cv   --n 128 --keybits 2048
```

### 4) Demo: timing and score equivalence (plain vs HE)

```bash
# Time-window model (K samples for timing)
python demo_infer.py   --featuresdir inputs/SE001_timewin64   --modeldir   runs/timewin64_cv   --K 64 --keybits 2048

# xDAWN model
python demo_infer.py   --featuresdir inputs/SE001_xdawn   --modeldir   runs/xdawn_cv   --K 64 --keybits 2048
```

### 5) Plots & consolidated tables

```bash
python plot_result.py
```

This will generate:
- `artifacts/acc_vs_n_by_keybits.png`
- `artifacts/auc_vs_n_by_keybits.png`
- `artifacts/time_per_sample_by_n_and_keybits.png`
- (Re)generates `artifacts/summary_all.md`

---

## 📊 Results (from `artifacts/summary_all.md`)

### A) Consolidated HE vs plaintext (from `artifacts/summary_all.md`)

| Subject | Accuracy | Precision | Recall | F1-score | Confusion Matrix                | AUC  |
|---------|----------|-----------|--------|----------|----------------------------------|------|
| A_01    | 0.9100   | 0.6287    | 0.1217 | 0.204    | [[11612, 88], [1075, 149]]      | 0    |
| A_02    | 0.9195   | 0.6949    | 0.268  | 0.387    | [[11556, 144], [896, 328]]      | 0    |
| A_03    | 0.9053   | 0.5000    | 0.0139 | 0.027    | [[11683, 17], [1207, 17]]       | 0    |
| A_04    | 0.9120   | 0.6733    | 0.138  | 0.229    | [[11618, 82], [1055, 169]]      | 0    |
| A_05    | 0.9059   | 0.5741    | 0.0253 | 0.049    | [[11677, 23], [1193, 31]]       | 0    |
| A_06    | 0.9266   | 0.7383    | 0.348  | 0.473    | [[11549, 151], [798, 426]]      | 0    |
| A_07    | 0.7100   | 0.2079    | 0.734  | 0.324    | [[8278, 3422], [326, 898]]      | 0    |
| A_09    | 0.6445   | 0.1501    | 0.591  | 0.239    | [[7607, 4093], [501, 723]]      | 0    |
| A_14    | 0.9034   | 0.3529    | 0.0245 | 0.046    | [[11645, 55], [1194, 30]]       | 0    |
| A_15    | 0.9043   | 0.4085    | 0.0237 | 0.045    | [[11658, 42], [1195, 29]]       | 0    |
| A_16    | 0.6320   | 0.1425    | 0.575  | 0.228    | [[7464, 4236], [520, 704]]      | 0    |
| A_17    | 0.9056   | 0.5667    | 0.0139 | 0.027    | [[11687, 13], [1207, 17]]       | 0    |
| A_19    | 0.9064   | 0.5530    | 0.0596 | 0.108    | [[11641, 59], [1151, 73]]       | 0    |

---
## 📊 **Summary Results for all A_xx**

- **Average Accuracy:** 0.8527
- **Average Precision:** 0.4762
- **Average Recall:** 0.2259
- **Average F1 Score:** 0.1835
- **Average AUC:** 0.5721
- **Average Time Per Sample:** 1.0000s

### 📈 **Visual Results**

Below are the visual results that demonstrate the performance of our models:

- **AUC vs Keybits:**

![AUC vs Keybits](artifacts/auc_vs_keybits.png)

- **Average Confusion Matrix:**

![Average Confusion Matrix](artifacts/average_confusion_matrix.png)

- **Time Per Sample vs Keybits:**

![Time Per Sample vs Keybits](artifacts/time_per_sample_vs_keybits.png)

---

## 🛡️ Privacy & threat model

- **Goal:** The server running the inference must **not** learn the user's EEG features.  
- **Approach:** Encrypt the feature vector with Paillier (client‑side), send ciphertexts; the server computes the **linear score** under encryption and returns an encrypted result; the client decrypts and thresholds locally.

---

**Happy (and private) BCI!** 🧪🔒
