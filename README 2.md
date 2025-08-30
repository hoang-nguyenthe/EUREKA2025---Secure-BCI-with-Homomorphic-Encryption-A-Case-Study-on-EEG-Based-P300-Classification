# 🧠 Privacy-Preserving Brain-Computer Interface with Homomorphic Encryption

## 📌 Overview
This project demonstrates how **Homomorphic Encryption (HE)** can be applied to **Brain-Computer Interface (BCI)** pipelines, enabling **privacy-preserving EEG signal classification**.  
We compare two feature extraction methods (**Time-window baseline** and **xDAWN**) and two evaluation settings (**plaintext inference** vs. **Paillier HE inference**).  

The results show that **LDA models** trained on BCI features achieve strong performance while still being executable under HE with only moderate computational overhead.

---

## 📂 Dataset
We use the **BigP3BCI Dataset** from [PhysioNet](https://physionet.org/content/bigp3bci/1.0.0/).  

- **Name**: BigP3BCI (Large-scale P300-based BCI speller dataset)  
- **Content**: EEG recordings from multiple subjects performing P300 speller tasks.  
- **Format**: EDF files, annotated with stimulus events.  
- **Sampling rate**: 256 Hz.  
- **Why this dataset?**  
  - Publicly available, peer-reviewed, and widely cited.  
  - Contains enough sessions and epochs to benchmark BCI + encryption methods.  
  - Standardized annotation scheme → easy preprocessing with MNE-Python.  

> ⚠️ **Note**: Raw EDF files are **not uploaded to this repo** (too large and license-restricted). Instead, we provide **processed features (`.npz`)** in the `artifacts/` folder for reproducibility.

---

## ⚙️ Pipeline
### 1. Feature Extraction
- **Time-window baseline**: simple band-pass filtering + epoch slicing → mean features per window.  
- **xDAWN**: spatial filtering to enhance P300 → sliding-window features.  

Scripts:
- `save_features64.py` → generate time-window features (`inputs/SE001_timewin64/`).  
- `make_xdawn_features.py` → generate xDAWN features (`inputs/SE001_xdawn/`).  

Outputs:
- `train_features.npz`, `test_features.npz` (X, y arrays).  
- `features_meta.json` (metadata).

---

### 2. Model Training
- Models tested: Logistic Regression (LR), SVC, KNN, Decision Tree, LDA.  
- Evaluation: **cross-validation** on training set, best model selected by ROC-AUC.  
- Tuned with grid search (`tune_models.py`).  
- Winner: **LDA** for HE compatibility.  

Outputs:
- `runs/.../model_best.joblib`, `scaler_best.joblib`, `best_config.json`.  
- Performance logs: `results_tuned.csv`.

---

### 3. Homomorphic Encryption Inference
- HE scheme: **Paillier**.  
- Scripts:  
  - `he_infer_paillier_artifacts.py` → run LDA inference under HE.  
  - `demo_infer.py` → sanity-check (plaintext vs. HE scores).  
  - `run_he_suite.sh` → run full benchmark (different N, key sizes).  

Outputs:
- `results_log_artifacts.csv` → per-run log.  
- `summary_all.csv` / `.md` → aggregated comparison.  

---

## 📊 Results
Example aggregated results (from `artifacts/summary_all.md`):

| Model | N   | Keybits | AUC_plain | ACC_plain@opt | AUC_HE | ACC_HE@opt | Time/sample (s) |
|-------|----:|--------:|----------:|--------------:|-------:|-----------:|----------------:|
| LDA   | 64  | 2048    | 0.7410    | 0.7031        | 0.7410 | 0.7031     | 7.043           |
| LDA   | 64  | 3072    | 0.7410    | 0.7031        | 0.7410 | 0.7031     | 29.632          |
| LDA   | 128 | 2048    | 0.6667    | 0.6562        | 0.6667 | 0.6562     | 7.082           |
| LDA   | 256 | 2048    | 0.6003    | 0.6016        | 0.6003 | 0.6016     | 8.778           |

Visualizations (in `artifacts/`):
- `auc_vs_n_by_keybits.png`  
- `acc_vs_n_by_keybits.png`  
- `time_per_sample_by_n_and_keybits.png`

---

## 📂 Repository Structure
```
.
├── inputs/                  # Generated features (not EDF)
│   ├── SE001_timewin64/     # Time-window features
│   └── SE001_xdawn/         # xDAWN features
├── runs/                    # Model training outputs
├── artifacts/               # Results, plots, and reports
├── save_features64.py       # Time-window feature extractor
├── make_xdawn_features.py   # xDAWN feature extractor
├── train_eval_models.py     # Train & evaluate models
├── tune_models.py           # Hyperparameter tuning
├── he_infer_paillier_artifacts.py # HE inference
├── demo_infer.py            # Demo HE vs plaintext
├── plot_result.py           # Plot final comparison
└── run_he_suite.sh          # Run full HE experiments
```

---

## 🚀 How to Run
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare features**
   ```bash
   python save_features64.py --session /path/to/EDF --outdir inputs/SE001_timewin64
   python make_xdawn_features.py --session /path/to/EDF --outdir inputs/SE001_xdawn
   ```

3. **Train models**
   ```bash
   python train_eval_models.py --indir inputs/SE001_timewin64 --outdir runs/timewin64_cv
   python train_eval_models.py --indir inputs/SE001_xdawn --outdir runs/xdawn_cv
   ```

4. **Homomorphic inference**
   ```bash
   python he_infer_paillier_artifacts.py --featuresdir inputs/SE001_xdawn --modeldir runs/xdawn_cv --n 128 --keybits 2048
   ```

5. **Run demo + plots**
   ```bash
   ./run_he_suite.sh
   python plot_result.py
   ```

---

## 📝 Citation
If you use this work, please cite the dataset:

```bibtex
@dataset{bigp3bci2020,
  author       = {Liu, Yijun and Sourina, Olga and Nguyen, Minh-Thang},
  title        = {BigP3BCI: A Large-Scale Dataset for P300-based Brain-Computer Interfaces},
  year         = {2020},
  publisher    = {PhysioNet},
  url          = {https://physionet.org/content/bigp3bci/1.0.0/}
}
```

---

## ✨ Authors
- Research & Implementation: *[Your Name]*  
- Data Source: PhysioNet BigP3BCI  
- Built with: Python, MNE, Scikit-learn, PyPaillier
