## Intern Performance ML Project

## 📌 Overview
This project predicts intern performance using **machine learning models (Random Forest & XGBoost)**.  
It follows an **industry‑standard structure** with separate modules for data loading, preprocessing, visualization, modeling, and evaluation.  

# You can:
- Train models (`main.py`) on your dataset  
- Save the best models (`rf_best.pkl`, `xgb_best.pkl`)  
- Use a **Streamlit app (`streamlit_app.py`)** as a simple input interface to predict performance labels based on user‑entered values  

---

# 📂 Project Structure
```
intern-performance-ml/
├─ data/                        # Place your dataset here
├─ notebooks/                   # Optional Jupyter notebooks for exploration
├─ src/                         # Source code modules
│  ├─ config.py
│  ├─ data_loader.py
│  ├─ preprocessing.py
│  ├─ visualization.py
│  ├─ modeling.py
│  ├─ evaluation.py
│  └─ utils.py
├─ models/                      # Trained models saved here
│  ├─ rf_best.pkl
│  └─ xgb_best.pkl
├─ reports/                     # Figures and metrics
│  ├─ figures/
│  └─ metrics_comparison.md
├─ main.py                      # Training + evaluation script
├─ streamlit_app.py             # User input interface for predictions
├─ requirements.txt             # Dependencies
└─ README.md                    # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone or download the project
```bash
git clone <your-repo-url>
cd intern-performance-ml
```

### 2. Create a virtual environment (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\activate.bat   # Use activate.bat to avoid PowerShell policy issues
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Training the Models

1. Place your dataset in `data/intern_performance_data.csv`.  
   Required columns:
   - `attendance_rate`
   - `task_completion_rate`
   - `avg_feedback_score`
   - `final_assessment_score`
   - `performance_label` (binary: 0 = Low, 1 = High)

2. Run the training script:
```bash
python main.py
```

3. Outputs:
   - Trained models saved in `models/` (`rf_best.pkl`, `xgb_best.pkl`)  
   - Evaluation metrics in `reports/metrics_comparison.md`  
   - Visualizations in `reports/figures/`  

---

## 🎛 Using the Streamlit App

Once models are trained and saved:

```bash
streamlit run streamlit_app.py
```

Open the browser link (usually `http://localhost:8501`).  

You’ll see a simple form where you can enter:
- Attendance Rate (%)  
- Task Completion Rate (%)  
- Average Feedback Score (0–10)  
- Final Assessment Score (0–100)  

Select **Random Forest** or **XGBoost**, click **Predict Performance**, and the app will return the predicted label:
- `Low Performance`
- `High Performance`

---

### 🧰 Requirements
Key dependencies (already listed in `requirements.txt`):
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `streamlit`
- `joblib`

---

# ✅ Best Practices
- Always run `main.py` first to build models before using Streamlit.  
- Keep training and prediction separate (training = `main.py`, prediction = `streamlit_app.py`).  
- Use a virtual environment for clean dependency management.  
- Add `.venv/` and `__pycache__/` to `.gitignore`.  

---

## 🚀 Quickstart Summary
1. **Train models:** `python main.py`  
2. **Run app:** `streamlit run streamlit_app.py`  
3. **Predict:** Enter values → get performance label  

---

