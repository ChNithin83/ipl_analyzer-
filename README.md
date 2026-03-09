# 🏏 IPL Analytics Hub — Real Kaggle Data

Built with Python, Streamlit & Plotly. Uses real IPL ball-by-ball data from Kaggle.

## 📊 Data
- `matches.csv` — 1095 matches (2008–2024)
- `deliveries.csv` — 260,920 ball-by-ball records
- 606 real players after processing

## 🚀 Setup (Run these in Terminal)

```bash
# Step 1 — Install libraries
pip3 install -r requirements.txt

# Step 2 — Process Kaggle data (run ONCE)
python3 preprocess.py

# Step 3 — Launch app
streamlit run app.py
```

## 📁 Files
```
ipl_analyzer/
├── app.py            ← Main Streamlit dashboard
├── preprocess.py     ← Converts Kaggle CSVs to ipl_data.csv
├── matches.csv       ← Kaggle dataset (you downloaded this)
├── deliveries.csv    ← Kaggle dataset (you downloaded this)
├── requirements.txt  ← Python libraries
└── ipl_data.csv      ← Created after running preprocess.py
```

## 🧠 ML Models
- Random Forest Regressor (best)
- Gradient Boosting Regressor
- Linear Regression

## 🛠️ Tech Stack
Python · Streamlit · Plotly · Scikit-learn · Pandas · NumPy

---
**Author:** Nithin Chowdary | BCA Final Year | Kakatiya University
