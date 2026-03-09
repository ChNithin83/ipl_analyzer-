# 🏏 IPL Analytics Hub

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)

An interactive IPL Cricket Analytics Dashboard built with Python, Streamlit, and Plotly. Features real ball-by-ball Kaggle data from 2008-2024, complete player and team comparisons, and a Machine Learning Auction Predictor.

## 🌟 Features
- **🏠 Overview:** Broad insights, top run-scorers, highest wicket-takers, and role distributions. Includes Interactive Treemaps and Season Trends.
- **👤 Player Analysis:** Deep dive into individual players. Explore their season-wise batting, bowling, and auction value progression with detailed charts.
- **📊 Teams:** Head-to-head metrics and statistical comparisons across franchises.
- **🏆 Leaderboard:** Sort through players by custom thresholds like "Best Strike Rate" or "Highest Auction Value".
- **🤖 Auction Predictor:** Input custom stats and let Machine Learning predict the player's value. Evaluates predictions against Random Forest, Gradient Boosting, and Linear Regression models.

## 📊 Data
- `matches.csv` — 1095 matches (2008–2024)
- `deliveries.csv` — 260,920 ball-by-ball records
- Results in 606 real modeled players after processing.

## 🚀 Setup & Installation (Local)

**Step 1 — Clone the Repo & Install requirements**
```bash
git clone <your-repo-link>
cd ipl_analyzer
pip install -r requirements.txt
```

**Step 2 — Process Kaggle data**
You need the original `matches.csv` and `deliveries.csv` in the root folder. Then, run the preprocessing script once to construct the main `ipl_data.csv`:
```bash
python preprocess.py
```

**Step 3 — Launch the App**
```bash
streamlit run app_4.py
```

## 🧠 ML Models Built-In
- **Random Forest Regressor** (Best accuracy utilizing 150 estimators)
- **Gradient Boosting Regressor**
- **Linear Regression**

---
**Author:** Nithin Chowdary | BCA Final Year | Kakatiya University
