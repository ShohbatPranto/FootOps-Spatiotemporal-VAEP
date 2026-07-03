# 📌 Project Overview
This repository contains a world-class, production-grade analytics pipeline for calculating **Expected Threat (xT)** and the **Value of Actions by Estimating Probabilities (VAEP)** in football (soccer). 

Instead of relying on basic pre-cleaned datasets, this project builds an end-to-end MLOps ecosystem from scratch using raw tracking/event paradigms. It demonstrates deep data engineering by parsing nested JSON event streams, strictly standardizing them to the **SPADL** (Soccer Player Action Dataset Language) schema, and applying complex mathematical coordinate transformations.

---

## 🏗️ Architectural Blueprint: The 7-Phase "Football-Ops" System

- **Phase 1: Ingestion & SPADL Standardization** - Parsing nested JSON streams from StatsBomb, handling arbitrary coordinate scales, and standardizing to a metric pitch ($105m \\times 68m$).
- **Phase 2: Mathematical Feature Engineering** - Calculating geometric distances, angles to the goal, time deltas, and progression vectors.
- **Phase 3: Mathematical Formulation & Model Training** - Building dual parallel LightGBM classifiers ($P_{scores}$ and $P_{concedes}$) utilizing a strict chronological match-level train/test split.
- **Phase 4: Action Valuation & Spatial Aggregation** - Computing the $\\Delta V$ for every on-ball action and aggregating spatial threat across a 96-zone 2D pitch grid.
- **Phase 5: Scouting/Recruitment Visualizations** - Generating production-grade Pitch Heatmaps and scout-ready profiles via `mplsoccer`.
- **Phase 6: Automated Data Lake & Feature Store** - Caching large-scale SPADL matrices into compressed `.parquet` formats to break network I/O bottlenecks and scale to 100,000+ actions.
- **Phase 7: Interactive Analytics Dashboard** - A dynamic Streamlit frontend allowing scouts to interactively filter player statistics, render custom xT heatmaps, and scrub through match timelines.

---

## 🧮 Mathematical Formulations

To bridge the gap between raw $(X, Y)$ coordinates and machine learning spatial awareness, this pipeline engineers the following game-state mechanics:

### 1. Spatial Geometry
We define the center of the opponent's goal on the SPADL pitch at coordinates $(105, 34)$.

**Euclidean Distance to Goal ($d$):**
$$d = \\sqrt{(105 - x)^2 + (34 - y)^2}$$

**Angle to Goal Center ($\\theta$):**
$$\\theta = \\left| \\arctan \\left( \\frac{34 - y}{105 - x} \\right) \\right| \\times \\frac{180}{\\pi}$$

### 2. VAEP (Value of Actions by Estimating Probabilities)
For every action $a_i$ in a match sequence, we look ahead $N=10$ actions to estimate scoring/conceding shifts based on the state vector $x_i$.

$$\\Delta P_{scores}(a_i) = P_{scores}(x_i) - P_{scores}(x_{i-1})$$
$$\\Delta P_{concedes}(a_i) = P_{concedes}(x_i) - P_{concedes}(x_{i-1})$$

The total value of the action incorporates possession shift logic (flipping threat if the ball is turned over):
$$V(a_i) = \\Delta P_{scores}(a_i) - \\Delta P_{concedes}(a_i)$$

---

## 📂 Repository Structure

```text
vaep_xt_pipeline/
├── data_store/                 # Automated Local Data Lake
│   ├── raw_json/               # Raw StatsBomb Match Events
│   └── feature_store/          # Cached .parquet SPADL matrices
│
├── outputs/                    # Exported Visualizations
│   └── spatial_xt_heatmap.png  # Generated 2D Threat Grids
│
├── src/
│   ├── config.py               # Constants, Magic Numbers, Pitch Dimensions
│   ├── data/
│   │   ├── ingestion.py        # StatsBomb API Handshake Layer
│   │   ├── spadl.py            # Coordinate Normalizer (120x80 -> 105x68)
│   │   └── feature_store.py    # Parquet Caching & Storage Orchestrator
│   │
│   ├── features/
│   │   ├── engineering.py      # Spatial & Temporal Feature Math
│   │   └── aggregation.py      # 2D Grid Threat Aggregation
│   │
│   ├── models/
│   │   ├── train.py            # Dual LightGBM Training & Chronological Splitting
│   │   └── valuation.py        # VAEP ΔV Mathematics & Possession Logic
│   │
│   └── visualization/
│       └── plots.py            # mplsoccer Dark-mode rendering
│
├── requirements.txt            # Pinned reproducible dependencies
├── run_pipeline_test.py        # Integration test for single-match parsing
├── run_scaleup.py              # Phase 6 Orchestrator (Model Training on Cache)
└── app.py                      # Phase 7 Streamlit MLOps Dashboard
```


#🚀 Installation & Quick Start
1. Environment Setup (Isolated Workspace)
Using standard venv to prevent dependency conflicts (e.g. bypassing Windows Store Python hangs).

# Initialize barebones environment
python -m venv .venv --without-pip

# Activate environment (Windows PowerShell)
& ".\\.venv\\Scripts\\Activate.ps1"

# Manually inject pip for clean build
python -m ensurepip --upgrade

# Install dependencies (pandas, pyarrow, lightgbm, statsbombpy, mplsoccer, streamlit)
pip install -r requirements.txt


2. Execution Flow
Run the automated pipeline to flood the Data Lake, compute the Parquet caches, and train the dual-estimators on 120,000+ actions.

python run_scaleup.py
streamlit run app.py


Top Spatial Feature Importances (Tree Splits)
The model relies heavily on geometric proximity, proving mathematically that the algorithm accurately "sees" the pitch:

end_dist_to_goal

start_dist_to_goal

end_angle_to_goal

start_angle_to_goal

progression_dist



