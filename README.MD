# Project Title

**Cap-Adjusted Player Mispricing and Team Performance in the NHL**

---

# Project Summary 

This project clusters NHL players into structural archetypes using Gaussian Mixture Models, then builds a cap-adjusted salary valuation model to estimate player-level mispricing (“alpha”). Player salaries are modeled as a percentage of the league salary cap to account for inflation across seasons. The project then aggregates player alpha to the team level and tests whether teams with higher aggregate alpha achieve stronger regular season performance or deeper playoff runs.

---

# Core Research Question

Do NHL teams that roster more underpriced players (relative to cap-adjusted expected salary) achieve greater competitive success?

---

# Project Structure Overview

1. Data Collection & Cleaning
2. Player Archetype Clustering (GMM)
3. Salary Share Valuation Model
4. Player Alpha Calculation
5. Team-Level Alpha Aggregation
6. Team Performance Analysis

---

# Detailed Step-by-Step Plan

---

## PHASE 1 — Data Collection & Preparation

### 1.1 Collect Player Data (Multi-Season)

For each player-season:

* Salary (AAV)
* Salary cap for that season
* Position
* Age
* TOI / usage
* Goals per game
* Assists per game
* xG per game
* xGA / on-ice usage features
* Special teams TOI
* Physical stats (hits, blocks...)
* Penalty kill and more ...

---

### 1.2 Normalize Salary

Create:

[
SalaryShare = \frac{Salary}{SalaryCap}
]

This ensures salaries are comparable across seasons.

---

### 1.3 Clean Dataset

* Remove very low TOI players with a light usage screen
* Build per-game performance features plus usage variables
* Merge salary + performance
* Ensure one row per player per season

**Output:**
`player_master.csv`

---

# PHASE 2 — Player Archetype Clustering (GMM)

### 2.1 Feature Selection

Cluster using structural per-game features plus usage variables:

* Goals per game
* Assists per game
* xG per game
* On-ice xG / xGA usage context
* TOI and deployment
* Age
* Physicality metrics

Standardize features before clustering.

---

### 2.2 Fit Gaussian Mixture Model

* Use GMM with 6–8 components
* Select number of clusters using BIC/AIC
* Extract cluster probabilities

Each player receives:

* Soft role probabilities
* Dominant role label
(maybe vector?)

---

### 2.3 Interpret & Label Clusters

Manually label clusters:

Examples:

* Elite Scoring Winger
* Playmaking Center
* Defensive Defenseman
* Depth Grinder
* Prospect
* Aging Veteran

**Output:**
`player_roles.csv`

---

# PHASE 3 — Salary Share Valuation Model

### 3.1 Define Target

Dependent variable:

[
SalaryShare
]

---

### 3.2 Model Specification

Model:

[
SalaryShare = f(PerformanceFeatures, RoleType, Age, Position, Usage)
]

Include:

* Performance metrics
* Role probabilities or role dummies
* Age
* Position

Model types:

Option A: Ridge Regression
Option B: Gradient Boosting (preferred if nonlinearities matter)

Use cross-validation.

---

### 3.3 Compute Player Alpha

[
Alpha = ActualSalaryShare - PredictedSalaryShare
]

Interpretation:

* Negative alpha → underpriced
* Positive alpha → overpriced

**Output:**
`player_alpha_table.csv`

---

# PHASE 4 — Team-Level Alpha Aggregation

For each team-season:

### 4.1 Compute Team Alpha

Weighted by TOI:

[
TeamAlpha = \sum Alpha_i \cdot TOIWeight_i
]

Alternative metrics:

* Average alpha
* % of roster with negative alpha
* Alpha concentration (top 5 players)

---

### 4.2 Build Team Summary Table

For each team-season:

* TeamAlpha
* Goal differential
* Points %
* Playoff rounds won
* Average age
* Cap utilization

**Output:**
`team_alpha_summary.csv`

---

# PHASE 5 — Team Performance Analysis

### 5.1 Regular Season Analysis

Regression:

[
GoalDifferential = f(TeamAlpha, AvgAge, CapUtilization)
]

Test:

* Is TeamAlpha significant?
* Does it predict performance?

---

### 5.2 Playoff Outcome Analysis

Model:

[
PlayoffRoundsWon = f(TeamAlpha, RegularSeasonPerformance)
]

Alternative:
Binary outcome (deep run vs not).

---

# 📊 Deliverables

You should produce:

* Cluster visualizations (PCA plots)
* Salary model diagnostics (R², feature importance)
* Distribution of player alpha
* Heatmap of alpha by role
* Regression results linking alpha to performance
* Clear written conclusions

---

# 🧠 What This Project Demonstrates

* Unsupervised learning (GMM)
* Cross-sectional regression modeling
* Cap-adjusted valuation
* Portfolio aggregation logic
* Empirical hypothesis testing
* Economic interpretation

This is structured quantitative modeling applied to hockey.

---

# Suggested File Structure
nhl_salary_alpha_project/
│
├── data/
│   ├── raw/                           # THE SOURCE (Do not edit these manually)
│   │   ├── hr_goalie_stats.csv        # Your historical 2008-2025 goalie file
│   │   ├── hr_player_stats.csv        # Your historical 18k row skater file
│   │   ├── mp_goalies.csv             # MoneyPuck goalie data
│   │   ├── mp_skaters.csv             # MoneyPuck skater data
│   │   ├── mp_teams.csv               # MoneyPuck team-level data
│   │   ├── player_caps_18-19.xlsx     # PuckPedia/Salary data by year
│   │   ├── player_caps_19-20.xlsx
│   │   ├── player_caps_20-21.xlsx
│   │   ├── player_caps_21-22.xlsx
│   │   ├── player_caps_22-23.xlsx
│   │   ├── player_caps_23-24.xlsx
│   │   └── player_caps_24-25.xlsx
│   │
│   └── processed/                     # THE REFINERY (Generated by your code)
│       ├── cleaned_data.csv           # Merged and Cleaned HR + MoneyPuck + Salaries
│       ├── modeling_ready_data.csv    # Merged HR + MoneyPuck + Salaries
│       ├── archetypes_assigned.csv    # Master file + GMM Cluster labels
│       ├── team_alpha_vs_wins.csv     # Combined Team Alpha stats + Win totals
│       └── alpha_segmentation.csv     # Alpha broken down by Scorer vs Grinder, G vs F
│
├── notebooks/                         # THE LABORATORY (Exploration & Visuals)
│   ├── 01_data_cleaning.ipynb         # Merging HR, MP, and the 7 XLSX salary files
│   ├── 02_gmm_clustering.ipynb        # Defining Scorer vs Grinder archetypes
│   ├── 03_salary_alpha_model.ipynb    # The actual pricing/Alpha math
│   └── 04_visual_analysis.ipynb       # Generating those final heatmaps and plots
│
├── src/                               # THE FACTORY (Reusable Production Code)
│   ├── data_cleaning.py               # Main script to clean and merge all raw files
│   ├── clustering.py                  # GMM modeling logic for archetypes
│   ├── salary_model.py                # "Alpha" calculation engine
│   ├── team_analysis.py               # Aggregates player Alpha into team totals
│   ├── utils.py                       # Team mappers, currency formatters, etc
│   ├── renaming_map.py                # Purely renames all features
│   └── features.py                    # In charge of feature engineering
│
├── results/                           # THE GALLERY (Final outputs)
│   ├── cluster_plots/                 # Visuals of player groupings
│   ├── model_diagnostics/             # Regression accuracy checks
│   ├── alpha_distributions.csv        # Raw stats on Alpha spread
│   └── alpha_heatmaps/                # Visuals (Team Alpha vs Wins, Archetype Value)
│
└── README.md                          # Project documentation

