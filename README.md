# NutriBiteBot: A Clinical Nutrition Decision Support System

NutriBiteBot is an end-to-end clinical nutrition platform that generates safe, personalized recipes for patients with chronic conditions (CKD, Hypertension, Type 2 Diabetes). It combines a fine-tuned TabNet deep-learning model for risk stratification, a hierarchical clinical rules engine for conflict resolution, computer vision for fridge scanning, and an LLM for recipe generation — all running locally with a Dockerized Supabase database for data privacy.
=======
---
title: Nutri Bite Bot
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# NutriBiteBot: A Clinical Nutrition Decision Support System

NutriBiteBot is an end-to-end clinical nutrition platform that generates safe, personalised recipes for patients with chronic conditions (CKD, Hypertension, Type 2 Diabetes). It combines a multi-target TabNet deep-learning model for risk stratification, a sigmoid-based portion computation engine, a hierarchical clinical rules engine for conflict resolution, computer vision for fridge scanning, and a bounded LLM for recipe generation.

**Live deployment:** Flask backend on [Hugging Face Spaces](https://gayathri-27-nutri-bite-bot.hf.space) · Frontend on Vercel
>>>>>>> 196878a9a3afb1baa3aa57e1a4a883260ad270e7

---

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
<<<<<<< HEAD
3. [Prerequisites](#prerequisites)
4. [Setup & Installation](#setup--installation)
5. [Running the Application](#running-the-application)
6. [Using the App](#using-the-app)
7. [API Reference](#api-reference)
8. [Project Structure](#project-structure)
9. [Stopping the Services](#stopping-the-services)
10. [Troubleshooting](#troubleshooting)
=======
3. [TabNet Model Architecture](#tabnet-model-architecture)
4. [Deployment](#deployment)
5. [Running Locally](#running-locally)
6. [Using the App](#using-the-app)
7. [API Reference](#api-reference)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)
10. [Clinical Guidelines Referenced](#clinical-guidelines-referenced)
>>>>>>> 196878a9a3afb1baa3aa57e1a4a883260ad270e7

---

## Features

<<<<<<< HEAD
- **Clinical Risk Stratification (TabNet ML)** — Predicts sodium sensitivity, potassium sensitivity, protein restriction need, and carbohydrate sensitivity from 14 lab/demographic features. No hardcoded thresholds; all risk levels come from model predictions.
- **Hierarchical Clinical Rules Engine** — Resolves conflicting dietary guidelines automatically (e.g., CKD potassium cap overrides HTN DASH recommendation). Priority order: Renal > Cardiac > Metabolic.
- **Fridge Scanner / Pantry Inventory** — Upload a photo of your fridge; Roboflow computer vision detects ingredients and maps them to the IFCT nutritional database.
- **Safe Portion Calculation** — Determines the maximum safe grams per ingredient per meal based on the patient's ML-derived risk profile and daily nutrient budget (sodium, potassium, protein, carbs, phosphorus).
- **Recipe Generation (Groq / Llama 3.3)** — Generates recipes that strictly respect per-ingredient portion limits, using SHARE methodology (Substitute, Halve, Add, Remove, Emphasize).
- **Local Secure Storage** — A fully local Dockerized Supabase (PostgreSQL) instance stores patient data and generated recipes, ensuring no data leaves your machine.
=======
- **Clinical Risk Stratification (TabNet)** — Predicts sodium sensitivity, potassium sensitivity, protein restriction need, and carbohydrate sensitivity from 14 EHR-derived features. Mean accuracy 99.91%, mean Cohen's κ 0.9983.
- **Sigmoid Portion Engine** — Converts continuous severity scores into ingredient-specific maximum safe gram quantities against a session-persistent daily nutrient budget, grounded in IFCT 2017.
- **Hierarchical Clinical Rules Engine** — Automatically resolves conflicting dietary guidelines across co-existing conditions. Priority: Renal (KDIGO) > Cardiac (AHA/ACC) > Metabolic (ADA).
- **Fridge Scanner** — Upload a fridge photo; Roboflow CV detects ingredients and maps them to the IFCT nutritional database.
- **Bounded Recipe Generation (Groq / Llama 3.3)** — Generates recipes strictly within pre-computed per-ingredient gram limits. The LLM cannot override clinical constraints.
- **Optional Local Storage** — Supabase (PostgreSQL) instance for persisting patient data and recipes. Gracefully disabled if not configured.
>>>>>>> 196878a9a3afb1baa3aa57e1a4a883260ad270e7

---

## System Architecture

```
<<<<<<< HEAD
[Browser / Frontend UI]
         |  HTTP
         v
[Flask API — app.py :5000]
    |           |           |
    v           v           v
[TabNet     [Portion    [Groq LLM]
 Model1]     Engine]    (Recipe Gen)
=======
[Vercel Frontend (SPA)]
         |  HTTPS
         v
[HF Spaces — Flask API :7860]
    |           |           |
    v           v           v
[TabNet     [Portion    [Groq LLM]
 Classifier] Engine]    (Recipe Gen)
>>>>>>> 196878a9a3afb1baa3aa57e1a4a883260ad270e7
    |           |
    v           v
[IFCT Nutrient DB]   [Roboflow CV API]
         |
         v
<<<<<<< HEAD
[Local Supabase (Docker)] ← stores patients + recipes
```

**Stack**
| Layer | Technology |
|---|---|
| Frontend | HTML / CSS / Vanilla JS (SPA) |
| Backend | Python 3.9+, Flask |
| ML Models | TabNet (`pytorch_tabnet`), PyTorch |
| Nutritional DB | IFCT CSV database |
| Computer Vision | Roboflow API |
| LLM | Groq API (Llama-3.3-70b-versatile) |
| Database | Local Supabase (PostgreSQL via Docker) |

---

## Prerequisites

Install the following before proceeding:

| Requirement | Version | Purpose |
|---|---|---|
| **Python** | 3.9 or later | Backend and ML models |
| **Docker Desktop** | Latest | Runs the local Supabase database stack |
| **Node.js** | 18 or later | Required to run the Supabase CLI via `npx` |
| **pip** | Bundled with Python | Python package manager |

You also need API keys for two external services:

| Service | Where to get it | Required for |
|---|---|---|
| **Groq API** | https://console.groq.com | Recipe generation |
| **Roboflow API** | https://app.roboflow.com | Fridge ingredient detection |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd nutri-bite-bot
```

### 2. Create your environment file

Create a `.env` file in the project root. Start with your external API keys — the Supabase credentials will be filled in after the next step.

```env
# External APIs
GROQ_API_KEY="your_groq_api_key_here"
ROBOFLOW_API_KEY="your_roboflow_api_key_here"

# Local Supabase (fill in after running: npx supabase start)
SUPABASE_URL="http://127.0.0.1:54331"
SUPABASE_ANON_KEY="your_local_anon_key"
SUPABASE_SERVICE_ROLE_KEY="your_local_service_role_key"
```

### 3. Start the local Supabase database

Make sure **Docker Desktop is open and running**, then execute:

```bash
npx --yes supabase start
```

This will pull and start all required containers (Postgres, PostgREST, Studio, etc.). The first run takes a few minutes to download images.

When it finishes, the CLI prints output similar to:

```
API URL:         http://127.0.0.1:54331
Studio URL:      http://127.0.0.1:54333
anon key:        eyJhbGci...
service_role key: eyJhbGci...
```

Copy the `anon key` and `service_role key` into your `.env` file under `SUPABASE_ANON_KEY` and `SUPABASE_SERVICE_ROLE_KEY`.

> The Supabase Studio dashboard (table viewer, SQL editor) is available at http://127.0.0.1:54333

### 4. Install Python dependencies
=======
[Supabase] ← optional, stores patients + recipes
```

**Stack**

| Layer | Technology | Hosting |
|---|---|---|
| Frontend | HTML / CSS / Vanilla JS (SPA) | Vercel |
| Backend | Python 3.9, Flask, Gunicorn | Hugging Face Spaces (Docker) |
| ML Model | TabNet (`pytorch_tabnet`), PyTorch | Bundled in Docker image |
| Nutritional DB | IFCT 2017 CSV (102 ingredients) | Bundled in Docker image |
| Computer Vision | Roboflow API | External API |
| LLM | Groq API (Llama-3.3-70b-versatile) | External API |
| Database | Supabase (PostgreSQL) | Optional / local |

---

## TabNet Model Architecture

### Overview

Four independent `TabNetClassifier` models are trained, one per clinical sensitivity target. Each model takes the same 14-feature input vector and outputs a 3-class probability distribution (Low / Moderate / High).

### Input Features (14)

| Feature | Description | Clinical Basis |
|---|---|---|
| `age` | Patient age in years | Demographics |
| `sex_male` | Binary sex flag | Demographics |
| `bmi` | Body mass index (kg/m²) | Metabolic |
| `sbp` / `dbp` | Systolic / diastolic BP (mmHg) | AHA/ACC |
| `has_htn` | Hypertension diagnosis flag | AHA/ACC |
| `has_dm` | Type 2 Diabetes flag | ADA 2024 |
| `has_ckd` | Chronic Kidney Disease flag | KDIGO 2024 |
| `serum_sodium` | Serum Na (mEq/L) | AHA/ACC |
| `serum_potassium` | Serum K (mmol/L) | KDIGO 2024 |
| `creatinine` | Serum creatinine (mg/dL) | KDIGO 2024 |
| `egfr` | Estimated GFR (mL/min/1.73m²) | KDIGO 2024 |
| `hba1c` | Glycated haemoglobin (%) | ADA 2024 |
| `fbs` | Fasting blood sugar (mg/dL) | ADA 2024 |

### Output Targets (4)

| Target | Dominant Features | Guideline |
|---|---|---|
| `sodium_sensitivity` | serum Na, SBP, has\_htn | AHA/ACC |
| `potassium_sensitivity` | eGFR, serum K | KDIGO 2024 |
| `protein_restriction` | eGFR, creatinine, has\_ckd | KDIGO 2024 |
| `carb_sensitivity` | HbA1c, FBS, has\_dm | ADA 2024 |

Each model outputs class probabilities `[P(Low), P(Moderate), P(High)]`. A continuous severity score is derived as:

```
severity = 0·P(Low) + 1·P(Moderate) + 2·P(High)   →   s ∈ [0, 2]
```

This score feeds directly into the sigmoid portion engine, preserving sub-threshold severity gradients.

### Preprocessing Pipeline

1. **Median imputation** — missing lab values replaced with training-set medians
2. **Z-score scaling** — all features standardised (mean 0, std 1)
3. **Per-target monotonic transformer** — isotonic regression enforces physiologically correct feature directionality (e.g. declining eGFR → higher restriction, rising HbA1c → higher carb restriction). Each target gets an independently fitted transformer to prevent cross-target contamination.

### Attention Mechanism

TabNet uses a sparsemax attention mask at each sequential decision step, selecting a sparse subset of features per step. Per-step masks are averaged to produce a per-patient feature attribution — native interpretability without SHAP or other post-hoc methods.

### Training Configuration

| Parameter | Value |
|---|---|
| Architecture | `TabNetClassifier` (pytorch_tabnet) |
| Training device | GPU |
| Inference device | CPU (Docker) |
| Early stopping patience | 15 epochs |
| Max epochs | 100 |
| Training data | MIMIC-IV EHR (80/20 stratified split) |
| Classes per target | 3 (Low / Moderate / High) |

### Performance (Held-Out Test Split)

| Target | Accuracy | F1 (weighted) | Cohen's κ |
|---|---|---|---|
| sodium\_sensitivity | 0.9996 | 0.9996 | 0.9992 |
| potassium\_sensitivity | 0.9995 | 0.9995 | 0.9992 |
| protein\_restriction | 0.9993 | 0.9993 | 0.9984 |
| carb\_sensitivity | 0.9979 | 0.9979 | 0.9963 |
| **Mean** | **0.9991** | **0.9991** | **0.9983** |

All misclassifications are confined to adjacent ordinal classes — no non-adjacent errors observed across any target.

### Model Artifacts

```
artifacts/models/
├── model_params.json     # TabNet architecture config (N_steps, N_a, N_d, etc.)
├── network.pt            # Trained weights (Git LFS, ~218 KB)
├── imputer.joblib        # Fitted SimpleImputer
├── scaler.joblib         # Fitted StandardScaler
└── feature_names.joblib  # Ordered list of 14 feature names
```

---

## Deployment

### Current Setup

| Component | Platform | URL |
|---|---|---|
| Backend (Flask + TabNet) | Hugging Face Spaces (Docker) | `https://gayathri-27-nutri-bite-bot.hf.space` |
| Frontend (SPA) | Vercel | Your Vercel project URL |

### HF Spaces — Backend

The backend runs as a Docker container on HF Spaces. The `Dockerfile` at the project root handles the full build:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ libpq-dev
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
```

**Required Secrets** (set in Space Settings → Variables and Secrets):

| Secret | Purpose |
|---|---|
| `GROQ_API_KEY` | Recipe generation via Llama 3.3 |
| `ROBOFLOW_API_KEY` | Fridge ingredient detection |
| `SUPABASE_URL` | Optional — cloud Supabase instance |
| `SUPABASE_SERVICE_ROLE_KEY` | Optional — cloud Supabase auth |

### Vercel — Frontend

The `frontend/` directory is a static SPA deployed to Vercel. `frontend/vercel.json` configures it as a static site with no build step.

The API base URL in `frontend/script.js` is set to the HF Space:

```javascript
const API_BASE = "https://gayathri-27-nutri-bite-bot.hf.space";
```

To deploy the frontend:

```bash
npm i -g vercel
cd frontend
vercel --prod
```

Or connect the repo to Vercel via the dashboard, setting the **Root Directory** to `frontend`.

---

## Running Locally

Supabase is optional — the app is fully functional without it (patient/recipe storage is simply skipped).

### 1. Clone and install
>>>>>>> 196878a9a3afb1baa3aa57e1a4a883260ad270e7

```bash
git clone <repo-url>
cd nutri-bite-bot
pip install -r requirements.txt
```

<<<<<<< HEAD
This installs Flask, PyTorch, pytorch_tabnet, scikit-learn, Groq, Pillow, pandas, and all other required packages.

> If you see torch or tabnet compilation errors, ensure your Python version is 3.9–3.11 and that you have a compatible pip version (`pip install --upgrade pip`).

---

## Running the Application

Start the Flask backend (which also serves the frontend):
=======
### 2. Create `.env`

```env
GROQ_API_KEY="your_groq_api_key"
ROBOFLOW_API_KEY="your_roboflow_api_key"

# Optional — leave blank to skip storage
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

### 3. Run the backend
>>>>>>> 196878a9a3afb1baa3aa57e1a4a883260ad270e7

```bash
python app.py
```

<<<<<<< HEAD
You should see output like:

```
Loading ML model weights (TabNet) …
  ✓ TabNet model loaded from model_params.json + network.pt
  ✓ sodium_sensitivity → TabNet model
  ✓ potassium_sensitivity → TabNet model
  ✓ protein_restriction → TabNet model
  ✓ carb_sensitivity → TabNet model
  ✓ imputer, scaler, feature_names (14 features) loaded
Loading IFCT nutritional database …
  ✓ IFCT database: N ingredients loaded

Starting NutriBiteBot server on http://localhost:5000
```

Open your browser and navigate to **http://localhost:5000**
=======
Expected startup output:

```
Loading ML model weights (TabNet) ...
  ✓ TabNet model loaded from model_params.json + network.pt
  ✓ sodium_sensitivity     → TabNet
  ✓ potassium_sensitivity  → TabNet
  ✓ protein_restriction    → TabNet
  ✓ carb_sensitivity       → TabNet
  ✓ imputer, scaler, feature_names (14 features) loaded
Loading IFCT nutritional database ...
  ✓ IFCT database: 102 ingredients loaded
>>>>>>> 196878a9a3afb1baa3aa57e1a4a883260ad270e7

Starting NutriBiteBot server on http://localhost:5000
```

Open **http://localhost:5000** — the backend serves the frontend SPA directly.

### 4. (Optional) Local Supabase database

If you want local persistent storage, start Supabase with Docker Desktop running:

```bash
npx --yes supabase start
```

Copy the printed `API URL` and `service_role key` into `.env`:

```env
SUPABASE_URL="http://127.0.0.1:54331"
SUPABASE_SERVICE_ROLE_KEY="<printed service_role key>"
```

Restart `app.py`. Stop the containers when done:

<<<<<<< HEAD
## Using the App

The frontend is a single-page application with three main phases:

### Phase 1 — Clinical Risk Assessment

Enter the patient's demographic and lab values:

| Field | Description |
|---|---|
| Age | Patient age in years |
| Sex | Male / Female |
| Has HTN | Hypertension diagnosis |
| Has DM | Type 2 Diabetes diagnosis |
| Has CKD | Chronic Kidney Disease diagnosis |
| Serum Sodium | mEq/L |
| Serum Potassium | mEq/L |
| Creatinine | mg/dL |
| eGFR | mL/min/1.73m² |
| HbA1c | % |
| FBS | mg/dL (Fasting Blood Sugar) |
| SBP / DBP | mmHg (Systolic / Diastolic) |
| BMI | kg/m² |

Click **Run Risk Assessment** to get TabNet model predictions for sodium sensitivity, potassium sensitivity, protein restriction, and carbohydrate sensitivity (each classified as Low / Moderate / High with confidence scores).

### Phase 2 — Fridge Scan / Ingredient Selection

Either:
- **Upload a fridge photo** — Roboflow CV detects ingredients and maps them to the IFCT database automatically.
- **Select ingredients manually** — Use the searchable ingredient list pulled from the IFCT database.

### Phase 3 — Portion Recommendations & Recipe Generation

Click **Get Portions** to compute the maximum safe grams for each ingredient based on the risk assessment. Ingredients are labelled:
- **Allowed** — Full portion safe
- **Half Portion** — Use sparingly (≤75g)
- **Avoid** — Clinically unsafe (≤5g)

Then select optional preferences (cuisine, equipment, time limit) and click **Generate Recipe**. The Groq LLM generates a recipe that strictly respects every per-ingredient limit. The result is saved to your local Supabase database.

---

## API Reference

The Flask backend exposes the following REST endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend SPA |
| `POST` | `/api/predict` | Run TabNet risk assessment on patient data |
| `GET` | `/api/model-info` | Model metadata and accuracy metrics |
| `GET` | `/api/thresholds` | Clinical nutrient reference thresholds |
| `GET` | `/api/ingredients?q=<query>` | Search or list IFCT ingredients |
| `POST` | `/api/recommend` | Full Model1→Model2 pipeline: risk + portions |
| `POST` | `/api/detect` | Fridge image → Roboflow CV → IFCT mapping |
| `POST` | `/api/generate-recipe` | Generate a safe recipe via Groq LLM |

**`POST /api/predict` — required fields:**
```json
{
  "age": 68, "sex_male": 1, "has_htn": 1, "has_dm": 1, "has_ckd": 1,
  "serum_sodium": 138, "serum_potassium": 4.2, "creatinine": 1.8,
  "egfr": 52, "hba1c": 7.8, "fbs": 145, "sbp": 148, "dbp": 88, "bmi": 27.4
}
```

**`POST /api/recommend` — required fields:**
```json
{
  "patient": { <same fields as /api/predict> },
  "ingredients": ["Banana, ripe", "Rice, milled (white)", "Carrot"]
}
```

---

## Project Structure

```
nutri-bite-bot/
├── app.py                        # Flask backend + ML inference
├── supabase_client.py            # Supabase REST API bindings
├── requirements.txt              # Python dependencies
├── .env                          # API keys (not committed)
│
├── frontend/
│   ├── index.html                # Single-page application
│   ├── script.js                 # Frontend logic
│   └── style.css
│
├── artifacts/
│   └── models/
│       ├── model_params.json     # TabNet architecture config
│       ├── network.pt            # Trained TabNet weights
│       ├── imputer.joblib        # Fitted median imputer
│       ├── scaler.joblib         # Fitted StandardScaler
│       ├── feature_names.joblib  # Ordered feature list
│       └── reports/
│           ├── accuracy_summary.txt
│           └── nutrient_thresholds_reference.json
│
├── clinical models/
│   ├── ifct_database.csv         # Indian Food Composition Table
│   ├── clinicalbb_model#1.ipynb  # Model 1 training notebook
│   ├── clinicalbb_model#2.ipynb  # Model 2 training notebook
│   ├── train_model1.py
│   ├── train_model2.py
│   └── ...
│
├── supabase/
│   ├── config.toml               # Supabase project config
│   └── migrations/               # Database schema migrations
│
├── clinical_rules_engine.py      # Hierarchical conflict resolver
├── pantry_inventory.py           # Pantry management module
├── recipe_generator.py           # SHARE recipe adaptation
├── main_integration.py           # Standalone CLI pipeline
└── mimic_cohort_extraction.py    # MIMIC-IV patient extraction
```

---

## Stopping the Services

When finished, stop the app with `Ctrl+C` in the terminal, then stop the Supabase containers to free up Docker resources:

=======
>>>>>>> 196878a9a3afb1baa3aa57e1a4a883260ad270e7
```bash
npx --yes supabase stop
```

---

<<<<<<< HEAD
## Troubleshooting

**`ModuleNotFoundError: No module named 'pytorch_tabnet'`**
Run `pip install pytorch_tabnet torch` separately, then retry `pip install -r requirements.txt`.

**TabNet model fails to load on startup**
Verify that `artifacts/models/model_params.json` and `artifacts/models/network.pt` both exist. These are the trained model files and must not be moved.

**Supabase containers fail to start**
Ensure Docker Desktop is running and that ports 54331–54333 are not occupied by another process. Run `docker ps` to check for conflicts.

**Groq API error during recipe generation**
Double-check `GROQ_API_KEY` in your `.env` file. Get a key at https://console.groq.com.

**Roboflow returns no detections**
Ensure `ROBOFLOW_API_KEY` is set correctly and that the uploaded image is a clear, well-lit photo. The model used is `ingredient-detection-5uzov/5`.

**Supabase credentials missing warning (non-fatal)**
If you see `Supabase credentials not found`, the app still works — patient/recipe storage is simply skipped. Add the `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` from `npx supabase start` output to fix this.
=======
## Using the App

### Phase 1 — Clinical Risk Assessment

Enter the patient's lab values and click **Run Risk Assessment**. TabNet predicts sodium sensitivity, potassium sensitivity, protein restriction, and carbohydrate sensitivity (Low / Moderate / High) with confidence scores and a continuous severity score per nutrient.

| Field | Unit |
|---|---|
| Age | years |
| Sex | Male / Female |
| Has HTN / DM / CKD | checkbox |
| Serum Sodium | mEq/L |
| Serum Potassium | mEq/L |
| Creatinine | mg/dL |
| eGFR | mL/min/1.73m² |
| HbA1c | % |
| FBS | mg/dL |
| SBP / DBP | mmHg |
| BMI | kg/m² |

### Phase 2 — Ingredient Selection

- **Upload a fridge photo** — Roboflow CV detects ingredients and maps them to IFCT automatically.
- **Select manually** — search the 102-ingredient IFCT database.

### Phase 3 — Portions & Recipe

Click **Get Portions** to compute per-ingredient maximum safe grams. Each ingredient is labelled:

| Label | Condition |
|---|---|
| Allowed | g* > 75 g |
| Half Portion | 5 g < g* ≤ 75 g |
| Avoid | g* ≤ 5 g |

Click **Generate Recipe** — Groq Llama 3.3 generates a recipe that strictly respects every gram limit. Results are saved to Supabase if configured.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend SPA |
| `POST` | `/api/predict` | TabNet risk assessment |
| `GET` | `/api/model-info` | Model metadata and accuracy |
| `GET` | `/api/thresholds` | Clinical nutrient reference thresholds |
| `GET` | `/api/ingredients?q=<query>` | Search IFCT ingredient list |
| `POST` | `/api/recommend` | Full pipeline: risk + portions |
| `POST` | `/api/detect` | Fridge image → Roboflow → IFCT |
| `POST` | `/api/generate-recipe` | Bounded recipe via Groq LLM |

**`POST /api/predict` payload:**
```json
{
  "age": 68, "sex_male": 1, "has_htn": 1, "has_dm": 1, "has_ckd": 1,
  "serum_sodium": 138, "serum_potassium": 4.2, "creatinine": 1.8,
  "egfr": 52, "hba1c": 7.8, "fbs": 145, "sbp": 148, "dbp": 88, "bmi": 27.4
}
```

**`POST /api/recommend` payload:**
```json
{
  "patient": { "age": 68, "sex_male": 1, ... },
  "ingredients": ["Banana, ripe", "Rice, milled (white)", "Carrot"]
}
```

---

## Project Structure

```
nutri-bite-bot/
├── app.py                        # Flask backend — inference, routing, startup
├── supabase_client.py            # Supabase REST bindings (optional)
├── clinical_rules_engine.py      # Hierarchical multi-condition conflict resolver
├── pantry_inventory.py           # Pantry + IFCT lookup module
├── recipe_generator.py           # SHARE recipe adaptation logic
├── requirements.txt
├── Dockerfile                    # HF Spaces Docker build
├── .env                          # API keys — never committed
│
├── frontend/
│   ├── index.html                # Single-page application
│   ├── script.js                 # API calls, rendering, wake-up logic
│   ├── style.css
│   └── vercel.json               # Vercel static site config
│
├── artifacts/models/
│   ├── model_params.json         # TabNet architecture config
│   ├── network.pt                # Trained weights (Git LFS)
│   ├── imputer.joblib            # Fitted median imputer
│   ├── scaler.joblib             # Fitted StandardScaler
│   ├── feature_names.joblib      # Ordered 14-feature list
│   └── reports/
│       ├── accuracy_summary.txt
│       ├── confusion_matrix_*.png
│       └── nutrient_thresholds_reference.json
│
├── clinical-models/
│   ├── ifct_database.csv         # IFCT 2017 — 102 Indian food ingredients
│   ├── clinicalbb_model#1.ipynb  # TabNet training notebook
│   ├── clinicalbb_model#2.ipynb  # Portion engine notebook
│   ├── train_model1.py
│   ├── train_model2.py
│   └── ...
│
└── supabase/
    ├── config.toml
    └── migrations/               # PostgreSQL schema migrations
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pytorch_tabnet'`**
Run `pip install pytorch_tabnet torch` separately, then retry.

**TabNet model fails to load on startup**
Ensure `artifacts/models/model_params.json` and `artifacts/models/network.pt` both exist and have not been moved. `network.pt` is tracked via Git LFS — run `git lfs pull` if it is missing.

**UnicodeEncodeError on Windows at startup**
Set `PYTHONIOENCODING=utf-8` before running: `set PYTHONIOENCODING=utf-8 && python app.py`

**Groq API error during recipe generation**
Check `GROQ_API_KEY` in `.env`. Free-tier keys at https://console.groq.com.

**Roboflow returns no detections**
Ensure `ROBOFLOW_API_KEY` is set and the image is a clear, well-lit photo of raw ingredients. Model: `ingredient-detection-5uzov/5`.

**HF Space cold-start timeout**
HF free-tier Spaces sleep after inactivity. The frontend shows a wake-up notice automatically — wait 20–30 seconds for the container to restart.

**Supabase credentials missing (non-fatal)**
`Supabase credentials not found` is a warning, not an error. The app works fully without Supabase; storage is silently skipped.
>>>>>>> 196878a9a3afb1baa3aa57e1a4a883260ad270e7

---

## Clinical Guidelines Referenced

<<<<<<< HEAD
- **KDOQI 2020** — CKD nutrition guidelines (potassium, protein, phosphorus)
- **AHA/ACC 2017** — Hypertension management (sodium limits)
- **ADA 2024** — Diabetes nutrition therapy (carbohydrate targets)
- **KDIGO 2017** — CKD staging and eGFR classification
=======
| Guideline | Scope |
|---|---|
| KDIGO 2024 | CKD staging, potassium and protein limits, eGFR thresholds |
| KDOQI 2020 | CKD nutrition — phosphorus and protein |
| AHA/ACC 2017 | Hypertension — sodium limits, BP targets |
| ADA 2024 | Type 2 Diabetes — carbohydrate and glycaemic targets |
>>>>>>> 196878a9a3afb1baa3aa57e1a4a883260ad270e7
