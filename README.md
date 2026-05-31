# Credit Scoring Business Understanding

## Basel II and the Need for Interpretable Credit Risk Models

The Basel II Accord emphasizes the importance of accurate risk measurement and sound risk management practices within financial institutions. Under Basel II, banks are required to quantify credit risk and maintain sufficient capital reserves to absorb potential losses. Because credit decisions directly affect regulatory capital requirements, financial institutions must be able to explain how risk estimates are generated.

As a result, credit scoring models should not only be accurate but also interpretable and well documented. Regulators, auditors, risk managers, and business stakeholders need to understand the factors influencing a customer's risk score. Transparent models make it easier to validate assumptions, identify potential biases, demonstrate regulatory compliance, and justify lending decisions. Well-documented models also support governance processes such as model monitoring, validation, and periodic review.

---

## The Need for a Proxy Variable and Its Associated Risks

In many real-world credit datasets, a direct "default" label is not available. Instead, organizations must construct a proxy variable that approximates default behavior using available information. For example, a customer may be classified as high risk if they are more than 90 days past due, have multiple missed payments, or exhibit prolonged delinquency.

A proxy variable is necessary because supervised machine learning models require a target variable for training. Without a target that represents credit risk, the model cannot learn the relationship between customer characteristics and repayment outcomes.

However, proxy-based prediction introduces several business risks:

* **Misclassification Risk:** The proxy may not perfectly represent actual default behavior, causing good customers to be labeled as risky or risky customers to be labeled as safe.
* **Model Bias:** If the proxy reflects historical business practices or operational constraints rather than true credit risk, the model may inherit those biases.
* **Reduced Predictive Accuracy:** The model learns patterns associated with the proxy rather than actual defaults, potentially reducing performance when deployed.
* **Regulatory Concerns:** Poorly defined proxies can make model decisions difficult to justify during audits or regulatory reviews.
* **Financial Impact:** Incorrect risk assessment can lead to increased loan losses, missed lending opportunities, or inefficient capital allocation.

Therefore, the definition of the proxy variable should be carefully designed, documented, and validated against business objectives.

---

## Trade-offs Between Interpretable and High-Performance Models

In regulated financial environments, there is often a trade-off between model interpretability and predictive performance.

### Logistic Regression with Weight of Evidence (WoE)

**Advantages**

* Highly interpretable and transparent.
* Easy to explain to regulators and business stakeholders.
* Coefficients clearly show how each variable influences risk.
* Supports regulatory compliance and model governance.
* Stable and straightforward to monitor over time.

**Disadvantages**

* May fail to capture complex non-linear relationships.
* Typically achieves lower predictive performance than advanced machine learning models.
* Requires significant feature engineering and binning.

### Gradient Boosting Models (e.g., XGBoost, LightGBM)

**Advantages**

* Often achieves superior predictive accuracy.
* Captures non-linear relationships and variable interactions automatically.
* Handles large and complex datasets effectively.
* Can improve risk discrimination and reduce prediction errors.

**Disadvantages**

* Less transparent and more difficult to explain.
* Model decisions may appear as a "black box" to stakeholders.
* Requires additional explainability techniques such as SHAP values or feature importance analysis.
* Can create challenges for regulatory approval, validation, and governance.

### Business Perspective

For highly regulated lending environments, interpretability is often as important as predictive accuracy. Logistic Regression with WoE remains a widely accepted industry standard because it balances performance with transparency and regulatory compliance. However, many institutions increasingly use Gradient Boosting models when higher predictive power is required, often supplementing them with explainability tools and robust model governance frameworks.

The choice between the two approaches depends on the organization's regulatory obligations, risk appetite, explainability requirements, and business objectives.


# Credit Risk Modeling & Target Engineering Pipeline

An end-to-end, production-grade credit risk modeling system. This repository implements automated data preprocessing, custom behavioral proxy target engineering via RFM clustering, experiment tracking with MLflow, containerized API deployment using FastAPI and Docker, and a continuous integration pipeline via GitHub Actions.

---

## 🛠️ Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline (Linting & Unit Tests)
├── data/                          # Ignored by git
│   ├── raw/                       # Immutable raw transaction data
│   └── processed/                 # Model-ready features with engineered proxy labels
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis & statistical visualization
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Sklearn Pipelines, feature engineering, WoE/IV, & RFM target assignment
│   ├── train.py                   # Model training, hyperparameter optimization, & MLflow logging
│   ├── predict.py                 # Batch/Single inference execution script
│   └── api/
│       ├── main.py                # FastAPI server utilizing registered MLflow model
│       └── pydantic_models.py     # Strict data validation schemas for API IO
├── tests/
│   └── test_data_processing.py    # Pytest unit tests for the transformation pipeline
├── Dockerfile                     # Containerization blueprint for the FastAPI application
├── docker-compose.yml             # Local multi-container deployment orchestration
├── requirements.txt               # Main pinned Python dependencies
├── .gitignore
└── README.md

```

---

## 🏛️ Credit Scoring Business Understanding

### 1. Basel II Accord Compliance & Model Interpretability

The **Basel II Accord** (specifically Pillar 1) establishes strict capital adequacy requirements determined by credit risk assessments. Financial institutions can use internal ratings-based (IRB) approaches to calculate regulatory capital, which directly scales with the model’s estimation of Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).

Because these estimations dictate the literal cash reserves a bank must lock away, **interpretability and rigorous documentation are non-negotiable regulatory mandates**:

* **Auditability & The "Black Box" Ban:** Regulators (such as the ECB, Federal Reserve, or central banks) must be able to trace exactly how a risk score is derived. Uninterpretable models cannot be verified against systemic bias or logical flaws.
* **Capital Optimization:** If a model cannot be clearly audited, regulators impose strict capital "add-ons" (penalties), forcing the institution to hold more idle capital. Clear documentation proves to auditors that the model's risk parameters are stable and statistically sound.
* **Right to Explanation:** Consumer protection laws often require financial institutions to provide adverse action notices explaining precisely *why* a customer was denied credit (e.g., "debt-to-income ratio too high"). A fully interpretable model ensures these compliance reasons can be mathematically extracted.

### 2. Proxy Variable Engineering & Associated Business Risks

In credit analytics, an explicit "default" label usually requires observing a borrower over an extended performance window (e.g., tracking 90+ days past due over 12–24 months). Without direct historical default labels, a **proxy variable** must be constructed using observable behavioral data—such as transactional inactivity, rapid balance depletion, or low engagement.

While necessary to jump-start modeling, introducing a proxy target injects specific business risks:

* **Label Misclassification (Basis Risk):** A highly active customer might stop using an account simply because they switched banks, not because they are insolvent. Labeling them "high-risk" causes **Type I errors (False Positives)**, resulting in lost revenue from creditworthy clients. Conversely, an inactive account might belong to a borrower who is quietly accumulating massive debts elsewhere, causing **Type II errors (False Negatives)** and unexpected credit losses.
* **Concept Drift / Proxy Decay:** Behavioral patterns shift over time due to macroeconomic changes (e.g., inflation, new payment apps). A proxy definition that holds true today might fail tomorrow, creating an artificial skew in who the model flags as risky.
* **Optimization Alignment Failure:** The machine learning model will optimize perfectly to predict the *proxy* (disengagement), not actual financial default. If the business relies entirely on this alignment, the credit policy may inadvertently restrict healthy but low-frequency transactional accounts.

### 3. Model Architecture Trade-offs in Regulated Finance

| Dimension | Simple Model (e.g., Logistic Regression + Weight of Evidence) | High-Performance Model (e.g., Gradient Boosting / XGBoost) |
| --- | --- | --- |
| **Interpretability** | **Exceptional.** WoE transforms continuous features into monotonic bins. The resulting model coefficients map directly to a traditional credit scorecard point system that any loan officer can read. | **Poor.** Ensemble trees rely on complex, non-linear interactions and splits that cannot be easily translated into a simple linear point system. |
| **Predictive Power** | **Moderate.** Struggles to capture complex, non-linear feature interactions naturally unless explicitly engineered by hand. | **High.** Automatically captures deep interaction effects and non-linear boundaries, leading to significantly higher ROC-AUC and Precision-Recall metrics. |
| **Regulatory Approval** | **Highly Streamlined.** It is the global industry standard. The math is transparent, local explanations are inherent, and checking for stability (e.g., Population Stability Index) is mathematically straightforward. | **Challenging.** Requires complex post-hoc explainability frameworks (e.g., SHAP, LIME). Auditors may reject it if the explanation cannot guarantee global stability across all edge cases. |
| **Operational Risk** | **Low.** Extremely robust to overfitting; predictable behavior on out-of-distribution data. | **Higher.** Prone to subtle overfitting if hyperparameters aren't strictly regularized; can exhibit erratic behavior on extreme, unseen edge cases. |

---

## 🏃‍♂️ Quick Start

### Prerequisites

* Python 3.10 or 3.11
* Docker & Docker Compose
* MLflow (installed via `requirements.txt`)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Executing the End-to-End Pipeline

The project is built around automated steps triggered sequentially across development branches:

```bash
# 1. Run exploratory data analysis (via Jupyter)
jupyter notebook notebooks/eda.ipynb

# 2. Run the feature engineering pipeline & target generation
python src/data_processing.py

# 3. Spin up the local MLflow tracking server in the background
mlflow server --host 127.0.0.1 --port 5000 &

# 4. Train models, perform hyperparameter tuning, and log to MLflow
python src/train.py

# 5. Execute local unit tests to ensure pipeline integrity
pytest tests/

```

### 3. Running the Local API Container

To build and run the FastAPI service locally using Docker Compose:

```bash
docker-compose up --build

```

The API will be available at `http://localhost:8000`. You can access interactive API documentation at `http://localhost:8000/docs`.

---

## 📝 Pipeline & Task Workflows

### Task 2: Exploratory Data Analysis (EDA)

* All core data distribution visualizations, outlier detection (Box Plots), missing value patterns, and feature correlation matrices are sandboxed inside `notebooks/eda.ipynb`.
* Top insights are compiled at the bottom of the notebook to dictate down-stream engineering choices.

### Task 3 & 4: Feature Pipeline & Proxy Targets

* **`src/data_processing.py`** encapsulates a strict `sklearn.pipeline.Pipeline` object.
* **Feature Generation:** Computes rolling customer aggregate statistics (Total, Average, Count, and Std Dev of transactions) alongside datetime extractions.
* **WoE & IV:** Categorical and continuous values are transformed via Weight of Evidence (WoE) to ensure monotonic relationships.
* **RFM Target Engineering:** Because a native default label does not exist, the script computes Recency, Frequency, and Monetary metrics per customer, applies `KMeans(n_clusters=3)`, and flags the low-frequency/low-monetary cluster as the high-risk target group (`is_high_risk = 1`).

### Task 5: Model Tracking & Registry

* **`src/train.py`** splits the data reproducibly, initializes baseline models (Logistic Regression vs. Gradient Boosting), and executes Grid/Random hyperparameter searches.
* Metrics ($Accuracy$, $Precision$, $Recall$, $F1$, and $ROC\text{-}AUC$) along with artifacts are logged directly to a local or remote **MLflow** server. The champion model is promoted programmatically to the MLflow Model Registry.

### Task 6: CI/CD Deployment

* **FastAPI:** Exposes a secure `/predict` endpoint that validates inputs via Pydantic schemas.
* **GitHub Actions:** `.github/workflows/ci.yml` enforces code health on every pull request to `main` by executing automated code style linters (`flake8`/`black`) and test suites (`pytest`).
