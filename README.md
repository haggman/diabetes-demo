# Demo: BigQuery ML + ADK Agent for Diabetes Prediction

## Overview

This demo showcases how Google Cloud's BigQuery ML and Agent Development Kit (ADK) can accelerate healthcare analytics and AI development. You'll build a diabetes risk prediction model using real-world data, then create an intelligent agent that can answer questions, analyze the dataset, and provide personalized risk assessments - all without complex infrastructure setup.

### What You'll Build

- A machine learning model trained on 100,000 patient records
- A conversational AI agent that combines:
  - General diabetes knowledge from web search
  - Dataset analytics using BigQuery
  - Personalized risk predictions using your trained model
  - Appropriate medical disclaimers and safety guidelines

### Key Technologies

- **BigQuery**: Google's serverless data warehouse for massive-scale analytics
- **BigQuery ML (BQML)**: Train and deploy ML models using just SQL
- **Agent Development Kit (ADK)**: Google's framework for building production-ready AI agents

### Dataset

Public diabetes prediction dataset with 100,000 patient records including demographics, medical history, and lab results.

> ⚠️ **Clinical Disclaimer:** This is an educational demonstration only. The model and predictions are NOT validated for clinical use and should NOT be used for medical diagnosis or treatment decisions.

---

## Prerequisites

Before starting, ensure you have:

1. **Google Cloud Project**: An active GCP project with billing enabled
2. **Required APIs**: Access to Gemini and BigQuery
3. **Cloud Shell**: Recommended environment (has gcloud CLI pre-installed)
   - Alternative: Local machine with gcloud SDK configured

---

## Setup Instructions

### Step 1: Enable APIs

1. Navigate to the [Vertex AI Console](https://console.cloud.google.com/vertex-ai)
2. Click **"Enable all recommended APIs"** when prompted

### Step 2: Clone the Repository

Open the Cloud Shell terminal (or your local terminal) and run:

```bash
git clone https://github.com/haggman/diabetes-demo
cd diabetes-demo
```

### Step 3: Configure Environment Variables

1. Review the configuration script:

```bash
edit activate.sh
```

2. Activate the environment:
```bash
source activate.sh
```

You should see:
```
✔ Environment configured:
  PROJECT_ID: your-project-id
  DATASET: demo_diabetes
  LOCATION: US

Ready to proceed with demo setup!
```

### Step 4: Load Data into BigQuery

1. Review the setup script to understand what it does:

```bash
edit 1_setup_bq.sh
```

2. Execute the script to create the dataset and load data:
```bash
source 1_setup_bq.sh
```

3. Verify the data loaded correctly:
   - Go to [BigQuery Console](https://console.cloud.google.com/bigquery)
   - Navigate to your project → `demo_diabetes` dataset → `diabetes_raw` table
   - Click **PREVIEW** to examine the data
   - Note the columns: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes

### Step 5: Train the ML Model

1. Review the model training script:

```bash
edit 2_train_model.sh
```

This script:
- Creates a logistic regression model for diabetes prediction
- Sets up a table-valued function (TVF) for easy predictions and that handles missing data with population averages (calculated from the study dataset)

2. Execute the script:

```bash
source 2_train_model.sh
```

Training takes approximately 1-3 minutes.

### Step 6: Explore the Model

Open the BigQuery console and run these queries to understand your model:

#### Query 1: Model Performance Metrics

```sql
SELECT 
  ROUND(roc_auc, 4) as AUC_ROC,
  ROUND(accuracy, 4) as accuracy,
  ROUND(precision, 4) as precision,
  ROUND(recall, 4) as recall,
  ROUND(f1_score, 4) as f1_score,
  ROUND(log_loss, 4) as log_loss
FROM ML.EVALUATE(MODEL `demo_diabetes.diabetes_model`);
```

#### Query 2: Feature Importance
```sql
SELECT 
  processed_input as feature,
  ROUND(ABS(weight), 4) as importance,
  CASE 
    WHEN weight > 0 THEN 'Increases risk'
    ELSE 'Decreases risk'
  END as effect_direction
FROM ML.WEIGHTS(MODEL `demo_diabetes.diabetes_model`)
WHERE processed_input != '__INTERCEPT__'
ORDER BY ABS(weight) DESC
LIMIT 10;
```

### Step 7: Test Predictions

Try these example predictions in BigQuery:

#### Example 1: Direct Model Prediction
```sql
SELECT 
  'Complete Data Example' as scenario,
  ROUND(predicted_diabetes_probs[OFFSET(1)].prob * 100, 1) AS diabetes_probability_pct,
  CASE 
    WHEN predicted_diabetes_probs[OFFSET(1)].prob < 0.3 THEN 'Low Risk'
    WHEN predicted_diabetes_probs[OFFSET(1)].prob < 0.7 THEN 'Moderate Risk'
    ELSE 'High Risk'
  END AS risk_category,
  'Educational demo only - not for clinical use' AS disclaimer
FROM ML.PREDICT(
  MODEL `demo_diabetes.diabetes_model`,
  (SELECT 
    'Male' as gender,
    55.0 as age,
    1 as hypertension,
    0 as heart_disease,
    'former' as smoking_history,
    28.5 as bmi,
    6.8 as HbA1c_level,
    145 as blood_glucose_level
  )
);
```

#### Example 2: Using the Table-Valued Function
```sql
SELECT * FROM `demo_diabetes.predict_diabetes`(
  'Male',     -- gender
  55.0,       -- age
  1,          -- hypertension (1=yes, 0=no)
  0,          -- heart_disease (1=yes, 0=no)
  'former',   -- smoking_history
  28.5,       -- bmi
  6.8,        -- HbA1c_level
  145         -- blood_glucose_level
);
```

---

## Loadup the AI Agent

### Step 8: Setup Python Environment

1. Switch back to the Cloud Shell terminal then create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install required packages (including the ADK):
```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

3. Verify ADK installation:
```bash
adk --version
```

You should see the ADK version number (e.g., `1.14.1` or higher).

### Step 9: Explore the Agent Architecture

1. **Review the agent configuration** (`diabetes_agent/agent.py`):
   - BigQuery tool setup for data analysis
   - Search agent for web information
   - Root agent orchestrating both capabilities

2. **Review the prompts** (`diabetes_agent/prompts.py`):
   - Agent personality and behavior
   - Risk assessment workflow
   - Safety guidelines and disclaimers

### Step 10: Test the Agent

1. Start the ADK development interface:
```bash
adk web
```

2. Click the `http://127.0.0.1:8080` link to open the agent in your browser.

3. Select **diabetes_agent** from the dropdown menu

4. Test with progressively complex queries:

#### Test 1 - General Knowledge
```
What are the main risk factors for type 2 diabetes?
```
*Expected: The agent uses web search to provide evidence-based information*

#### Test 2 - Dataset Analysis
```
What percentage of people in your diabetes study dataset actually had diabetes?
```
*Expected: The agent queries BigQuery to analyze the diabetes_raw table*

#### Test 3 - Complex Analysis
```
How does BMI correlate with diabetes risk in the dataset?
```
*Expected: The agent runs SQL to analyze patterns in the data*

#### Test 4 - Risk Assessment
```
Can you assess my diabetes risk?
```
*Expected: The agent guides you through providing information and then uses the prediction model. You might try this a couple of times with both partial and complete data.*


---

## Understanding the Architecture

### Data Flow

```
User Query → ADK Agent → Decision
                ↓
    ┌───────────┴───────────┐
    ↓                       ↓
BigQuery Tools          Search Agent
    ↓                       ↓
- Dataset queries      - Web search
- ML predictions       - Medical info
    ↓                       ↓
    └───────────┬───────────┘
                ↓
        Combined Response
```

### Key Components

1. **BigQuery Dataset** (`demo_diabetes`)
   - `diabetes_raw`: Training data with 100k records
   - `diabetes_model`: Trained logistic regression model
   - `predict_diabetes`: TVF for easy predictions

2. **ADK Agent**
   - **Root Agent**: Orchestrates the conversation
   - **Search Agent**: Retrieves web information
   - **BigQuery Tools**: Analyzes data and runs predictions

3. **Safety Features**
   - Medical disclaimers on all predictions
   - Educational purpose emphasis
   - Encouragement to consult healthcare providers

---

## Customization Ideas

### Enhance the Model
- Add feature engineering in the SQL
- Try different model types (DNN, XGBoost)
- Implement cross-validation

### Expand Agent Capabilities
- Add visualization generation
- Implement conversation memory
- Create follow-up appointment scheduling

### Dataset Improvements
- Add temporal analysis
- Include medication history
- Incorporate genetic markers

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| **Permission denied on BigQuery** | Ensure you have BigQuery Admin role: `gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="user:your-email@domain.com" --role="roles/bigquery.admin"` |
| **Dataset not found** | Verify PROJECT_ID is set correctly and dataset was created in Step 4 |
| **Model training fails** | Check for NULL values in data; ensure dataset location is US |
| **ADK web won't start** | Ensure virtual environment is activated and requirements installed |
| **Agent can't find data** | Verify PROJECT_ID environment variable is set in your shell |
| **Predictions return NULL** | Ensure all input parameters use correct data types (see TVF specification) |

### Debugging Commands

```bash
# Check environment variables
echo $PROJECT_ID
echo $BQ_DATASET

# Verify dataset exists
bq ls -d --project_id=$PROJECT_ID

# Check model status
bq show --model demo_diabetes.diabetes_model

# Test ADK installation
python -c "import google.adk; print('ADK OK')"
```

---

## Clean Up

To avoid incurring charges, clean up resources when done:

```bash
# Delete the BigQuery dataset and all contents
bq rm -r -f -d ${PROJECT_ID}:demo_diabetes

# Deactivate Python environment
deactivate

# Optional: Remove local files
cd ..
rm -rf diabetes-demo
```

---

## Additional Resources

- **BigQuery ML Documentation**: https://cloud.google.com/bigquery-ml/docs
- **ADK Documentation**: https://google.github.io/adk-docs/
- **Healthcare & Life Sciences Solutions**: https://cloud.google.com/solutions/healthcare-life-sciences
- **Diabetes Dataset on Kaggle**: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

---

## Security & Compliance Note

This demo is for educational purposes only. In production healthcare applications, ensure:
- HIPAA compliance for US healthcare data
- Proper encryption and access controls
- Audit logging for all data access
- Appropriate consent and privacy measures
- Clinical validation before any medical use

---

## Questions or Issues?

- Check the [troubleshooting section](#troubleshooting) above
- Review the [ADK documentation](https://google.github.io/adk-docs/)
- Explore [BigQuery ML tutorials](https://cloud.google.com/bigquery-ml/docs/tutorials)

---

**Remember**: This is an educational demonstration showcasing the integration of BigQuery ML and ADK. The predictions are NOT clinically validated and should NOT be used for medical decisions. Always consult healthcare professionals for medical advice.