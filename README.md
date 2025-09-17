# Demo: BigQuery ML + ADK Agent for Diabetes Prediction

## Overview
This demo showcases how Google Cloud's BigQuery ML and Agent Development Kit (ADK) can accelerate healthcare analytics and AI development. We'll build a diabetes risk prediction model using real-world data, then create an intelligent agent that can answer questions, analyze the dataset, and provide personalized risk assessments - all without complex infrastructure setup.

**Key Technologies:**
- **BigQuery**: Google's serverless data warehouse for massive-scale analytics
- **BigQuery ML (BQML)**: Train and deploy ML models using just SQL
- **Agent Development Kit (ADK)**: Google's framework for building production-ready AI agents

**Dataset:** Public diabetes prediction dataset with 100,000 patient records including demographics, medical history, and lab results.

> ⚠️ **Clinical Disclaimer:** This is an educational demonstration only. The model and predictions are NOT validated for clinical use and should NOT be used for medical diagnosis or treatment decisions.

---

## What We'll Build

### Phase 1: Data Foundation & ML Model (Steps 1-6)
Load healthcare data into BigQuery and train an explainable logistic regression model to predict diabetes risk - all with just SQL, no data preprocessing needed.

### Phase 2: Direct Model Predictions (Step 7)
Demonstrate how to run predictions directly against the model using simple SQL - no views or special functions needed.

### Phase 3: Intelligent Agent Evolution (Steps 8-10)
Build an ADK agent that evolves through three versions:
- **v1**: Answers general diabetes questions using web search
- **v2**: Analyzes our actual dataset with dynamic SQL queries
- **v3**: Conducts personalized risk assessments through conversational interaction

---

## Part 1: Environment Setup

### Prerequisites
- Google Cloud project with BigQuery enabled
- Access to Cloud Shell (click the `>_` icon in the top-right of the GCP Console)
- The diabetes dataset is pre-staged at: `gs://class-demo/diabetes_prediction_dataset.csv`

### Step 1: Create Persistent Environment Configuration

Cloud Shell resets environment variables between sessions. We'll create a reusable configuration script:

```bash
# Create the activation script
cat > ~/activate.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

# Auto-detect or reuse PROJECT_ID
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "ERROR: PROJECT_ID is not set. Run: export PROJECT_ID=<your-project-id>" >&2
  return 1 2>/dev/null || exit 1
fi
export PROJECT_ID

# Dataset and model configuration
export BQ_DATASET="${BQ_DATASET:-demo_diabetes}"
export BQ_LOCATION="${BQ_LOCATION:-US}"
export GCS_URI="${GCS_URI:-gs://class-demo/diabetes_prediction_dataset.csv}"

# Table and model names
export RAW_TABLE="${BQ_DATASET}.diabetes_raw"
export MODEL_NAME="${BQ_DATASET}.diabetes_model"

echo "✓ Environment configured:"
echo "  PROJECT_ID: ${PROJECT_ID}"
echo "  DATASET: ${BQ_DATASET}"
echo "  LOCATION: ${BQ_LOCATION}"
echo ""
echo "Ready to proceed with demo setup!"
EOS

# Make executable and source it
chmod +x ~/activate.sh
source ~/activate.sh
```

**What this does:** Creates a script that sets all necessary environment variables. You can re-run `source ~/activate.sh` anytime you reconnect to Cloud Shell.

---

## Part 2: Data Pipeline & Model Training

### Step 2: Create the BigQuery Dataset

```bash
# Create a dedicated dataset for our diabetes demo
bq --location="${BQ_LOCATION}" mk \
  --dataset \
  --description "Healthcare demo: Diabetes prediction with BQML and ADK" \
  "${PROJECT_ID}:${BQ_DATASET}" || echo "Dataset already exists"
```

**What this does:** Creates a BigQuery dataset (logical container) to organize our tables and models. The `|| echo` prevents errors if it already exists.

### Step 3: Load the Raw Data

```bash
# Load CSV with automatic schema detection
bq --location="${BQ_LOCATION}" load \
  --autodetect \
  --skip_leading_rows=1 \
  --source_format=CSV \
  --replace \
  "${RAW_TABLE}" \
  "${GCS_URI}"

# Verify the load
echo "Data loaded. Checking row count..."
bq query --use_legacy_sql=false "SELECT COUNT(*) as total_records FROM \`${PROJECT_ID}.${RAW_TABLE}\`"
```

**What this does:** Loads 100,000 patient records from CSV into BigQuery. The `--autodetect` flag intelligently infers column types from the data.

### Step 4: Explore the Data (BigQuery Console)

Switch to the **BigQuery Console** in your browser:
1. Navigate to your project → `demo_diabetes` dataset → `diabetes_raw` table
2. Click **SCHEMA** tab to see the columns
3. Click **PREVIEW** tab to see sample records

**Key columns in our dataset:**
- **Demographics**: gender, age
- **Medical History**: hypertension, heart_disease, smoking_history
- **Clinical Measurements**: bmi, HbA1c_level (glycated hemoglobin), blood_glucose_level
- **Outcome**: diabetes (0 = No, 1 = Yes)

### Step 5: Train the Prediction Model

Run this SQL in the **BigQuery Console** to train a logistic regression model directly on the raw data:

```sql
-- Train a logistic regression model - BQML handles all the preprocessing!
CREATE OR REPLACE MODEL `demo_diabetes.diabetes_model`
OPTIONS (
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['diabetes'],
  auto_class_weights = TRUE,  -- Handle class imbalance
  data_split_method = 'AUTO_SPLIT',  -- 80/20 train/test split
  max_iterations = 20
) AS
SELECT 
  * 
FROM `demo_diabetes.diabetes_raw`;

-- Quick check of what BQML did with our data
SELECT 
  diabetes,
  COUNT(*) as count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM ML.TRAINING_INFO(MODEL `demo_diabetes.diabetes_model`)
GROUP BY diabetes
ORDER BY diabetes;
```

**What this does:** 
- Trains a logistic regression model directly on the raw data
- **BQML automatically handles**: data type casting, missing values, one-hot encoding of categorical variables (gender, smoking_history), and feature scaling
- Uses automatic class weighting to handle the imbalanced dataset (more non-diabetic patients)
- Splits data into training (80%) and validation (20%) sets
- Training typically completes in 10-30 seconds

**Key point for the demo:** "Notice we're training directly on raw data - no data cleaning or feature engineering needed. BQML handles all of that automatically!"

### Step 6: Evaluate Model Performance

### Step 6: Evaluate Model Performance

Run these queries in the **BigQuery Console** to assess model quality:

```sql
-- 1. Overall model metrics
SELECT 
  ROUND(roc_auc, 4) as AUC_ROC,
  ROUND(accuracy, 4) as accuracy,
  ROUND(precision, 4) as precision,
  ROUND(recall, 4) as recall,
  ROUND(f1_score, 4) as f1_score,
  ROUND(log_loss, 4) as log_loss
FROM ML.EVALUATE(MODEL `demo_diabetes.diabetes_model`);

-- 2. Feature weights (what factors matter most?)
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

-- 3. Confusion matrix at default threshold (0.5)
WITH predictions AS (
  SELECT 
    diabetes as actual_label,
    predicted_diabetes as predicted_label
  FROM ML.PREDICT(MODEL `demo_diabetes.diabetes_model`, 
    (SELECT * FROM `demo_diabetes.diabetes_raw`))
)
SELECT 
  CASE actual_label WHEN 1 THEN 'Has Diabetes' ELSE 'No Diabetes' END as actual,
  CASE predicted_label WHEN 1 THEN 'Has Diabetes' ELSE 'No Diabetes' END as predicted,
  COUNT(*) as count
FROM predictions
GROUP BY actual_label, predicted_label
ORDER BY actual_label DESC, predicted_label DESC;
```

**Expected results:**
- AUC-ROC around 0.85-0.95 (excellent discrimination)
- Top features likely include: HbA1c_level, blood_glucose_level, age, BMI
- Model should identify ~70-80% of diabetes cases (recall)

---

## Part 3: Test Direct Model Predictions

### Step 7: Demonstrate Direct Model Usage

Run this SQL in the **BigQuery Console** to show how easy it is to get predictions directly from the model:

```sql
-- Example 1: Prediction with complete patient data
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
    145 as blood_glucose_level  -- INT64, not FLOAT64
  )
);

-- Example 2: Prediction with missing lab values (using defaults)
SELECT 
  'Missing Labs Example' as scenario,
  ROUND(predicted_diabetes_probs[OFFSET(1)].prob * 100, 1) AS diabetes_probability_pct,
  CASE 
    WHEN predicted_diabetes_probs[OFFSET(1)].prob < 0.3 THEN 'Low Risk'
    WHEN predicted_diabetes_probs[OFFSET(1)].prob < 0.7 THEN 'Moderate Risk'
    ELSE 'High Risk'
  END AS risk_category,
  'Used default values for missing labs' AS note
FROM ML.PREDICT(
  MODEL `demo_diabetes.diabetes_model`,
  (SELECT 
    'Female' as gender,
    45.0 as age,
    0 as hypertension,
    0 as heart_disease,
    'never' as smoking_history,
    26.5 as bmi,  -- Using US average
    5.7 as HbA1c_level,  -- Pre-diabetic threshold
    100 as blood_glucose_level  -- Normal fasting level (INT64)
  )
);

-- Example 3: Batch predictions for multiple patients
SELECT 
  CONCAT('Patient ', CAST(ROW_NUMBER() OVER() AS STRING)) as patient_id,
  age,
  gender,
  ROUND(predicted_diabetes_probs[OFFSET(1)].prob * 100, 1) AS diabetes_probability_pct,
  CASE 
    WHEN predicted_diabetes_probs[OFFSET(1)].prob < 0.3 THEN 'Low Risk'
    WHEN predicted_diabetes_probs[OFFSET(1)].prob < 0.7 THEN 'Moderate Risk'
    ELSE 'High Risk'
  END AS risk_category
FROM ML.PREDICT(
  MODEL `demo_diabetes.diabetes_model`,
  (SELECT * FROM UNNEST([
    STRUCT('Female' AS gender, 35.0 AS age, 0 AS hypertension, 0 AS heart_disease,
           'never' AS smoking_history, 22.0 AS bmi, 5.2 AS HbA1c_level, 85 AS blood_glucose_level),
    STRUCT('Male' AS gender, 65.0 AS age, 1 AS hypertension, 1 AS heart_disease,
           'current' AS smoking_history, 32.0 AS bmi, 7.8 AS HbA1c_level, 180 AS blood_glucose_level),
    STRUCT('Female' AS gender, 50.0 AS age, 1 AS hypertension, 0 AS heart_disease,
           'former' AS smoking_history, 29.0 AS bmi, 6.5 AS HbA1c_level, 130 AS blood_glucose_level)
  ]))
);
```

**What this demonstrates:** 
- Direct model access without any intermediate views or functions
- How to handle missing data by using reasonable defaults
- Batch predictions for multiple patients at once
- The model returns probabilities that can be interpreted as risk levels

**Key talking point:** "Notice we don't need any special infrastructure or APIs - just SQL queries directly against the model. The agent will use this same pattern, dynamically building queries based on user input."

---

## Part 4: Building an ADK Agent

Now we'll create an intelligent agent that can discuss diabetes, analyze our dataset, and provide personalized risk assessments using the Agent Development Kit (ADK).

### Step 8: Create the Basic Agent Structure

First, let's scaffold a basic agent using ADK:

```bash
# Ensure you're in the workspace directory
cd ~/mlb-agent-lab/workspace

# Create the agent with ADK
adk create \
  --model gemini-2.0-flash-exp \
  --project $PROJECT_ID \
  --region $LOCATION \
  diabetes_agent

# Navigate to the new agent directory
cd diabetes_agent
```

This creates:
- `agent.py` - The main agent definition
- `.env` - Environment configuration
- `__init__.py` - Python module initialization

### Step 9: Create Agent Instructions

Create a separate file for agent instructions to keep things organized:

```bash
# Create prompts.py with agent personality and instructions
cat > prompts.py <<'EOF'
"""
Diabetes Risk Assessment Agent - Instructions and Configuration
"""

AGENT_DESCRIPTION = """
A knowledgeable healthcare information assistant that helps users understand 
diabetes risk factors and provides educational assessments based on validated data.
"""

AGENT_INSTRUCTIONS = """
You are a friendly healthcare information assistant specializing in diabetes education 
and risk assessment. You have access to a diabetes prediction model trained on 100,000 
patient records.

PERSONALITY:
- Warm, empathetic, and supportive
- Clear and accessible explanations
- Encouraging but realistic
- Professional yet approachable

CAPABILITIES:
1. Answer general questions about diabetes, risk factors, and prevention
2. Analyze patterns in the diabetes dataset
3. Provide personalized risk assessments (educational only)

IMPORTANT GUIDELINES:
- ALWAYS include this disclaimer prominently: "This is an educational tool only. 
  It is NOT medical advice. Please consult a healthcare provider for proper screening 
  and medical guidance."
- Be encouraging about lifestyle modifications
- Focus on modifiable risk factors when providing recommendations
- If asked off-topic questions, politely redirect to diabetes-related topics

RESPONSE STYLE:
- Use clear, non-technical language
- Provide context for medical terms
- Be concise but thorough
- Include relevant statistics when helpful
"""
EOF
```

### Step 10: Update the Agent with Google Search

Now let's update the agent to use our instructions and add Google Search for general diabetes information:

```bash
# Update agent.py
cat > agent.py <<'EOF'
"""
Diabetes Risk Assessment Agent - Educational healthcare assistant
"""

from google.adk.agents import Agent
from google.adk.tools import Tool
from prompts import AGENT_DESCRIPTION, AGENT_INSTRUCTIONS

# Create the agent with Google Search capability
root_agent = Agent(
    name="diabetes_agent",
    model="gemini-2.0-flash-exp",
    description=AGENT_DESCRIPTION,
    instruction=AGENT_INSTRUCTIONS,
    tools=[Tool.google_search()],  # Add Google Search tool
)

# Debug output when run directly
if __name__ == "__main__":
    print(f"✅ Diabetes agent configured")
    print(f"📋 Name: {root_agent.name}")
    print(f"🧠 Model: {root_agent.model}")
    print(f"🛠️ Tools: {len(root_agent.tools)} configured")
EOF
```

Test the basic agent:

```bash
# Quick test from workspace directory
cd ~/mlb-agent-lab/workspace
python -m diabetes_agent.agent
```

### Step 11: Add BigQuery Access

Now let's give the agent access to our BigQuery data. First, create a BigQuery tool configuration:

```bash
cd diabetes_agent

# Create bigquery_tools.py
cat > bigquery_tools.py <<'EOF'
"""
BigQuery tools for accessing diabetes data and model
"""

import os
from typing import Dict, Any
from google.cloud import bigquery
from google.adk.tools import Tool

PROJECT_ID = os.environ.get('PROJECT_ID')
DATASET_ID = 'demo_diabetes'

def analyze_diabetes_data(query: str) -> str:
    """
    Execute a SQL query against the diabetes dataset.
    
    Args:
        query: SQL query to run against the demo_diabetes dataset
    
    Returns:
        Query results as a formatted string
    """
    client = bigquery.Client(project=PROJECT_ID)
    
    # Add safety check for dataset
    if DATASET_ID not in query:
        query = query.replace('FROM ', f'FROM `{PROJECT_ID}.{DATASET_ID}.')
    
    try:
        query_job = client.query(query)
        results = query_job.result()
        
        # Format results
        rows = list(results)
        if not rows:
            return "No results found."
        
        # Convert to readable format
        output = []
        for row in rows[:10]:  # Limit to 10 rows for readability
            output.append(str(dict(row)))
        
        return "\n".join(output)
    except Exception as e:
        return f"Query error: {str(e)}"

def predict_diabetes_risk(
    gender: str,
    age: float,
    hypertension: int,
    heart_disease: int,
    smoking_history: str,
    bmi: float,
    hba1c: float = 5.7,
    blood_glucose: int = 100
) -> str:
    """
    Predict diabetes risk using the trained model.
    
    Args:
        gender: 'Male' or 'Female'
        age: Age in years
        hypertension: 0 (No) or 1 (Yes)
        heart_disease: 0 (No) or 1 (Yes)
        smoking_history: 'never', 'former', 'current', etc.
        bmi: Body Mass Index
        hba1c: HbA1c level (default 5.7 if unknown)
        blood_glucose: Blood glucose level (default 100 if unknown)
    
    Returns:
        Risk assessment with probability and category
    """
    client = bigquery.Client(project=PROJECT_ID)
    
    query = f"""
    SELECT 
        ROUND(predicted_diabetes_probs[OFFSET(1)].prob * 100, 1) AS diabetes_probability_pct,
        CASE 
            WHEN predicted_diabetes_probs[OFFSET(1)].prob < 0.3 THEN 'Low Risk'
            WHEN predicted_diabetes_probs[OFFSET(1)].prob < 0.7 THEN 'Moderate Risk'
            ELSE 'High Risk'
        END AS risk_category
    FROM ML.PREDICT(
        MODEL `{PROJECT_ID}.{DATASET_ID}.diabetes_model`,
        (SELECT 
            '{gender}' as gender,
            {age} as age,
            {hypertension} as hypertension,
            {heart_disease} as heart_disease,
            '{smoking_history}' as smoking_history,
            {bmi} as bmi,
            {hba1c} as HbA1c_level,
            {blood_glucose} as blood_glucose_level
        )
    )
    """
    
    try:
        query_job = client.query(query)
        results = list(query_job.result())
        
        if results:
            row = results[0]
            return (f"Risk Assessment:\n"
                   f"- Probability: {row['diabetes_probability_pct']}%\n"
                   f"- Category: {row['risk_category']}\n"
                   f"- Note: This is for educational purposes only")
        return "Unable to calculate risk."
    except Exception as e:
        return f"Prediction error: {str(e)}"

def get_dataset_statistics() -> str:
    """
    Get basic statistics about the diabetes dataset.
    
    Returns:
        Summary statistics about the dataset
    """
    client = bigquery.Client(project=PROJECT_ID)
    
    query = f"""
    SELECT 
        COUNT(*) as total_records,
        ROUND(AVG(CASE WHEN diabetes = 1 THEN 100.0 ELSE 0 END), 1) as diabetes_percentage,
        ROUND(AVG(age), 1) as avg_age,
        ROUND(AVG(bmi), 1) as avg_bmi,
        ROUND(AVG(HbA1c_level), 1) as avg_hba1c
    FROM `{PROJECT_ID}.{DATASET_ID}.diabetes_raw`
    """
    
    try:
        query_job = client.query(query)
        results = list(query_job.result())
        
        if results:
            row = results[0]
            return (f"Dataset Statistics:\n"
                   f"- Total records: {row['total_records']:,}\n"
                   f"- Diabetes prevalence: {row['diabetes_percentage']}%\n"
                   f"- Average age: {row['avg_age']} years\n"
                   f"- Average BMI: {row['avg_bmi']}\n"
                   f"- Average HbA1c: {row['avg_hba1c']}%")
        return "Unable to retrieve statistics."
    except Exception as e:
        return f"Query error: {str(e)}"
EOF
```

### Step 12: Update Agent with All Tools

Update the agent to include BigQuery tools:

```bash
# Update agent.py to include BigQuery tools
cat > agent.py <<'EOF'
"""
Diabetes Risk Assessment Agent - Educational healthcare assistant
"""

from google.adk.agents import Agent
from google.adk.tools import Tool
from prompts import AGENT_DESCRIPTION, AGENT_INSTRUCTIONS
from bigquery_tools import (
    analyze_diabetes_data,
    predict_diabetes_risk, 
    get_dataset_statistics
)

# Enhanced instructions for data-aware agent
ENHANCED_INSTRUCTIONS = AGENT_INSTRUCTIONS + """

DATA ACCESS:
You have access to a BigQuery dataset (demo_diabetes) with:
- diabetes_raw: 100,000 patient records with demographics and clinical data
- diabetes_model: Trained logistic regression model for risk prediction

Use these tools:
- get_dataset_statistics(): Overview of the dataset
- analyze_diabetes_data(query): Run SQL queries for analysis
- predict_diabetes_risk(): Calculate personalized risk scores
- google_search(): Find current medical information

WORKFLOW FOR RISK ASSESSMENT:
1. Greet warmly and explain you can provide an educational risk assessment
2. Ask for information conversationally:
   - Demographics (age, gender)
   - Medical history (hypertension, heart disease)
   - Lifestyle (smoking history, BMI)
   - Lab values if known (HbA1c, blood glucose)
3. Use reasonable defaults for missing values and explain them
4. Run the prediction and interpret results supportively
5. Provide 2-3 personalized recommendations
6. Remind about consulting healthcare providers
"""

# Create the agent with all tools
root_agent = Agent(
    name="diabetes_agent",
    model="gemini-2.0-flash-exp",
    description=AGENT_DESCRIPTION,
    instruction=ENHANCED_INSTRUCTIONS,
    tools=[
        Tool.google_search(),
        Tool.from_function(get_dataset_statistics),
        Tool.from_function(analyze_diabetes_data),
        Tool.from_function(predict_diabetes_risk),
    ],
)

if __name__ == "__main__":
    print(f"✅ Diabetes agent configured")
    print(f"📋 Name: {root_agent.name}")
    print(f"🧠 Model: {root_agent.model}")
    print(f"🛠️ Tools: {len(root_agent.tools)} configured")
    print(f"📊 BigQuery dataset: demo_diabetes")
EOF
```

### Step 13: Test the Complete Agent

Start the ADK test interface:

```bash
cd ~/mlb-agent-lab/workspace
adk web
```

Open the web interface and select **diabetes_agent**. Test with progressively complex queries:

**Test 1 - General Knowledge:**
```
What are the main risk factors for type 2 diabetes?
```

**Test 2 - Dataset Analysis:**
```
What percentage of people in your dataset have diabetes?
```

**Test 3 - Complex Query:**
```
How does BMI correlate with diabetes risk in the dataset?
```

**Test 4 - Risk Assessment:**
```
Can you assess my diabetes risk?
```
Follow the conversational flow as the agent gathers information.

### Step 14: Deploy the Agent (Optional)

Deploy to Cloud Run for production access:

```bash
# Add requirements file
cat > diabetes_agent/requirements.txt <<EOF
google-adk
google-cloud-aiplatform
google-cloud-bigquery
pandas
EOF

# Deploy to Cloud Run
adk deploy cloud_run \
  --project $PROJECT_ID \
  --region $LOCATION \
  --service_name diabetes-risk-agent \
  --with_ui \
  diabetes_agent

# Update environment variables
gcloud run services update diabetes-risk-agent \
  --region $LOCATION \
  --update-env-vars PROJECT_ID=$PROJECT_ID
```

### What You've Built

Your diabetes risk assessment agent now:
- ✅ Answers general diabetes questions using web search
- ✅ Analyzes patterns in your 100k patient dataset  
- ✅ Provides personalized risk assessments using your ML model
- ✅ Maintains appropriate medical disclaimers
- ✅ Guides users through conversational risk assessment

The architecture demonstrates how ADK agents can combine:
- Built-in LLM knowledge
- External web information
- Private analytical data
- Machine learning predictions

This same pattern works for any healthcare or life sciences application where you need to blend general knowledge with specific analytical capabilities.

---

## Demo Flow & Talk Track

### Opening (2 minutes)
"Today I'll show you how Google Cloud can accelerate healthcare AI development. We'll go from raw data to an intelligent agent in under 30 minutes - something that traditionally takes weeks or months."

### Part 1: Data & Model (5 minutes)
- Show the BigQuery console with loaded data
- Run a quick SQL query to show the data
- Show the model training SQL - emphasize it's just SQL, no Python/R needed
- Show model metrics - "85% AUC means good predictive power"
- Show feature importance - "HbA1c and glucose are top predictors, which aligns with medical knowledge"

### Part 2: Agent Evolution (10 minutes)

**Version 1 Demo:**
- "First, a simple agent that answers general questions"
- Ask: "What are early signs of diabetes?"
- "Notice it uses web search and cites reputable sources"

**Version 2 Demo:**
- "Now let's add data analysis capabilities"
- Ask: "What percentage of patients over 50 have diabetes in our dataset?"
- Show the SQL it generates and the results
- "The agent writes the query, runs it, and interprets results"

**Version 3 Demo:**
- "Finally, personalized risk assessment"
- Start: "Can you assess my diabetes risk?"
- Walk through the conversational flow
- "Notice how it gathers information naturally, runs the prediction, and provides personalized guidance"

### Closing (3 minutes)
"What we've built today could support population health screening, clinical decision support, or patient engagement apps. The key advantages:
- No infrastructure to manage with BigQuery
- Models deploy instantly with BQML
- ADK agents can be enhanced without coding
- Everything scales automatically

For life sciences companies, this means faster time to insights, reduced development costs, and the ability to focus on science rather than infrastructure."

---

## Additional Resources

- **BigQuery ML Documentation**: https://cloud.google.com/bigquery-ml/docs
- **ADK Documentation**: https://google.github.io/adk-docs/
- **Healthcare & Life Sciences Solutions**: https://cloud.google.com/solutions/healthcare-life-sciences
- **ADK Sample Code**: https://github.com/google/adk-samples
- **Diabetes Prediction Dataset**: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

---

## Troubleshooting

**Common Issues:**

1. **Permission Errors**: Ensure your account has BigQuery Admin role
2. **Location Mismatches**: Keep all resources in the same location (US)
3. **Model Training Fails**: Check for NULL values in training data
4. **Agent Can't Access BigQuery**: Verify ADK service account has BigQuery access
5. **Predictions Return NULL**: Ensure all input parameters are provided with correct types

**Quick Fixes:**
```bash
# Grant BigQuery access to ADK service account
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:adk-agent@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.user"
```

