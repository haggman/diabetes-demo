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
- A deployment of that agent to **Agent Runtime** (formerly Vertex AI Agent Engine)

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
✓ Wrote diabetes_agent/.env
✓ Environment configured:
  PROJECT_ID:   your-project-id
  DATASET:      demo_diabetes
  BQ LOCATION:  US
  AGENT REGION: us-central1
  AGENT MODEL:  gemini-flash-latest

Ready to proceed with demo setup!
```

`activate.sh` also writes `diabetes_agent/.env` from these values. That file is how
your project id reaches the agent - both when you run it locally and when you
deploy it to Agent Runtime later. Re-source `activate.sh` any time you switch
projects.

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

Note: the next three steps may also be accomplished by running `source 3_setup_python.sh`

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

You should see `2.8.0` or higher. This demo requires ADK 2.x.

> **Note on dependencies:** `requirements.txt` lives in the repo root, and
> `diabetes_agent/requirements.txt` is a symlink pointing back at it. `adk deploy`
> installs the file it finds *inside the agent folder*, and it follows the symlink
> when it packages the agent - so the root file stays the single source of truth
> for both local runs and deployments. Note the `google-adk[gcp]` extra: the
> BigQuery toolset needs `google-cloud-bigquery` and `google-cloud-dataplex`, and
> plain `google-adk` installs neither.

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
adk web --allow_origins "*"
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

## Deploy the Agent to Agent Runtime

Everything so far has run on your machine. Agent Runtime is the managed runtime
that hosts the agent for you - sessions, memory, scaling, IAM and tracing
included. It is the service that used to be called **Vertex AI Agent Engine**,
now part of the Gemini Enterprise Agent Platform.

You will see all three names in the tooling, and they all mean the same thing:

| Where you see it | What it says |
|------------------|--------------|
| Product / console | Agent Runtime |
| ADK CLI subcommand | `adk deploy agent_engine` |
| REST resource path | `.../reasoningEngines/RESOURCE_ID` |

### Step 11: Understand what travels with the agent

Three things are packaged and sent to the runtime:

| What | Where it comes from |
|------|--------------------|
| Agent code | the `diabetes_agent/` folder |
| Python dependencies | `diabetes_agent/requirements.txt` (the symlink to the root file) |
| Environment variables | `diabetes_agent/.env`, generated by `activate.sh` |

That last row is the one people get wrong. `PROJECT_ID` is a shell variable in
Cloud Shell; it does not exist inside the deployed container. The deploy reads
`diabetes_agent/.env` and turns each line into an environment variable on the
running agent, which is how `prompts.py` still knows which project holds the
`demo_diabetes` dataset. Confirm the file looks right before deploying:

```bash
cat diabetes_agent/.env
```

```
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
PROJECT_ID=your-project-id
BQ_DATASET=demo_diabetes
AGENT_MODEL=gemini-flash-latest
```

`GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` do double duty: the deploy
also uses them to decide *where* to deploy, unless you pass `--project` and
`--region` explicitly (`4_deploy.sh` passes both).

### Step 12: Enable the APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project "${PROJECT_ID}"
```

### Step 13: Give the agent's service account access to BigQuery

**This is the step that catches everyone.** Running locally, the agent uses your
Application Default Credentials, and you have BigQuery Admin. Deployed, the exact
same `google.auth.default()` call returns the *Reasoning Engine service agent* -
a different identity that starts with no BigQuery access at all.

The symptom is specific and easy to misread: the agent deploys successfully,
answers general diabetes questions correctly (those use search), and fails only
when someone asks a data question.

```bash
# Make sure the service agent exists in this project
gcloud beta services identity create \
  --service=aiplatform.googleapis.com \
  --project="${PROJECT_ID}"

# Work out its email address
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
AGENT_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
echo "${AGENT_SA}"

# Let it run queries...
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${AGENT_SA}" \
  --role="roles/bigquery.jobUser" \
  --condition=None

# ...and read the table, the model, and the prediction function
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${AGENT_SA}" \
  --role="roles/bigquery.dataViewer" \
  --condition=None
```

Read-only roles are deliberate here - they match `WriteMode.BLOCKED` in
`agent.py`. For a tighter setup, grant `dataViewer` on the `demo_diabetes`
dataset only rather than the whole project.

### Step 14: Deploy

Steps 12 and 13 plus the deploy itself are all in one script:

```bash
source 4_deploy.sh
```

Or run the deploy on its own:

```bash
adk deploy agent_engine \
  --project="${PROJECT_ID}" \
  --region="${AGENT_REGION}" \
  --display_name="Diabetes Risk Agent" \
  diabetes_agent
```

The first deploy takes several minutes - it builds a container image. When it
finishes you get a resource name ending in a numeric id:

```
Created a new instance: projects/123456789/locations/us-central1/reasoningEngines/8901234567890
Deployed to Agent Platform: projects/123456789/locations/us-central1/reasoningEngines/8901234567890
```

To **redeploy over the same instance** instead of creating a second one, keep the
id and re-run:

```bash
export AGENT_ENGINE_ID=8901234567890
source 4_deploy.sh
```

> **Flags from older tutorials that no longer apply:** `--staging_bucket`,
> `--env_file`, `--requirements_file`, `--adk_app` and `--trace_to_cloud` are all
> deprecated in ADK 2.x. Configuration now comes from the agent folder itself
> (`.env`, `requirements.txt`, and optionally `.agent_engine_config.json`), and
> tracing is `--otel_to_cloud`.

### Step 15: Test the deployed agent

**From the console** (easiest): open Agent Runtime in the Google Cloud console,
select the agent, and use the built-in test pane. Ask the same four questions from
Step 10 - the dataset question is the one that proves Step 13 worked.

**Over REST:**

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://${AGENT_REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${AGENT_REGION}/reasoningEngines/${AGENT_ENGINE_ID}:streamQuery?alt=sse" \
  -d '{
        "class_method": "stream_query",
        "input": {
          "user_id": "demo-user",
          "message": "What percentage of people in your diabetes study dataset actually had diabetes?"
        }
      }'
```

**Sharing the deployed session store with your local UI:**

```bash
adk web --allow_origins "*" --session_service_uri="agentengine://${AGENT_ENGINE_ID}"
```

This runs the agent locally but stores sessions in Agent Runtime, so a
conversation started in the cloud shows up in your local UI. It is a nice way to
show that sessions are a managed service rather than something in memory.

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
| **`ImportError: cannot import name 'dataplex_v1'`** | The BigQuery toolset needs the `[gcp]` extra. Reinstall with `pip install -r requirements.txt` - the entry must be `google-adk[gcp]`, not `google-adk` |
| **`ResolutionImpossible` on pip install** | Install everything from `requirements.txt` in one command rather than package by package, so pip can resolve `google-genai` and OpenTelemetry versions jointly |
| **`[EXPERIMENTAL] feature FeatureName.GOOGLE_CREDENTIALS_CONFIG is enabled`** | Harmless warning from `BigQueryCredentialsConfig` on startup. Ignore it |
| **Agent talks about project `your-project-id`** | `PROJECT_ID` was not set. Run `source activate.sh`, and for a deployed agent confirm `diabetes_agent/.env` had the right project *before* you deployed |
| **Deployed agent: "permission denied" on any data question** | The Reasoning Engine service agent has no BigQuery access. Run Step 13, then ask again - no redeploy needed |
| **Deploy installs the wrong dependencies** | `adk deploy` reads `diabetes_agent/requirements.txt`. If that symlink is missing it writes a minimal file with no BigQuery support. Recreate it: `ln -sfn ../requirements.txt diabetes_agent/requirements.txt` |
| **Every deploy creates a new agent** | Set `AGENT_ENGINE_ID` to the resource id from the first deploy before re-running `4_deploy.sh` |

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

# Confirm the ADK version (2.x required)
adk --version

# Confirm the deploy will see the right dependencies and environment
cat diabetes_agent/requirements.txt
cat diabetes_agent/.env
```

---

## Clean Up

To avoid incurring charges, clean up resources when done:

```bash
# Delete the deployed agent (skip if you never ran 4_deploy.sh)
python - <<'PY'
import os, vertexai
client = vertexai.Client(project=os.environ["PROJECT_ID"],
                         location=os.environ.get("AGENT_REGION", "us-central1"))
for agent in client.agent_engines.list():
    print("deleting", agent.api_resource.name)
    client.agent_engines.delete(name=agent.api_resource.name, force=True)
PY

# Delete the BigQuery dataset and all contents
bq rm -r -f -d ${PROJECT_ID}:${BQ_DATASET}

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
