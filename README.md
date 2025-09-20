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
Prerequisits:
- A project in Google Cloud 
- Access to Gemini, BigQuery, and if you do the optional deployment, Agent Engine
- Either the GCP SDK setup to work with the project, or Cloud Shell
---

## What We'll Build
...

---

If you haven't already done so, visit the Vertex AI console and **Enable all recommended APIs**

Clone down this repo either into Cloud Shell or onto your local machine `https://github.com/haggman/diabetes-demo`

Change into the `diabetes-demo` folder

Open and examine the `activate.sh`

. activate.sh

Open and examine `1_setup_bq.sh`. This pulls the diabetes dataset and loads it into BigQuery

. 1_setup_bq.sh

Jump over the BQ and take a few moments to explore the diabetes_raw table

Open and examine `2_train_model.sh`. Note how this uses BQML to train a diabetes prediction model and then exposes the model as a TVF (table-valued function)

Either run the two statements in the BQ page or execute . 2_train_model.sh

Switch to the BQ Studio UI and execute the 2 below queries to explore the model
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

Now let's do some predictions using SQL:

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

-- Example 2: Same prediction using the TVF
SELECT * FROM `demo_diabetes.predict_diabetes`(
  'Male',                  -- gender
  55.0,                    -- age
  1,                       -- hypertension
  0,                       -- heart_disease
  'former',                -- smoking_history
  28.5,                    -- bmi
  6.8,                     -- HbA1c_level
  145                      -- blood_glucose_level
);

Switch back to the Cloud Shell terminal and setup your Python Env

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt
```

Test to make sure ADK is installed and working properly

```bash
adk --version
```
Open and examine diabetes_agent/agent.py
Things to note:
- The ADK's native BigQuery and Google Search tool setups
- The search agent (since a single agent can't use two search tools directly)
- The root agent, with access to the search agent and the BQ toolset

Open and examine diabetes_agent/prompts.py and notice how the description and instructions are setup.


### Test the Complete Agent

Start the ADK test interface:

```bash
adk web
```

Open the web interface and select **diabetes_agent**. Test with progressively complex queries:

**Test 1 - General Knowledge:**
```
What are the main risk factors for type 2 diabetes?
```

**Test 2 - Dataset Analysis:**
```
What percentage of the people in your diabetes study dataset actually had diabetes?
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

