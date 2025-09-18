bq query --use_legacy_sql=false '
CREATE OR REPLACE MODEL `demo_diabetes.diabetes_model`
OPTIONS (
  model_type = "LOGISTIC_REG",
  input_label_cols = ["diabetes"],
  auto_class_weights = TRUE,
  data_split_method = "AUTO_SPLIT",
  max_iterations = 20
) AS SELECT 
  * FROM `demo_diabetes.diabetes_raw`;
'

bq query --use_legacy_sql=false '
CREATE OR REPLACE TABLE FUNCTION `demo_diabetes.predict_diabetes`(
  gender               STRING,
  age                  FLOAT64,
  hypertension         INT64,
  heart_disease        INT64,
  smoking_history      STRING,
  bmi                  FLOAT64,
  HbA1c_level          FLOAT64,
  blood_glucose_level  INT64
)
AS (
  WITH input AS (
    SELECT
      COALESCE(gender, 'Female') AS gender,
      COALESCE(age, CASE WHEN COALESCE(gender, 'Female') = 'Male' THEN 41.08 ELSE 42.46 END) AS age,
      COALESCE(hypertension, 0) AS hypertension,
      COALESCE(heart_disease, 0) AS heart_disease,
      COALESCE(smoking_history, 'never') AS smoking_history,
      COALESCE(bmi, CASE WHEN COALESCE(gender, 'Female') = 'Male' THEN 27.14 ELSE 27.45 END) AS bmi,
      COALESCE(HbA1c_level, CASE WHEN COALESCE(gender, 'Female') = 'Male' THEN 5.55 ELSE 5.51 END) AS HbA1c_level,
      COALESCE(blood_glucose_level, CASE WHEN COALESCE(gender, 'Female') = 'Male' THEN 139 ELSE 137 END) AS blood_glucose_level
  )
  SELECT
    gender,
    ROUND(age, 2) AS age,
    hypertension,
    heart_disease,
    smoking_history,
    ROUND(bmi, 1) AS bmi,
    ROUND(HbA1c_level, 1) AS hba1c,
    blood_glucose_level AS blood_glucose,
    CAST(predicted_diabetes AS INT64) AS prediction,
    ROUND((
      SELECT prob FROM UNNEST(predicted_diabetes_probs)
      WHERE CAST(label AS INT64) = 1 LIMIT 1
    ), 4) AS probability_of_diabetes,
    CASE
      WHEN (
        SELECT prob FROM UNNEST(predicted_diabetes_probs)
        WHERE CAST(label AS INT64) = 1 LIMIT 1
      ) < 0.30 THEN 'Low Risk'
      WHEN (
        SELECT prob FROM UNNEST(predicted_diabetes_probs)
        WHERE CAST(label AS INT64) = 1 LIMIT 1
      ) < 0.70 THEN 'Moderate Risk'
      ELSE 'High Risk'
    END AS risk_category
  FROM ML.PREDICT(
    MODEL `demo_diabetes.diabetes_model`,
    TABLE input
  )
);
'