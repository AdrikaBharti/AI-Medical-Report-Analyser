# Limitations

## Data sources
All five models are trained on public research datasets (Pima Diabetes, UCI Heart Disease multi-center data, UCI Chronic Kidney Disease, the Indian Liver Patient Dataset, and a hematological dataset for anemia), not on data collected from this application's own users. Their patient populations, lab equipment, reference ranges, and demographics may not match the people who use this app, so reported performance should be read as an upper bound rather than a guarantee of real-world accuracy.

## No external validation
Every result in this project comes from cross-validation on the training dataset itself. No model has been tested on an independent, held-out dataset collected separately, and none has been evaluated against real lab reports processed through the OCR/extraction pipeline. Before any clinical or public-facing claim, the models need validation on data they have never seen during development.

## Kidney: small sample size
The Kidney model is trained on only 400 records (250 CKD, 150 not-CKD) from a single hospital in India. Cross-validated ROC-AUC is at or near 1.000 for several models, which reflects how separable this particular dataset is (creatinine, hemoglobin and related markers differ sharply between classes) rather than a claim that CKD detection is a solved problem. With so few records, performance estimates carry wide uncertainty, and the model has not been shown to generalize beyond this cohort.

## Liver: a hard prediction problem
The Liver model is the weakest of the five, with cross-validated ROC-AUC around 0.75 and accuracy around 0.72. This is consistent with prior published work on the Indian Liver Patient Dataset, which is a known difficult dataset. The chosen decision threshold favors recall (catching more true cases) at the cost of precision, so a meaningful share of flagged users will not actually have liver disease. Liver predictions should be presented with visibly lower confidence than the other four diseases, and never as a stand-alone diagnostic signal.

## Heart: dependent on manually entered clinical data
Restricting the Heart model to values a lab report can supply (cholesterol, fasting blood sugar) drops its cross-validated ROC-AUC from about 0.88 to about 0.74. Most of its predictive power comes from clinical fields a report does not contain: chest pain type, resting ECG results, maximum heart rate, and exercise-induced angina. The app therefore cannot predict heart disease risk from a report alone; it depends on a short manual questionnaire, and its accuracy is bounded by how honestly and precisely a user answers that questionnaire.

## Diabetes: known data quality issue, corrected
The supplied Pima Diabetes file was found to contain target leakage: missing Insulin, Skin Thickness and Blood Pressure values had been filled in with per-class (diagnosis-conditional) medians rather than a single overall median, which let a model infer the diagnosis directly from the imputed value. This was identified and corrected before any reported result (the affected values were reset to missing and re-imputed inside each cross-validation fold, so no test information leaks into training). Post-correction Diabetes performance (ROC-AUC about 0.80-0.84) is consistent with the published range for this dataset. This case is included here as a reminder that any additional data brought into the project should be checked for similar imputation artifacts before use.

## Feature availability at inference time
Kidney, Liver and Anemia predictions can be made from lab report values alone. Diabetes benefits from a few additional manually entered fields (BMI, blood pressure, pregnancy count). Heart requires a short clinical questionnaire, as noted above. Each model declares a minimum set of required fields; when a report does not contain them, the application is designed to say plainly that there is not enough data, rather than to produce a prediction from an incomplete or heavily imputed feature set.

## Decision thresholds and probabilities
Classification thresholds were chosen to prioritize recall (minimizing missed positive cases) for a screening context, which means the reported precision is lower than it would be at a standard 0.5 cutoff. Displayed probabilities are calibrated (isotonic regression, 5-fold) but calibration quality has not itself been externally validated and should be re-checked if the underlying data or population changes.

## Not a diagnostic tool
None of these models, individually or combined, constitute a medical diagnosis. They are intended as a screening aid to help a user decide whether to seek professional medical advice. All outputs are shown with an explicit disclaimer to this effect, and the application does not recommend treatment, medication, or dosage of any kind.
