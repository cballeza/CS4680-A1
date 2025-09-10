# CS4680-A1  
Carol Balleza  

---

## Problem Identification  
**Goal:** Predict which diabetes class a patient belongs to and predict a patient's BMI based on lab results and demographic features.  

- **Target:** Class and BMI  
- **Features:** Gender, Age, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL  

---

## Model Training  
- The dataset was split into **80% training** and **20% testing** using `train_test_split` with `random_state=42`.  
- **Models used:**  
  - **Classification:** `LogisticRegression(max_iter=5000)` → trained on features, target = Class  
  - **Regression:** `LinearRegression()` → trained on same features as classification model, target = BMI  

---

## Model Evaluation  

### Classification Results  
- Accuracy (overall correctness): **0.89 (89%)**  
- Precision (how many predicted positives are real (weighted)): **0.89**  
- Recall (how many real positives are caught by model (weighted)): **0.89**  
- F1 Score (balance of precision and recall (weighted)): **0.89**  

The model predicts the diabetes class with high accuracy. The model struggles with class 1 due to fewer samples in the class, but performs well on classes 0 and 2.  

### Regression Results  
- RMSE (typical size of errors): **4.15**  
- MAE (average size of errors): **3.26**  
- R² (how much variance is explained by model): **0.37**  

The model shows there is 37% variance in BMI.  

---

## Comparison and Suitability  

- **Classification:** Suitable for this dataset due to high accuracy and reliability in predicting patient diabetes class.  
- **Regression:** Less suitable since there is moderate performance. BMI could depend on other unmeasured factors not found in the dataset, which can limit predictions.  

---

## Conclusion  
The classification model is best suited for this dataset since it gives a strong performance. The regression model is not as suited since it demonstrates that BMI cannot be predicted accurately with the features from the dataset.  
