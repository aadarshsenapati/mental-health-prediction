# Mental Health Severity Prediction using Machine Learning

A comprehensive machine learning system for predicting depression and anxiety severity levels, developed as part of the Machine Learning Laboratory course at SRM University AP.

---

## Project Overview
This project implements a complete machine learning pipeline to classify mental health conditions into four severity categories: minimal, mild, moderate, and severe. It is built using psychological, demographic, and clinical features collected from undergraduate students.

The system includes multiple machine learning models, a full training pipeline, and a deployed web application for real-time predictions.

---

## Key Features
- Dual prediction for depression and anxiety severity
- Multiple machine learning algorithms: Decision Tree, Logistic Regression, K-Nearest Neighbors, Stacking Classifier
- End-to-end web application using Flask
- Deployment on PythonAnywhere
- Feature importance analysis and model comparison
- Based on validated PHQ-9 and GAD-7 assessment scales

---

## Objectives
- Develop accurate classification models for mental health severity prediction
- Perform comprehensive data preprocessing and feature engineering
- Compare individual models with ensemble learning
- Deploy trained models in a user-friendly web application
- Analyze key predictors using feature importance

---

## Dataset
**Source:** Undergraduate students at the University of Lahore, Pakistan  
**Total Samples:** 787 complete records  
**Total Features:** 19

### Feature Categories
- **Demographic:** Age, gender, BMI
- **Psychological:** PHQ‑9 score, GAD‑7 score, Epworth Sleepiness Scale
- **Clinical:** Treatment history, previous diagnosis, suicidal thoughts

**Target Variables:**  
- Depression severity (4 classes)  
- Anxiety severity (4 classes)

---

## Methodology

### 1. Data Preprocessing
- Missing value handling
- Outlier detection and treatment
- Label and one‑hot encoding
- Feature scaling
- 70‑15‑15 train–test–validation split

### 2. Machine Learning Models
- Decision Tree Classifier
- Logistic Regression
- K‑Nearest Neighbors
- Stacking Classifier (ensemble model)

### 3. Model Optimization
- Hyperparameter tuning using GridSearchCV
- Five‑fold stratified cross‑validation
- Evaluation metrics: Accuracy, Precision, Recall, F1‑Score

### 4. Web Application
- Frontend: HTML5, CSS3, JavaScript
- Backend: Flask
- Hosting: PythonAnywhere
- Outputs: Real‑time predictions, severity visualization, informative resources

---

## Results

### Model Performance Comparison
| Model | Depression Accuracy | Anxiety Accuracy |
|--------|----------------------|-------------------|
| Decision Tree | 58% | 53% |
| Logistic Regression | 65% | 63% |
| K‑Nearest Neighbors | 61% | 60% |
| Stacking Classifier | 71% | 68% |

### Key Insights
- The Stacking Classifier achieved the highest overall performance.
- Prior mental health history was the strongest predictor.
- Sleep disturbances showed significant correlation with PHQ‑9 and GAD‑7 scores.
- Most classification errors occurred between adjacent severity levels.

---

## Live Application
Access the deployed web application:  
**https://aadarshsenapati2005.pythonanywhere.com/**

---

## Installation and Usage

### Prerequisites
- Python 3.9+
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/aadarshsenapati/mental-health-prediction.git
cd mental-health-prediction
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Web Application Locally
```bash
python app.py
```
Navigate to http://localhost:5000 in your browser.

### Step 4 — Run the Machine Learning Training Pipeline
```bash
python train_models.py
```

---

## Project Structure
```
mental-health-prediction/
├── app.py                    # Flask application
├── train_models.py           # Model training script
├── requirements.txt          # Python dependencies
├── models/                   # Saved ML models
│   ├── depression_model.pkl
│   └── anxiety_model.pkl
├── static/                   # CSS, JS, images
├── templates/                # HTML templates
│   └── index.html
├── notebooks/                # Data analysis notebooks
├── data/                     # Dataset and processed files
├── README.md                 # Project documentation
└── Project_Report.pdf        # Detailed project report
```

---

## Dependencies
Key Python libraries:
- scikit‑learn
- pandas
- numpy
- flask
- matplotlib
- seaborn
- joblib

Manual installation:
```bash
pip install scikit-learn pandas numpy flask matplotlib seaborn joblib
```

---

## References
- Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). PHQ‑9 validation.
- Spitzer, R. L., et al. (2006). GAD‑7 assessment.
- Pedregosa, F., et al. (2011). Scikit‑learn: Machine Learning in Python.
- Zhou, Z.‑H. (2012). Ensemble Methods.

---

## Contributors
- Aadarsh Senapati (AP23110010458)
- Praveen Kumar (AP23110010460)

Faculty Guide  
Department of Computer Science and Engineering  
SRM University – AP, Amaravati, Andhra Pradesh

---

## License
This project is licensed under the MIT License.

---

## Disclaimer
This tool is intended solely for academic and screening purposes. It should not be used as a substitute for professional medical diagnosis or treatment.

---

## Future Work
- Integration of deep learning models
- Mobile application development
- Longitudinal analysis for mental health trends
- Multimodal data integration (text, voice)
- Cross‑cultural dataset validation

---

## Contact
**Aadarsh Senapati**  
GitHub: [https://github.com/aadarshsenapati](https://github.com/aadarshsenapati)
**Praveen Kumar**  
GitHub: [https://github.com/KommanaboyinaPraveenKumar/](https://github.com/KommanaboyinaPraveenKumar/)
Repository:  
https://github.com/aadarshsenapati/mental-health-prediction

