from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ------------ LOAD MODELS ------------
depression_model = joblib.load("model/depression_model.pkl")
anxiety_model = joblib.load("model/anxiety_model.pkl")

scaler_age = joblib.load("model/scaler_age.pkl")
scaler_bmi = joblib.load("model/scaler_bmi.pkl")

# ------------ REVERSE MAPS ------------
dep_map = {
    0: "None-minimal",
    1: "Mild",
    2: "Moderate",
    3: "Moderately severe",
    4: "Severe"
}

anx_map = {
    0: "None-minimal",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # User inputs
    age = scaler_age.transform([[float(request.form["age"])]])[0][0]
    bmi = scaler_bmi.transform([[float(request.form["bmi"])]])[0][0]

    gender = int(request.form["gender"])
    suicidal = int(request.form["suicidal"])
    sleepiness = int(request.form["sleepiness"])
    anxiousness = int(request.form["anxiousness"])
    depressiveness = int(request.form["depressiveness"])
    dep_diag = int(request.form["depression_diag"])
    dep_treat = int(request.form["depression_treat"])

    anx_diag = int(request.form["anxiety_diag"])
    anx_treat = int(request.form["anxiety_treat"])
    school_year = int(request.form["school_year"])

    # ------------------------------------------------------
    #   Depression model → uses NO BMI and requires
    #   'anxiety_severity' and 'depressiveness'
    # ------------------------------------------------------
    dep_row = pd.DataFrame([{
        "school_year": school_year,
        "age": age,
        "gender": gender,
        "depressiveness": depressiveness,
        "suicidal": suicidal,
        "depression_diagnosis": dep_diag,
        "depression_treatment": dep_treat,
        "anxiousness": anxiousness,
        "anxiety_diagnosis": anx_diag,
        "anxiety_treatment": anx_treat,
        "sleepiness": sleepiness
    }])

    dep_pred_val = depression_model.predict(dep_row)[0]
    dep_pred = dep_map[dep_pred_val]

    # ------------------------------------------------------
    #  Anxiety model → needs BMI AND depression_severity
    #  (we will feed it depression_model prediction)
    # ------------------------------------------------------
    anx_row = pd.DataFrame([{
        "school_year": school_year,
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "depression_severity": dep_pred_val,   # using depression prediction as input
        "depressiveness": depressiveness,
        
        "suicidal": suicidal,
        "depression_diagnosis": dep_diag,
        "depression_treatment": dep_treat,
        "anxiousness": anxiousness,
        "anxiety_diagnosis": anx_diag,
        "anxiety_treatment": anx_treat,
        "sleepiness": sleepiness
    }])

    anx_pred = anx_map[anxiety_model.predict(anx_row)[0]]

    return render_template("result.html", dep=dep_pred, anx=anx_pred)


if __name__ == "__main__":
    app.run(debug=True)
