def validate_patient_data(patient):
    warnings = []

    if patient["age"] < 1 or patient["age"] > 120:
        warnings.append("Age value is outside the valid human range.")

    if patient["trestbps"] <= 0:
        warnings.append("Resting blood pressure must be greater than 0.")

    if patient["chol"] <= 0:
        warnings.append("Cholesterol must be greater than 0.")

    if patient["thalach"] <= 0:
        warnings.append("Maximum heart rate must be greater than 0.")

    if patient["oldpeak"] < 0:
        warnings.append("Oldpeak cannot be negative.")

    if patient["sex"] not in [0, 1]:
        warnings.append("Sex must be 0 for female or 1 for male.")

    if patient["cp"] not in [1, 2, 3, 4]:
        warnings.append("Chest pain type must be between 1 and 4.")

    if patient["fbs"] not in [0, 1]:
        warnings.append("Fasting blood sugar must be 0 or 1.")

    if patient["restecg"] not in [0, 1, 2]:
        warnings.append("Resting ECG value must be 0, 1, or 2.")

    if patient["exang"] not in [0, 1]:
        warnings.append("Exercise-induced angina must be 0 or 1.")

    if patient["slope"] not in [1, 2, 3]:
        warnings.append("Slope value must be 1, 2, or 3.")

    if patient["ca"] not in [0, 1, 2, 3]:
        warnings.append("CA value must be between 0 and 3.")

    if patient["thal"] not in [3, 6, 7]:
        warnings.append("Thal value must be 3, 6, or 7.")

    if patient["trestbps"] > 200:
        warnings.append("Resting blood pressure is unusually high.")

    if patient["chol"] > 500:
        warnings.append("Cholesterol value is unusually high.")

    if patient["thalach"] > 220:
        warnings.append("Maximum heart rate is unusually high.")

    return warnings