import joblib

model = joblib.load('../models/tuned_gradient_boosting_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

def predict_house_value(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]
