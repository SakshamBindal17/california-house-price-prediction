import gradio as gr
import joblib
import numpy as np
import os

# Load model and scaler from the 'models' subfolder
model = joblib.load(os.path.join("models", "tuned_gradient_boosting_model.pkl"))
scaler = joblib.load(os.path.join("models", "scaler.pkl"))

def predict_house_value(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return f"${prediction * 100000:.2f}"

inputs = [
    gr.Number(label="Median Income"),
    gr.Number(label="House Age"),
    gr.Number(label="Average Rooms"),
    gr.Number(label="Average Bedrooms"),
    gr.Number(label="Population"),
    gr.Number(label="Average Occupancy"),
    gr.Number(label="Latitude"),
    gr.Number(label="Longitude"),
]

output = gr.Textbox(label="Predicted Median House Value")

demo = gr.Interface(
    fn=predict_house_value,
    inputs=inputs,
    outputs=output,
    title="California House Price Predictor",
    description="Enter the features to predict the median house value (in USD)."
)

if __name__ == "__main__":
    demo.launch()
