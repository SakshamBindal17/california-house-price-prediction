# California House Price Prediction

This project is a machine learning application that predicts median house prices in California using the Gradient Boosting algorithm. The model is trained on the California Housing dataset and deployed as an interactive web app using Gradio.

---

## Project Overview

- **Dataset:** California Housing dataset from scikit-learn
- **Model:** Gradient Boosting Regressor with hyperparameter tuning
- **Deployment:** Gradio web app hosted on Hugging Face Spaces

---

## Features Used

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

---

## Repository Structure
```
california-house-price-prediction/
│
├── app.py # Gradio web app code
├── models/ # Folder containing saved model and scaler
│ ├── tuned_gradient_boosting_model.pkl
│ └── scaler.pkl
├── src/ # Scripts for training and prediction
│ ├── train_model.py
│ └── predict.py
├── notebooks/ # Google Colab notebooks
│ └── california_house_price_prediction.ipynb
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore file
```

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```
git clone https://github.com/yourusername/california-house-price-prediction.git
cd california-house-price-prediction
```

2. **Install dependencies:**
```
pip install -r requirements.txt
```

---

## Running the App Locally

1. Ensure the `models/` folder contains the trained model and scaler files.
2. Run the Gradio app:
```
python app.py
```
3. Open the provided local URL in your browser to use the app.

---

## Usage

- Enter the house features in the input fields.
- Click the **Predict** button.
- View the predicted median house value.

---

## Training the Model

If you want to retrain the model:

1. Navigate to the `src/` folder.
2. Run the training script:
```
python train_model.py
```
This will train the Gradient Boosting model with the best parameters and save the model and scaler in the `models/` folder.

---

## Deployment

The app is deployed on Hugging Face Spaces using Gradio.  
**Try it online:**  
[California House Price Predictor on Hugging Face Spaces](https://huggingface.co/spaces/sakshambindal/california-house-price-prediction)

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## License

This project is licensed under the MIT License.
