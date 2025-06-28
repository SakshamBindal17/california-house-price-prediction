# California House Price Prediction

This project is a machine learning application that predicts median house prices in California using the Gradient Boosting algorithm. The model is trained on the California Housing dataset and deployed as an interactive web app using Gradio.

---

## Project Overview

- **Dataset:** California Housing dataset from scikit-learn
- **Model:** Gradient Boosting Regressor with hyperparameter tuning
- **Deployment:** Gradio web app hosted on Hugging Face Spaces

---

## Dataset Information

The project uses the **California Housing dataset** from the scikit-learn library. This dataset contains information collected from the 1990 U.S. Census, specifically for California housing.

### Dataset Source
- The dataset is available directly via scikit-learn's `fetch_california_housing` function.
- Official scikit-learn documentation: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

### Dataset Features (Columns)
The dataset includes the following 8 features:

1. **MedInc**: Median income in block group
2. **HouseAge**: Median house age in block group
3. **AveRooms**: Average number of rooms per household
4. **AveBedrms**: Average number of bedrooms per household
5. **Population**: Block group population
6. **AveOccup**: Average number of household members
7. **Latitude**: Block group latitude
8. **Longitude**: Block group longitude

### Target Variable
- **MedHouseVal**: Median house value for California districts (in units of 100,000s of dollars)

### Dataset Size
- Number of rows (samples): 20,640
- Number of columns: 8 (features) + 1 (target)

### How to Download
You can load the dataset in Python using scikit-learn as follows:
```
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame
```
This will download the dataset and load it into a pandas DataFrame named `df`.

---

## Repository Structure
```
california-house-price-prediction/
│
├── app.py                                   # Gradio web app code
├── california_housing_dataset.csv           # optional dataset to view the data
├── models/                                  # Folder containing saved model and scaler
│ ├── tuned_gradient_boosting_model.pkl
│ └── scaler.pkl
├── src/                                     # Scripts for training and prediction
│ ├── train_model.py
│ └── predict.py
├── california_house_price_prediction.ipynb  # Google Colab notebook
├── requirements.txt                         # Python dependencies
├── README.md
└── .gitignore
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
