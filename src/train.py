from sklearn.linear_model import LinearRegression
import joblib
import os
from data_preprocessing import load_data, preprocess_data

def train():
    # Path to the data
    data_path = "data/Student_Performance.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please ensure the dataset is in the data directory.")
        return

    # Load and preprocess
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # Save the model
    joblib.dump(model, "models/model_v1.pkl")

    print("Model trained and saved successfully as models/model_v1.pkl")

if __name__ == "__main__":
    train()
