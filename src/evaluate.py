import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error
from data_preprocessing import load_data, preprocess_data

def evaluate():
    # Path to the data
    data_path = "data/Student_Performance.csv"
    model_path = "models/model_v1.pkl"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train the model first.")
        return

    # Load data for evaluation
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Load the trained model
    model = joblib.load(model_path)

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f"Model Evaluation results:")
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

if __name__ == "__main__":
    evaluate()
