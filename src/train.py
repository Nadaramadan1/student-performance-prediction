from sklearn.linear_model import LinearRegression
import joblib
import os
import mlflow
import mlflow.sklearn
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

    # Initialize the model
    model = LinearRegression()

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")

        # Train the model
        model.fit(X_train, y_train)

        # Calculate metrics
        r2 = model.score(X_test, y_test)
        mlflow.log_metric("r2_score", r2)

        # Log the model to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="StudentPerformanceModel"
        )


        # Create models directory if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")

        # Save the model locally
        joblib.dump(model, "models/model_v1.pkl")

        print("Model trained and logged successfully.")

if __name__ == "__main__":
    train()
