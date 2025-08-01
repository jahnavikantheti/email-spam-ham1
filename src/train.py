import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from src.preprocessing import preprocess_data

data_path = "data/spam_ham_dataset.csv"
model_dir = "model"

def train_model():
    # Preprocess data
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data_path)

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Start MLflow experiment tracking
    mlflow.set_experiment("spam-detection")

    with mlflow.start_run():
        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate model
        acc = model.score(X_test, y_test)

        # Log to MLflow
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("vectorizer", "TFIDF")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(vectorizer, "vectorizer")

        print(f"Model logged with accuracy: {acc:.4f}")

        # Save model & vectorizer using joblib
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))
        print("Model and vectorizer saved with joblib.")

if __name__ == "__main__":
    train_model()
