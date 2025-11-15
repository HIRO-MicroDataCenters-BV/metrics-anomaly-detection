from typing import Any, List, Tuple

import argparse
import os
import sys
import time
from io import StringIO

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from watchdog.events import DirModifiedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers.polling import PollingObserver


class ModelTrainer:
    model: RandomForestClassifier
    model_path: str

    def __init__(self, model_path: str):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_path = model_path

    def train(self, X_train: Any, X_test: Any, y_train: Any, y_test: Any) -> None:
        print("\n🔹 Training Random Forest model...")

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🔹 Accuracy: {accuracy:.4f}")

        # Print classification report
        print("🔹 Classification Report:")
        print(classification_report(y_test, y_pred))

        # Save the model
        self.save_model(self.model_path)

    def save_model(self, output_dir: str = "models") -> None:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "random_forest_model.joblib")
        joblib.dump(self.model, model_path)
        print(f"✔️ Model saved to {model_path}")


class DatasetHandler(FileSystemEventHandler):
    trainer: ModelTrainer
    last_modified: int

    def __init__(self, model_path: str):
        self.trainer = ModelTrainer(model_path)
        self.last_modified = 0

    def process_dataset(self, dataset_path: str) -> None:
        # Add a small delay to ensure the file is completely written
        time.sleep(1)

        try:
            print(f"📌 Processing dataset: {dataset_path}")
            # Read the dataset
            df = pd.read_csv(dataset_path, low_memory=False)

            # Remove rows with NaN values in the target column
            df = df.dropna(subset=["Abnormality class"])

            print("📌 Data shape:", df.shape)
            print("📌 Class distribution:")
            print(df["Abnormality class"].value_counts())

            # Drop non-numeric columns and unnecessary columns
            drop_columns = ["Unnamed: 0", "timestamp", "Microservice", "Experiment"]
            # Find all columns with '_deployed_at' suffix
            deployed_columns = [col for col in df.columns if col.endswith("_deployed_at")]
            drop_columns.extend(deployed_columns)

            # Prepare features and target
            X = df.drop(drop_columns + ["Abnormality class"], axis=1)
            y = df["Abnormality class"]

            # Convert all columns to numeric, replacing non-numeric values with NaN
            X = X.apply(pd.to_numeric, errors="coerce")

            # Fill NaN values with 0
            X = X.fillna(0)

            print("📌 Features shape after preprocessing:", X.shape)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"✔️ Training set size: {X_train.shape}")
            print(f"✔️ Test set size: {X_test.shape}")

            # Train the model
            self.trainer.train(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ Error processing dataset: {str(e)}")

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        src_path = str(event.src_path)
        if src_path.endswith(".csv"):
            current_time = int(time.time())
            # Prevent multiple processing of the same event
            if current_time - self.last_modified > 1:
                print(f"\n🔄 Dataset modified: {src_path}")
                self.process_dataset(src_path)
                self.last_modified = current_time


def run_ml(args: List[str]) -> None:
    is_success, csv_path_or_error, model_path = parse_args(args)
    if is_success:
        ml_main(csv_path_or_error, model_path)
    else:
        print(csv_path_or_error)


def parse_args(args: List[str]) -> Tuple[bool, str, str]:
    parser = argparse.ArgumentParser(
        description="Anomaly Detection",
        exit_on_error=False,
    )
    parser.add_argument(
        "--csv-path",
        dest="csv_path",
        action="store",
        help="Data input path",
        required=True,
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        action="store",
        help="Model output path",
        required=True,
    )

    try:
        ns: argparse.Namespace = parser.parse_args(args)
        return True, ns.csv_path, ns.model_path
    except argparse.ArgumentError as e:
        message = StringIO()
        parser.print_help(message)
        message.write(str(e))
        return False, message.getvalue(), ""


def ml_main(data_path: str, model_path: str) -> None:
    # Create the handler and observer
    handler = DatasetHandler(model_path)
    observer = PollingObserver(timeout=1.0)  # Poll every second
    print("🔧 Setting up file watcher with 1-second polling interval...")
    observer.schedule(handler, path=data_path, recursive=False)
    observer.start()

    print("👀 Monitoring for dataset changes...")

    # Initial processing if dataset exists
    if os.path.exists("final_dataset.csv"):
        print("📋 Found existing dataset, processing...")
        handler.process_dataset("final_dataset.csv")

    try:
        while True:
            time.sleep(1)
            # Debug: Check if file has changed
            if os.path.exists("final_dataset.csv"):
                mtime = os.path.getmtime("final_dataset.csv")
                print(f"🔍 Checking dataset... Last modified: {time.ctime(mtime)}")
    except KeyboardInterrupt:
        observer.stop()
        print("\n⚡ Stopping model training service...")
    observer.join()


if __name__ == "__main__":
    run_ml(sys.argv[1:])
