import requests
import json
import time

def test_mlflow_serving():
    url = "http://localhost:1234/invocations"
    print(f"Testing MLflow Serving Endpoint at {url}")
    
    # Example payload with features (adapt to your dataset)
    # The cleaned_dataset.csv has ~17 columns for classification.
    # For testing, we send a dummy payload.
    payload = {
        "dataframe_split": {
            "columns": ["Price", "ID", "Prod. year", "Cylinders", "Airbags"],
            "data": [[15000, 1234, 2015, 4, 6], [20000, 5678, 2018, 6, 8]]
        }
    }
    
    try:
        response = requests.post(
            url, 
            json=payload, 
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            print("Success! Predictions:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Failed with status code: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("Connection error. Make sure 'mlflow models serve' is running on port 1234.")

if __name__ == "__main__":
    test_mlflow_serving()
