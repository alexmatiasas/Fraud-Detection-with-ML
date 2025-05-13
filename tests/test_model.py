import requests
import json
import os

def test_api_returns_200_and_probability():
    # Ruta al archivo JSON de ejemplo
    test_file = os.path.join("notebooks", "request_example.json")
    
    with open(test_file, "r") as f:
        input_data = json.load(f)

    response = requests.post("http://localhost:8000/predict", json=input_data)

    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert isinstance(data["fraud_probability"], float)
    assert 0.0 <= data["fraud_probability"] <= 1.0