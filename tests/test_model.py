import json
import os

import requests


def test_api_returns_200_and_probability():
    # Ruta al archivo JSON de ejemplo
    test_file = os.path.join("notebooks", "request_example.json")

    with open(test_file, "r") as f:
        input_data = json.load(f)

    response = requests.post(
        "http://localhost:8000/predict", json=input_data, timeout=5
    )

    assert response.status_code == 200  # nosec
    data = response.json()
    assert "fraud_probability" in data  # nosec
    assert isinstance(data["fraud_probability"], float)  # nosec
    assert 0.0 <= data["fraud_probability"] <= 1.0  # nosec
