import requests

url = "http://localhost:8000/predict"
data = {
    "features": [2, 120, 70, 20, 79, 25.0, 0.5, 33]  # Example input
}

response = requests.post(url, json=data)

if response.ok:
    print("Prediction:", response.json()['prediction'])
else:
    print("Error:", response.text)