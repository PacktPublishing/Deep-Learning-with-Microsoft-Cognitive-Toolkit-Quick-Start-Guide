import requests
import json

service_url = "<service-url>"
data = [[1.4, 0.2, 4.9, 3.0]]

response = requests.post(service_url, json=data)

print(response.json())
