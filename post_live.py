"""
This is the code for posting from deployed app.
Author: George Christodoulou
Date: 24/06/23
"""
import requests
import json
from settings import settings


url = "https://project-udacity.onrender.com/inference"

# Get the sample
sample = settings["sample"]

data = json.dumps(sample)

# Post to API
response = requests.post(url, data=data)

# Display Response
print("Status", response.status_code)
print("Response:")
print(response.json())
