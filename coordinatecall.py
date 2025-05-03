import requests
import json

city    = "Miami"
state   = "FL"
country = "US"
limit   = 5
api_key = "apikey"

url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},{state},{country}&limit={limit}&appid={api_key}"

# 1. Send the request
response = requests.get(url)

# 2. Check for success
if response.status_code == 200:
    # 3. Parse JSON
    data = response.json()
    # 4a. Print raw Python object
    print(data)
    # 4b. Or pretty‐print with indentation
    print(json.dumps(data, indent=2))
else:
    print(f"Error fetching geocode data: HTTP {response.status_code}")
    print(response.text)
