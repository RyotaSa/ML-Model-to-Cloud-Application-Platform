import requests

person = {
  "age": 35,
  "workclass": "State-gov",
  "fnlgt": 83311,
  "education": "Bachelors",
  "education_num": 10,
  "marital_status": "Married-civ-spouse",
  "occupation": "Adm-clerical",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2000,
  "capital_loss": 0,
  "hours_per_week": 30,
  "native_country": "United-States"
}


# Test for request page
request = requests.post('https://render-deployment-ml-cloud.onrender.com/predict',json=person)

print(request.json())

