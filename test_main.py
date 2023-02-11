from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

person_1 = {
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

person_2 = {
  "age": 35,
  "workclass": "State-gov",
  "fnlgt": 83311,
  "education": "Bachelors",
  "education_num": 18,
  "marital_status": "Married-civ-spouse",
  "occupation": "Adm-clerical",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital_gain": 5000,
  "capital_loss": 0,
  "hours_per_week": 50,
  "native_country": "United-States"
}

# Test for greeting page
def test_greeting():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"greeting": "Welcome to the prediction of salary from census data!"}

# Test for a user
def test_get_user_1():
    response = client.post('/predict', json = person_1)
    assert response.status_code == 200
    assert response.json()['the result'] == "Less 50k"

# Test for a user
def test_get_user_2():
    response = client.post('/predict', json = person_2)
    assert response.status_code == 200
    assert response.json()['the result'] == "Over 50k"

test_greeting()
test_get_user_1()
test_get_user_2()