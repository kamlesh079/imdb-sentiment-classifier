from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# create FastAPI app
app = FastAPI()

# load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# request body structure
class TextInput(BaseModel):
    text: str

# home route
@app.get("/")
def home():
    return {"message": "IMDB Sentiment API is running"}

# prediction route
@app.post("/predict")
def predict(input: TextInput):
    # preprocess text
    text = input.text.lower()

    # convert text to vector
    vec = vectorizer.transform([text])

    # predict
    result = model.predict(vec)[0]

    # convert output to label
    sentiment = "Positive" if result == 1 else "Negative"

    return {"prediction": sentiment}