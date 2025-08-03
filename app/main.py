from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# Initialize FastAPI app
app = FastAPI(title="Email Spam-Ham Classifier API")

# -------------------- CORS Configuration --------------------
# Only domains, no endpoint paths
origins = [
    "http://127.0.0.1:5500",
    "*"
    "http://localhost:5500",
    "https://emailspamham.azurewebsites.net",
    "https://emailspamham-1-hea6f7d6hecpa4fv.canadacentral-01.azurewebsites.net",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        
    allow_credentials=True,
    allow_methods=["*"],          
    allow_headers=["*"],          
)

# -------------------- Load Model and Vectorizer --------------------
MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    raise FileNotFoundError(
        "Model or vectorizer file not found. Please train the model first and include .pkl files in your repo."
    )

# -------------------- Root Endpoint --------------------
@app.get("/")
def home():
    return {"message": "✅ Email Spam-Ham API is running! Use POST /predict to classify emails."}

# -------------------- Input Schema --------------------
class EmailRequest(BaseModel):
    text: str

# -------------------- Prediction Route --------------------
@app.post("/predict")
def predict_email(req: EmailRequest):
    try:
        vect_text = vectorizer.transform([req.text])
        prediction = model.predict(vect_text)[0]
        label = "spam" if prediction == 1 else "ham"
        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
