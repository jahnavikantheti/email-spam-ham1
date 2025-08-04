from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Email Spam-Ham Classifier API")

# -------------------- CORS --------------------
origins = [
    "http://127.0.0.1:5500",
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

# -------------------- Serve Frontend --------------------
# Mount frontend folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def get_index():
    return FileResponse("frontend/index.html")

# -------------------- Model Load --------------------
MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    raise FileNotFoundError("Model/vectorizer file not found.")

# -------------------- Schema --------------------
class EmailRequest(BaseModel):
    text: str

# -------------------- Prediction Endpoint (GET + POST) --------------------
@app.api_route("/predict", methods=["GET", "POST"])
async def predict_email(request: Request):
    try:
        if request.method == "GET":
            text = request.query_params.get("text")
        else:
            data = await request.json()
            text = data.get("text")

        if not text:
            raise HTTPException(status_code=400, detail="Text input is required")

        vect_text = vectorizer.transform([text])
        prediction = model.predict(vect_text)[0]
        label = "spam" if prediction == 1 else "ham"
        return {"prediction": label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Handle OPTIONS (for CORS Preflight) --------------------
@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str = ""):
    return JSONResponse(status_code=200, content={})
