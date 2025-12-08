from fastapi import FastAPI, HTTPException
import pandas as pd
import os, traceback, joblib
from enum import Enum, IntEnum
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI application
app = FastAPI()

# --- CORS CONFIGURATION ---
# This middleware allows the API to be accessed from any frontend (like your Streamlit app).
# "allow_origins=['*']" means it accepts requests from any domain/URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATH CONFIGURATION ---
# Dynamic path resolution ensures the code works both locally and inside a Docker container.
# current_dir: Gets the folder where this script (api_file.py) resides.
current_dir = os.path.dirname(os.path.abspath(__file__))

# models_dir: Navigates one level up ('..') to find the 'models' directory.
models_dir = os.path.join(current_dir, '..', 'models')

# Define paths for the trained model and the label encoder
MODEL_PATH = os.path.join(models_dir, 'optimized_xgb_pipeline.pkl')
LABEL_ENCODER_PATH = os.path.join(models_dir, 'label_encoder.pkl')

# Define the exact list of features expected by the model (order matters!)
CORE_FEATURES = ['day_of_week', 'hour', 'department', 'surface_condition', 'road_category', 'speed_limit']

print(f"🔄 Loading model from: {MODEL_PATH}")
model, label_encoder = None, None

# --- MODEL LOADING ---
# Try to load the trained XGBoost model and LabelEncoder from disk at startup.
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")

    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print("✅ LabelEncoder loaded successfully")
    else:
        print("⚠️ LabelEncoder not found (using default mapping)")
except Exception as e:
    # If loading fails, the API will start but predictions will fail.
    print(f"❌ Error loading files: {e}")

# --- ENUMS (DATA VALIDATION) ---
# These classes strictly define the allowed input values for the API.
# FastAPI uses them to generate the automatic documentation (Swagger UI) and validate requests.

class Department(str, Enum):
    """Restricts input to specific French departments (Ile-de-France region)."""
    paris = "75 (Paris)"
    seine_et_marne = "77 (Seine-et-Marne)"
    yvelines = "78 (Yvelines)"
    essonne = "91 (Essonne)"
    hauts_de_seine = "92 (Hauts-de-Seine)"
    seine_saint_denis = "93 (Seine-Saint-Denis)"
    val_de_marne = "94 (Val-de-Marne)"
    val_d_oise = "95 (Val-d'Oise)"

class RoadCategory(str, Enum):
    """Classifies the type of road involved in the accident."""
    major = "Major Roads"
    secondary = "Secondary Roads"
    local = "Local & Access Roads"
    off_network = "Other / Off-Network"

class SurfaceCondition(str, Enum):
    """Describes the state of the road surface."""
    normal = "Normal"
    wet = "Wet / Slippery"

class DayOfWeek(str, Enum):
    """Standard days of the week."""
    mon = "Monday"; tue = "Tuesday"; wed = "Wednesday"; thu = "Thursday"
    fri = "Friday"; sat = "Saturday"; sun = "Sunday"

class SpeedLimit(int, Enum):
    """Restricts speed limits to standard French road limits (10km/h to 130km/h)."""
    km_10=10; km_20=20; km_30=30; km_40=40; km_50=50; km_60=60
    km_70=70; km_80=80; km_90=90; km_100=100; km_110=110; km_130=130

class Hour(IntEnum):
    """Restricts hours to integers between 0 and 23."""
    h00=0; h01=1; h02=2; h03=3; h04=4; h05=5; h06=6; h07=7; h08=8
    h09=9; h10=10; h11=11; h12=12; h13=13; h14=14; h15=15; h16=16
    h17=17; h18=18; h19=19; h20=20; h21=21; h22=22; h23=23

# --- DATA MAPPING ---
# DEPT_MAPPER: Cleans the department string.
# The UI sends "75 (Paris)", but the model expects just "Paris".
DEPT_MAPPER = {
    "75 (Paris)": "Paris", "77 (Seine-et-Marne)": "Seine-et-Marne",
    "78 (Yvelines)": "Yvelines", "91 (Essonne)": "Essonne",
    "92 (Hauts-de-Seine)": "Hauts-de-Seine", "93 (Seine-Saint-Denis)": "Seine-Saint-Denis",
    "94 (Val-de-Marne)": "Val-de-Marne", "95 (Val-d'Oise)": "Val-d'Oise"
}

# SEVERITY_MAP: Decodes the numerical output of the model into human-readable text.
SEVERITY_MAP = {
    2: "Death",
    3: "Hospitalized",
    4: "Slightly injured"
}

# --- ENDPOINTS ---

@app.get("/")
def root():
    """Health check endpoint to verify if the API is running."""
    return {"message": "SafeRoute AI API is running!"}

@app.post("/predict")
def predict(
    department: Department, day_of_week: DayOfWeek, hour: Hour,
    road_category: RoadCategory, speed_limit: SpeedLimit, surface_condition: SurfaceCondition
):
    """
    Main prediction endpoint.
    Receives accident details and returns the predicted severity and probabilities.
    """

    # 1. Safety check: Ensure model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 2. Preprocessing: Clean the department string using the mapper
    raw_dept = department.value
    clean_dept = DEPT_MAPPER.get(raw_dept, raw_dept)

    # 3. Data Formatting: Create a dictionary with the exact structure required by the model
    input_data = {
        "day_of_week": day_of_week.value,
        "hour": int(hour.value),
        "department": str(clean_dept),
        "surface_condition": surface_condition.value,
        "road_category": road_category.value,
        "speed_limit": int(speed_limit.value)
    }

    # 4. Convert to DataFrame (The model expects a Pandas DataFrame, not a dict)
    df = pd.DataFrame([input_data])
    df = df[CORE_FEATURES] # Ensure columns are in the correct order

    # Explicit type casting to prevent Scikit-Learn errors
    df["hour"] = df["hour"].astype(int)
    df["speed_limit"] = df["speed_limit"].astype(int)

    try:
        # 5. Prediction
        pred = int(model.predict(df)[0]) # Get the predicted class (e.g., 2, 3, or 4)
        prob_array = model.predict_proba(df)[0] # Get probabilities for all classes

        # 6. Post-processing: Map probabilities to human-readable labels
        mapped_probs = {}
        for idx, prob in enumerate(prob_array):
            # Decode the class index back to the original label (if Encoder exists)
            if label_encoder is not None:
                original_class_val = label_encoder.inverse_transform([idx])[0]
            else:
                original_class_val = idx

            # Get text description (e.g., "Death")
            severity_text = SEVERITY_MAP.get(int(original_class_val), f"Class {original_class_val}")
            mapped_probs[severity_text] = f"{prob*100:.1f}%"

        # 7. Decode the final prediction
        if label_encoder is not None:
            pred_original_val = label_encoder.inverse_transform([pred])[0]
        else:
            pred_original_val = pred

        severity_text = SEVERITY_MAP.get(int(pred_original_val), "Unknown")

        # 8. Return JSON response
        return {
            "prediction_class": pred,
            "severity_label": str(pred_original_val),
            "severity_text": severity_text,
            "probabilities": mapped_probs
        }

    except Exception as e:
        # Log the full error trace for debugging and return a 400 Bad Request
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Model execution error: {str(e)}")

# --- SERVER STARTUP ---
if __name__ == "__main__":
    import uvicorn
    # Gets the PORT from environment variables (defaults to 8080).
    # This is crucial for deployment on Cloud Run.
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
