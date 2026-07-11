import numpy as np
import os
import json
from supabase import create_client
import tensorflow as tf
from tensorflow import keras


# ─── Config ───────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_BUCKET = "models"

# Load epochs from training_config
def get_epochs():
    try:
        response = supabase.table("training_config") \
            .select("epochs") \
            .eq("id", 1) \
            .single() \
            .execute()
        return response.data.get("epochs", 100)
    except:
        return 100


# ─── Update status ────────────────────────────────────────────────────────────
def update_status(status, progress=0, message=''):
    try:
        supabase.table("training_status").upsert({
            "id": 1,
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": "now()"
        }).execute()
        print(f"Status: {status} | {message}")
    except Exception as e:
        print(f"Status update error: {e}")

# ─── Get num_draws from training_config ───────────────────────────────────────
def get_num_draws():
    try:
        config = supabase.table("training_config") \
            .select("num_draws") \
            .eq("id", 1) \
            .single() \
            .execute()
        num = config.data.get("num_draws", 1000) if config.data else 1000
        print(f"num_draws from config: {num}")
        return num
    except Exception as e:
        print(f"Config error: {e}")
        return 1000

# ─── Load draws ───────────────────────────────────────────────────────────────
def load_draws():
    num_draws = get_num_draws()
    update_status("loading", 20, message=f"Loading {num_draws} draws from Supabase...")
    # Load LATEST draws first then reverse for LSTM
    response = supabase.table("toto_results") \
        .select("draw_no, draw_date, winning_no, additional_no") \
        .order("draw_no", desc=True) \
        .limit(num_draws) \
        .range(0, num_draws - 1) \
        .execute()
    data = list(reversed(response.data))
    print(f"Loaded {len(data)} draws")
    print(f"From: {data[0]['draw_date']} to {data[-1]['draw_date']}")
    return data
   
# ─── Convert to multihot ──────────────────────────────────────────────────────
def draws_to_multihot(draws):
    X = []
    for row in draws:
        v = np.zeros(49, dtype=np.float32)
        nums = [int(n.strip()) for n in str(row["winning_no"]).split(",")]
        for n in nums:
            v[n - 1] = 1.0
        if row["additional_no"]:
            v[int(row["additional_no"]) - 1] = 1.0
        X.append(v)
    return np.array(X)

# ─── Predict ──────────────────────────────────────────────────────────────────
def predict_and_save(model, draws):
    update_status("predicting", 70, message="Running predictions...")
    print("Running predictions...")

    data_X = draws_to_multihot(draws)
    
    # Read window size from training_config (set by phone app)
    try:
        config = supabase.table("training_config") \
            .select("window_size") \
            .eq("id", 1) \
            .single() \
            .execute()
        window = config.data.get("window_size", 15) if config.data else 15
    except Exception as e:
        print(f"Config error: {e}")
        window = 15
    print(f"Using window size: {window}")
        
    last_seq = data_X[-window:].reshape((1, window, 49)).astype(np.float32)

    mc_samples = 100
    probs_accum = np.zeros(49, dtype=np.float64)
    for i in range(mc_samples):
        pred = model(last_seq, training=True).numpy().reshape(-1)
        probs_accum += pred
    avg_probs = probs_accum / mc_samples

    # Pure LSTM — just sort by probability, no bias
    all_sorted = sorted(
        [(i + 1, float(avg_probs[i])) for i in range(49)],
        key=lambda x: x[1], reverse=True
    )
    
    # Take top 7 purely by probability
    top7 = all_sorted[:7]
        
    numbers = [x[0] for x in top7]
    probabilities = [round(x[1], 4) for x in top7]

    latest_draw = draws[-1]
    print(f"Latest draw used: #{latest_draw['draw_no']} — {latest_draw['draw_date']}")

    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc).isoformat()

    # Get actual best epoch from last training run
    try:
        meta = supabase.table("model_meta").select("best_epoch").eq("id", 1).single().execute()
        actual_epochs = meta.data.get("best_epoch", get_epochs()) if meta.data else get_epochs()
    except Exception as e:
        print(f"Could not load best_epoch: {e}")
        actual_epochs = get_epochs()

    supabase.table("predictions").upsert({
        "id": 1,
        "predicted_at": now_utc,
        "numbers": json.dumps(numbers),
        "probabilities": json.dumps(probabilities),
        "draw_date": latest_draw['draw_date'],
        "draw_no": latest_draw['draw_no'],
        "window_size": window,
        "epochs": actual_epochs,
        "total_draws": len(draws)
    }).execute()    
    
    print(f"Predicted: {numbers}")
    return numbers, probabilities

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TOTO Predictor Started ===")
    update_status("starting", 0, message="Starting prediction...")

    try:
        # Download model from Supabase
        update_status("loading", 10, message="Downloading model from Supabase...")
        model_data = supabase.storage.from_(MODEL_BUCKET).download('lstm_model.h5')
        with open('lstm_model.h5', 'wb') as f:
            f.write(model_data)
        print("Model downloaded!")

        # Load model
        update_status("loading", 40, message="Loading model...")
        model = keras.models.load_model('lstm_model.h5')
        print("Model loaded!")

        # Load draws
        draws = load_draws()

        # Predict
        numbers, probs = predict_and_save(model, draws)

        # Done
        update_status("complete", 100, message=f"✅ Predicted: {numbers}")
        print("=== Prediction Done! ===")

    except Exception as e:
        print(f"Error: {e}")
        update_status("error", 0, message=f"Error: {str(e)}")
