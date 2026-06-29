import numpy as np
import os
import tempfile
import time
import requests
from bs4 import BeautifulSoup
from supabase import create_client
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import struct

# ─── Config ───────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_BUCKET = "models"
SEED = 42

# ─── Load settings from Supabase ─────────────────────────────────────────────
def load_settings():
    try:
        response = supabase.table("training_config") \
            .select("*") \
            .eq("id", 1) \
            .single() \
            .execute()
        if response.data:
            return response.data
    except Exception as e:
        print(f"Settings load error: {e}")
    # Default settings
    return {
        "epochs": 100,
        "batch_size": 64,
        "train_ratio": 0.85,
        "window_size": 15,
        "num_draws": 1000
    }

# ─── Update training status in Supabase ───────────────────────────────────────
def update_status(status, progress=0, epoch=0, total=0, loss=0, val_loss=0, message=''):
    try:
        supabase.table("training_status").upsert({
            "id": 1,
            "status": status,
            "progress": progress,
            "current_epoch": epoch,
            "total_epochs": total,
            "loss": float(loss),
            "val_loss": float(val_loss),
            "message": message,
            "updated_at": "now()"
        }).execute()
        print(f"Status: {status} | {message}")
    except Exception as e:
        print(f"Status update error: {e}")

# ─── Save prediction to Supabase ──────────────────────────────────────────────
def save_prediction(numbers, probabilities, draw_date, trained_epochs):
    try:
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc).isoformat()
        supabase.table("predictions").upsert({
            "id": 1,
            "predicted_at": now_utc,
            "numbers": json.dumps(numbers),
            "probabilities": json.dumps(probabilities),
            "draw_date": draw_date,
            "window_size": WINDOW_SIZE,
            "epochs": trained_epochs
        }).execute()
        print(f"Prediction saved: {numbers}")
    except Exception as e:
        print(f"Save prediction error: {e}")

# ─── Scrape ───────────────────────────────────────────────────────────────────
def scrape_toto_latest():
    update_status("scraping", 0, message="Scraping latest draws...")
    print("Scraping latest draws...")
    url = "https://en.lottolyzer.com/history/singapore/toto?page=1"
    response = requests.get(url, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select("table tbody tr")
    draws = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 4:
            try:
                draws.append({
                    "draw_no": int(cols[0].text.strip()),
                    "draw_date": cols[1].text.strip(),
                    "winning_no": cols[2].text.strip(),
                    "additional_no": int(cols[3].text.strip()) if cols[3].text.strip() else None
                })
            except Exception:
                continue
    print(f"Scraped {len(draws)} draws")
    return draws

# ─── Update Supabase ──────────────────────────────────────────────────────────
def update_supabase(draws):
    update_status("scraping", 10, message=f"Saving {len(draws)} draws to Supabase...")
    print("Updating Supabase...")
    for draw in draws:
        supabase.table("toto_results").upsert(
            draw, on_conflict="draw_no"
        ).execute()
    print("Supabase updated!")

# ─── Load draws ───────────────────────────────────────────────────────────────
def load_draws():
    update_status("loading", 15, message=f"Loading latest {NUM_DRAWS} draws...")
    # update_status("loading", 15, message="Loading draws from Supabase...")
    print("Loading draws from Supabase...")
    # Load LATEST draws first then reverse for LSTM
    response = supabase.table("toto_results") \
        .select("draw_no, draw_date, winning_no, additional_no") \
        .order("draw_no", desc=True) \
        .limit(NUM_DRAWS) \
        .range(0, NUM_DRAWS - 1) \
        .execute()
    data = list(reversed(response.data))
    print(f"Loaded {len(data)} draws")
    print(f"From: {data[0]['draw_date']} to {data[-1]['draw_date']}")
    update_status("loading", 18, message=f"Loaded {len(data)} draws ({data[0]['draw_date']} → {data[-1]['draw_date']})")
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

# ─── Train model ──────────────────────────────────────────────────────────────
def train_model(draws):
    update_status("training", 20, message="Preparing sequences...")
    print("Training LSTM model...")
    data_X = draws_to_multihot(draws)
    window = WINDOW_SIZE

    sequences, targets = [], []
    for i in range(len(data_X) - window):
        sequences.append(data_X[i:i + window])
        targets.append(data_X[i + window])

    sequences = np.array(sequences)
    targets = np.array(targets)
    print(f"Prepared {len(sequences)} sequences")

    update_status("training", 25, message="Building LSTM model...")

    tf.random.set_seed(SEED)
    model = keras.Sequential([
        keras.layers.Input(shape=(window, 49)),
        # layers.LSTM(128, return_sequences=False),
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(49, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    val_split = 1.0 - TRAIN_RATIO
    start = time.time()
    
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_weights = None
    best_epoch = 0
    
    for ep in range(TRAIN_EPOCHS):
        hist = model.fit(
            sequences, targets,
            epochs=1,
            batch_size=BATCH_SIZE,
            validation_split=val_split,
            verbose=0
        )
        loss = hist.history['loss'][0]
        val_loss = hist.history.get('val_loss', [0])[0]
        elapsed = time.time() - start
        avg = elapsed / (ep + 1)
        remaining = avg * (TRAIN_EPOCHS - (ep + 1))
    
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            best_epoch = ep + 1
            patience_counter = 0
        else:
            patience_counter += 1
    
        # Map epoch progress to 25-85% range
        progress = 25 + int(((ep + 1) / TRAIN_EPOCHS) * 60)
        message = f"Epoch {ep+1}/{TRAIN_EPOCHS} — loss: {loss:.4f} val: {val_loss:.4f} — ETA: {remaining:.1f}s"
    
        update_status(
            "training",
            progress,
            epoch=ep + 1,
            total=TRAIN_EPOCHS,
            loss=loss,
            val_loss=val_loss,
            message=message
        )
        print(message)
        
        # Stop if no improvement for patience epochs (but only after MIN_EPOCHS)
        MIN_EPOCHS = 100
        if patience_counter >= patience and (ep + 1) >= MIN_EPOCHS:
            print(f"Early stopping at epoch {ep+1}! Best epoch: {best_epoch}")
            update_status(
                "training",
                85,
                epoch=ep + 1,
                total=TRAIN_EPOCHS,
                loss=loss,
                val_loss=best_val_loss,
                message=f"Early stopping at epoch {ep+1}! Best was epoch {best_epoch} (val_loss={best_val_loss:.4f})"
            )
            break
    
    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)
        print(f"Restored best weights from epoch {best_epoch}")
    
    print(f"Training complete in {time.time()-start:.1f}s")
    return model, draws, best_epoch
    
# ─── Upload to Supabase Storage ───────────────────────────────────────────────
def upload_to_supabase(local_path, storage_path, content_type):
    with open(local_path, 'rb') as f:
        supabase.storage.from_(MODEL_BUCKET).upload(
            path=storage_path,
            file=f,
            file_options={"upsert": "true", "content-type": content_type}
        )
    print(f"Uploaded: {storage_path}")

# ─── Convert and upload TF.js ─────────────────────────────────────────────────
def convert_and_upload_tfjs(model):
    update_status("converting", 87, message="Converting model to TF.js format...")
    print("Converting to TF.js format...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tfjs_dir = os.path.join(tmpdir, 'tfjs_model')
        os.makedirs(tfjs_dir)

        weights = {}
        weight_specs = []
        all_weights_bytes = b''

        for layer in model.layers:
            for weight in layer.weights:
                w_array = weight.numpy()
                w_name = weight.name
                w_shape = list(w_array.shape)
                w_bytes = w_array.astype('float32').tobytes()

                weight_specs.append({
                    'name': w_name,
                    'shape': w_shape,
                    'dtype': 'float32',
                    'quantization': None
                })
                all_weights_bytes += w_bytes

        bin_filename = 'group1-shard1of1.bin'
        bin_path = os.path.join(tfjs_dir, bin_filename)
        with open(bin_path, 'wb') as f:
            f.write(all_weights_bytes)

        model_config = model.get_config()
        model_json = {
            'format': 'layers-model',
            'generatedBy': 'keras v2',
            'convertedBy': 'TensorFlow.js Converter',
            'modelTopology': {
                'class_name': 'Sequential',
                'config': model_config,
                'keras_version': '2.15.0',
                'backend': 'tensorflow'
            },
            'weightsManifest': [{
                'paths': [bin_filename],
                'weights': weight_specs
            }]
        }

        json_path = os.path.join(tfjs_dir, 'model.json')
        with open(json_path, 'w') as f:
            json.dump(model_json, f)

        files = os.listdir(tfjs_dir)
        print(f"Files to upload: {files}")

        update_status("uploading", 90, message="Uploading model to Supabase...")

        for fname in files:
            fpath = os.path.join(tfjs_dir, fname)
            if os.path.isfile(fpath):
                content_type = 'application/json' if fname.endswith('.json') else 'application/octet-stream'
                upload_to_supabase(fpath, f'tfjs/{fname}', content_type)

    print("TF.js model uploaded to Supabase!")

# ─── Predict and save to Supabase ─────────────────────────────────────────────
def predict_and_save(model, draws, trained_epochs):
    update_status("predicting", 93, message="Running predictions...")
    print("Running predictions...")

    data_X = draws_to_multihot(draws)
    window = WINDOW_SIZE
    last_seq = data_X[-window:].reshape((1, window, 49)).astype(np.float32)

    mc_samples = 20
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
    latest_draw_date = draws[-1]['draw_date']

    save_prediction(numbers, probabilities, latest_draw_date, trained_epochs)
    print(f"Predicted: {numbers}")
    return numbers, probabilities

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TOTO Auto Trainer Started ===")
    update_status("starting", 0, message="Starting TOTO Auto Trainer...")

    # Load settings from Supabase
    settings = load_settings()
    TRAIN_EPOCHS = settings.get("epochs", 100)
    BATCH_SIZE = settings.get("batch_size", 64)
    TRAIN_RATIO = settings.get("train_ratio", 0.85)
    WINDOW_SIZE = settings.get("window_size", 15)
    NUM_DRAWS = settings.get("num_draws", 1000)
    PREDICT_ONLY = settings.get("predict_only", False)
    print(f"Settings: epochs={TRAIN_EPOCHS} batch={BATCH_SIZE} window={WINDOW_SIZE} predict_only={PREDICT_ONLY}")

    try:
        if PREDICT_ONLY:
            # Only predict using existing model
            update_status("predicting", 50, message="Loading draws for prediction...")
            draws = load_draws()
    
            # Download model from Supabase Storage
            from tensorflow import keras
            try:
                update_status("predicting", 30, message="Downloading model from Supabase...")
                model_data = supabase.storage.from_(MODEL_BUCKET).download('lstm_model.h5')
                with open('lstm_model.h5', 'wb') as f:
                    f.write(model_data)
                model = keras.models.load_model('lstm_model.h5')
                print("Model downloaded and loaded!")
            except Exception as e:
                print(f"Model download error: {e}")
                model = None

            if model is None:
                update_status("error", 0, message="No trained model found. Please train first!")
            else:
                # Fetch the best epoch count from the last training run
                try:
                    meta_response = supabase.table("model_meta").select("best_epoch").eq("id", 1).single().execute()
                    trained_epochs = meta_response.data.get("best_epoch", TRAIN_EPOCHS) if meta_response.data else TRAIN_EPOCHS
                except Exception as e:
                    print(f"Could not load best_epoch from model_meta: {e}")
                    trained_epochs = TRAIN_EPOCHS

                numbers, probs = predict_and_save(model, draws, trained_epochs)
    
                update_status("complete", 100, message=f"✅ Predicted: {numbers}")
                # Reset predict_only flag
                supabase.table("training_config").upsert({
                    "id": 1,
                    "predict_only": False
                }).execute()
                print("=== Prediction Done! ===")

        
        else:
            # Full pipeline: scrape + train + predict
            # Step 1 - Scrape
            draws = scrape_toto_latest()
            if draws:
                update_supabase(draws)
                
            # Step 2 - Load & Train
            draws = load_draws()
            model, draws, best_epoch = train_model(draws)

            # Save best_epoch so predict-only runs can reference it later
            supabase.table("model_meta").upsert({
                "id": 1,
                "best_epoch": best_epoch
            }).execute()
    
            # Step 3 - Save model locally and upload to Supabase
            model.save('lstm_model.h5')
            with open('lstm_model.h5', 'rb') as f:
                supabase.storage.from_(MODEL_BUCKET).upload(
                    path='lstm_model.h5',
                    file=f,
                    file_options={"upsert": "true", "content-type": "application/octet-stream"}
                )
            print("Model saved to Supabase Storage!")
             
            # Step 4 - Convert & Upload TF.js
            convert_and_upload_tfjs(model)

            # Step 5 - Done! (no prediction here)
            update_status(
                "complete",
                100,
                message=f"✅ Training done! Tap Predict to get numbers."
            )
            print("=== Training Done! ===")
   
    except Exception as e:
        print(f"Error: {e}")
        update_status("error", 0, message=f"Error: {str(e)}")    
