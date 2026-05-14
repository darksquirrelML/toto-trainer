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
TRAIN_EPOCHS = 100
BATCH_SIZE = 64
TRAIN_RATIO = 0.85
WINDOW_SIZE = 15
NUM_DRAWS = 1000
SEED = 42

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
def save_prediction(numbers, probabilities, draw_date):
    try:
        supabase.table("predictions").insert({
            "numbers": json.dumps(numbers),
            "probabilities": json.dumps(probabilities),
            "draw_date": draw_date,
            "window_size": WINDOW_SIZE,
            "epochs": TRAIN_EPOCHS
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
    update_status("loading", 15, message="Loading draws from Supabase...")
    print("Loading draws from Supabase...")
    response = supabase.table("toto_results") \
        .select("draw_no, draw_date, winning_no, additional_no") \
        .order("draw_no", desc=False) \
        .limit(NUM_DRAWS) \
        .execute()
    print(f"Loaded {len(response.data)} draws")
    return response.data

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
        layers.LSTM(128, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(49, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')

    val_split = 1.0 - TRAIN_RATIO
    start = time.time()

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

        # Map epoch progress to 25-85% range
        progress = 25 + int(((ep + 1) / TRAIN_EPOCHS) * 60)
        message = f"Epoch {ep+1}/{TRAIN_EPOCHS} — loss: {loss:.4f} — ETA: {remaining:.1f}s"

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

    print(f"Training complete in {time.time()-start:.1f}s")
    return model, draws

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
def predict_and_save(model, draws):
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

    # Recent numbers priority
    recent = draws[-10:]
    recent_numbers = set()
    for row in recent:
        nums = [int(n.strip()) for n in str(row["winning_no"]).split(",")]
        recent_numbers.update(nums)
        if row["additional_no"]:
            recent_numbers.add(int(row["additional_no"]))

    all_sorted = sorted(
        [(i + 1, float(avg_probs[i])) for i in range(49)],
        key=lambda x: x[1], reverse=True
    )

    top7 = []
    for num, prob in all_sorted:
        if num in recent_numbers:
            top7.append((num, prob))
        if len(top7) == 7:
            break
    if len(top7) < 7:
        for num, prob in all_sorted:
            if num not in [x[0] for x in top7]:
                top7.append((num, prob))
            if len(top7) == 7:
                break

    numbers = [x[0] for x in top7]
    probabilities = [round(x[1], 4) for x in top7]
    latest_draw_date = draws[-1]['draw_date']

    save_prediction(numbers, probabilities, latest_draw_date)
    print(f"Predicted: {numbers}")
    return numbers, probabilities

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TOTO Auto Trainer Started ===")
    update_status("starting", 0, message="Starting TOTO Auto Trainer...")

    try:
        # Step 1 - Scrape
        draws = scrape_toto_latest()
        if draws:
            update_supabase(draws)

        # Step 2 - Load & Train
        draws = load_draws()
        model, draws = train_model(draws)

        # Step 3 - Convert & Upload TF.js
        convert_and_upload_tfjs(model)

        # Step 4 - Predict & Save
        numbers, probs = predict_and_save(model, draws)

        # Step 5 - Done!
        update_status(
            "complete",
            100,
            message=f"✅ Done! Predicted: {numbers}"
        )
        print("=== All Done! ===")

    except Exception as e:
        print(f"Error: {e}")
        update_status("error", 0, message=f"Error: {str(e)}")
