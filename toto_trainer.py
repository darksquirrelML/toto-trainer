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

# ─── Scrape ───────────────────────────────────────────────────────────────────
def scrape_toto_latest():
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
    print("Updating Supabase...")
    for draw in draws:
        supabase.table("toto_results").upsert(
            draw, on_conflict="draw_no"
        ).execute()
    print("Supabase updated!")

# ─── Load draws ───────────────────────────────────────────────────────────────
def load_draws():
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
        print(f"Epoch {ep+1}/{TRAIN_EPOCHS} — loss: {loss:.4f} val: {val_loss:.4f} — ETA: {remaining:.1f}s")

    print(f"Training complete in {time.time()-start:.1f}s")
    return model

# ─── Upload to Supabase ───────────────────────────────────────────────────────
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
    print("Converting to TF.js format...")
    import json
    import struct

    with tempfile.TemporaryDirectory() as tmpdir:
        tfjs_dir = os.path.join(tmpdir, 'tfjs_model')
        os.makedirs(tfjs_dir)

        # Get model weights manually
        weights_manifest = []
        all_weights_bytes = b''
        weight_specs = []
        offset = 0

        for layer in model.layers:
            for weight in layer.weights:
                w_array = weight.numpy()
                w_name = weight.name
                w_shape = list(w_array.shape)
                w_dtype = 'float32'
                w_bytes = w_array.astype('float32').tobytes()
                w_size = len(w_bytes)

                weight_specs.append({
                    'name': w_name,
                    'shape': w_shape,
                    'dtype': w_dtype,
                    'quantization': None
                })

                all_weights_bytes += w_bytes
                offset += w_size

        # Save weights binary file
        bin_filename = 'group1-shard1of1.bin'
        bin_path = os.path.join(tfjs_dir, bin_filename)
        with open(bin_path, 'wb') as f:
            f.write(all_weights_bytes)
        print(f"Weights saved: {len(all_weights_bytes)} bytes")

        # Build model topology
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

        # Save model.json
        json_path = os.path.join(tfjs_dir, 'model.json')
        with open(json_path, 'w') as f:
            json.dump(model_json, f)
        print("model.json saved")

        # Upload to Supabase
        files = os.listdir(tfjs_dir)
        print(f"Files to upload: {files}")

        for fname in files:
            fpath = os.path.join(tfjs_dir, fname)
            content_type = 'application/json' if fname.endswith('.json') else 'application/octet-stream'
            upload_to_supabase(fpath, f'tfjs/{fname}', content_type)

    print("TF.js model uploaded to Supabase!")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TOTO Auto Trainer Started ===")

    # Step 1 - Scrape
    draws = scrape_toto_latest()
    if draws:
        update_supabase(draws)

    # Step 2 - Load & Train
    draws = load_draws()
    model = train_model(draws)

    # Step 3 - Convert & Upload TF.js
    convert_and_upload_tfjs(model)

    print("=== All Done! Phone app can now predict! ===")
