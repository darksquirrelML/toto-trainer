import streamlit as st
import numpy as np
import os
import tempfile
import time
import requests
from bs4 import BeautifulSoup
from supabase import create_client
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs

# ─── Config ───────────────────────────────────────────────────────────────────
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_BUCKET = "models"

st.set_page_config(page_title="TOTO Control Panel", page_icon="🎰")
st.title("🎰 TOTO Control Panel")

# ─── Sidebar settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Training Settings")
    train_epochs = st.number_input("Epochs", min_value=10, max_value=600, value=100)
    batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=64)
    train_ratio = st.slider("Train ratio", 0.5, 0.95, 0.85)
    window_size = st.number_input("Window size", min_value=5, max_value=30, value=15)
    num_draws = st.number_input("Number of draws", min_value=100, max_value=5000, value=1000)
    seed = st.number_input("Random seed", value=42)

# ─── Helper functions ─────────────────────────────────────────────────────────
def scrape_toto_latest():
    url = "https://en.lottolyzer.com/history/singapore/toto?page=1"
    response = requests.get(url)
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
    return draws

def load_draws(limit=1000):
    response = supabase.table("toto_results") \
        .select("draw_no, draw_date, winning_no, additional_no") \
        .order("draw_no", desc=True) \
        .limit(limit) \
        .execute()
    return response.data

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

def upload_to_supabase(local_path, storage_path, content_type):
    with open(local_path, 'rb') as f:
        supabase.storage.from_(MODEL_BUCKET).upload(
            path=storage_path,
            file=f,
            file_options={"upsert": "true", "content-type": content_type}
        )

def convert_and_upload_tfjs(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        tfjs_dir = os.path.join(tmpdir, 'tfjs_model')
        os.makedirs(tfjs_dir)
        tfjs.converters.save_keras_model(model, tfjs_dir)
        files = os.listdir(tfjs_dir)
        for fname in files:
            fpath = os.path.join(tfjs_dir, fname)
            content_type = 'application/json' if fname.endswith('.json') else 'application/octet-stream'
            upload_to_supabase(fpath, f'tfjs/{fname}', content_type)
            st.write(f"✅ Uploaded: tfjs/{fname}")
    st.success("🎉 TF.js model uploaded to Supabase!")

# ─── SECTION 1: Latest Draw & Trends ─────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Latest Draws & Trends")

if st.button("🔄 Load Latest Data"):
    with st.spinner("Loading..."):
        draws = load_draws(int(num_draws))
        st.session_state['draws'] = draws

if 'draws' in st.session_state:
    draws = st.session_state['draws']
    df = pd.DataFrame(draws)

    latest = draws[0]
    st.success(
        f"Draw #{latest['draw_no']} | "
        f"Date: {latest['draw_date']} | "
        f"Numbers: {latest['winning_no']} | "
        f"Additional: {latest['additional_no']}"
    )

    st.markdown("**Recent 10 Draws:**")
    st.dataframe(df[['draw_no', 'draw_date', 'winning_no', 'additional_no']].head(10))

    st.markdown("**Number Frequency:**")
    all_nums = []
    for draw in draws:
        nums = [int(n.strip()) for n in str(draw['winning_no']).split(',')]
        all_nums.extend(nums)
        if draw['additional_no']:
            all_nums.append(int(draw['additional_no']))

    freq = pd.Series(all_nums).value_counts().sort_index()
    freq_df = pd.DataFrame({'Number': freq.index, 'Count': freq.values})
    st.bar_chart(freq_df.set_index('Number'))

    top6 = freq_df.sort_values('Count', ascending=False).head(6)['Number'].tolist()
    bottom6 = freq_df.sort_values('Count').head(6)['Number'].tolist()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔥 Top 6 (most frequent)", ', '.join(map(str, top6)))
    with col2:
        st.metric("🧊 Bottom 6 (least frequent)", ', '.join(map(str, bottom6)))

# ─── SECTION 2: Scrape ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔄 Scrape Latest Draws")

if st.button("Scrape & Update Supabase"):
    with st.spinner("Scraping..."):
        scraped = scrape_toto_latest()
        if scraped:
            for draw in scraped:
                supabase.table("toto_results").upsert(
                    draw, on_conflict="draw_no"
                ).execute()
            st.success(f"✅ {len(scraped)} draws updated!")
            st.write(f"Latest: Draw #{scraped[0]['draw_no']} — {scraped[0]['draw_date']}")
        else:
            st.warning("No draws found.")

# ─── SECTION 3: Train ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧠 Train LSTM Model")
st.write(f"Settings: {train_epochs} epochs | batch {batch_size} | window {window_size} | ratio {train_ratio}")

if st.button("🧠 Train LSTM Model"):
    if 'draws' not in st.session_state:
        st.warning("Load data first!")
    else:
        with st.spinner("Loading draws..."):
            draws = load_draws(int(num_draws))
            draws_asc = list(reversed(draws))

        data_X = draws_to_multihot(draws_asc)
        window = int(window_size)

        sequences, targets = [], []
        for i in range(len(data_X) - window):
            sequences.append(data_X[i:i + window])
            targets.append(data_X[i + window])

        sequences = np.array(sequences)
        targets = np.array(targets)
        st.write(f"Prepared {len(sequences)} sequences")

        tf.random.set_seed(int(seed))
        model = keras.Sequential([
            keras.layers.Input(shape=(window, 49)),
            layers.LSTM(128, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(49, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')

        val_split = 1.0 - float(train_ratio)
        progress = st.progress(0)
        status = st.empty()
        loss_chart = st.empty()
        history_logs = {"loss": [], "val_loss": []}
        start_time = time.time()

        for ep in range(int(train_epochs)):
            hist = model.fit(
                sequences, targets,
                epochs=1,
                batch_size=int(batch_size),
                validation_split=val_split,
                verbose=0
            )
            loss = hist.history['loss'][0]
            val_loss = hist.history.get('val_loss', [0])[0]
            history_logs['loss'].append(loss)
            history_logs['val_loss'].append(val_loss)
            percent = int(((ep + 1) / int(train_epochs)) * 100)
            progress.progress(percent)
            elapsed = time.time() - start_time
            avg = elapsed / (ep + 1)
            remaining = avg * (int(train_epochs) - (ep + 1))
            status.text(
                f"Epoch {ep+1}/{train_epochs} — loss: {loss:.4f} val: {val_loss:.4f} — ETA: {remaining:.1f}s"
            )
            loss_chart.line_chart({
                "loss": history_logs['loss'],
                "val_loss": history_logs['val_loss']
            })

        progress.progress(100)
        st.success(f"✅ Training done in {time.time()-start_time:.1f}s!")
        st.session_state['trained_model'] = model

# ─── SECTION 4: Export ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📱 Export to Mobile App")

if 'trained_model' not in st.session_state:
    st.info("Train model first, then export here.")
else:
    if st.button("📱 Convert & Upload TF.js to Supabase"):
        convert_and_upload_tfjs(st.session_state['trained_model'])
        st.balloons()

# ─── SECTION 5: Verify ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("✅ Verify Model in Supabase")

if st.button("Check TF.js Files in Supabase"):
    try:
        files = supabase.storage.from_(MODEL_BUCKET).list('tfjs')
        if files:
            st.success(f"Found {len(files)} TF.js files:")
            for f in files:
                st.write(f"✅ tfjs/{f['name']}")
        else:
            st.warning("No TF.js files found. Train and export first.")
    except Exception as e:
        st.error(f"Error: {e}")