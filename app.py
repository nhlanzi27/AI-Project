import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
from dotenv import load_dotenv
import joblib

# ===============================
# 1️⃣ LOAD ENV VARIABLES
# ===============================
load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# ===============================
# 2️⃣ INITIALIZE SPOTIFY CLIENT (used only as backup)
# ===============================
sp = None
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))
        print("✅ Spotify client initialized successfully.")
    except Exception as e:
        print("⚠️ Spotify client initialization failed:", e)
else:
    print("⚠️ Missing Spotify API credentials. Running in offline mode.")

# ===============================
# 3️⃣ FLASK APP SETUP
# ===============================
app = Flask(__name__)

# ===============================
# 4️⃣ LOAD TRAINED MODEL + LOCAL DATASET (updated with path fix)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ✅ base folder of app.py
MODEL_PATH = os.path.join(BASE_DIR, 'song_cluster_pipeline.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'spotify_data.csv')

song_cluster_pipeline = None
spotify_data = None
number_cols = None

try:
    song_cluster_pipeline = joblib.load(MODEL_PATH)
    spotify_data = pd.read_csv(DATA_PATH)
    number_cols = [
        'valence', 'year', 'acousticness', 'danceability', 'duration_ms',
        'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
        'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
    ]
    print("✅ Model and dataset loaded successfully.")
except Exception as e:
    print(f"⚠️ .pkl missing or invalid ({e}). Fitting pipeline now...")
    # Load data
    spotify_data = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded: {spotify_data.shape}")
    
    # Fit pipeline (your notebook code)
    song_cluster_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=20, verbose=False))
    ], verbose=False)

    X = spotify_data.select_dtypes(np.number)
    number_cols = list(X.columns)  # Use actual columns from data
    song_cluster_pipeline.fit(X)
    print("✅ Pipeline fitted!")
    
    # Save for next time
    joblib.dump(song_cluster_pipeline, MODEL_PATH)
    print("✅ .pkl saved successfully.")

# ===============================
# 5️⃣ HELPER FUNCTIONS
# ===============================

def find_song_online(name, year):
    """Fallback: Try finding song info + audio features via Spotify API."""
    if not sp:
        print(f"⚠️ Offline mode: cannot fetch '{name}' ({year}) from Spotify.")
        return None

    song_data = defaultdict(list)
    try:
        results = sp.search(q=f"track:{name} year:{year}", limit=1)
        if not results['tracks']['items']:
            return None
        track = results['tracks']['items'][0]
        track_id = track['id']
        audio_features = sp.audio_features([track_id])[0]
        if not audio_features:
            return None

        # Build DataFrame
        song_data['name'].append(track['name'])
        song_data['artists'].append(", ".join([a['name'] for a in track['artists']]))
        song_data['year'].append(year)
        song_data['explicit'].append(int(track['explicit']))
        song_data['duration_ms'].append(track['duration_ms'])
        song_data['popularity'].append(track['popularity'])

        for key, value in audio_features.items():
            song_data[key] = [value]

        print(f"✅ Found '{name}' online.")
        return pd.DataFrame(song_data)

    except Exception as e:
        print(f"⚠️ Error fetching '{name}' from Spotify: {e}")
        return None


def get_song_data(song):
    """Try to find song locally first, then online if missing."""
    try:
        match = spotify_data[
            (spotify_data['name'].str.lower() == song['name'].lower()) &
            (spotify_data['year'] == song['year'])
        ]
        if not match.empty:
            print(f"✅ Found '{song['name']}' in local dataset.")
            return match.iloc[0]
        else:
            return find_song_online(song['name'], song['year'])
    except Exception as e:
        print(f"⚠️ Error retrieving song data: {e}")
        return None


def get_mean_vector(song_list):
    """Compute mean vector for multiple input songs."""
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song)
        if song_data is None:
            print(f"⚠️ '{song['name']}' not found locally or online.")
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    if not song_vectors:
        return None

    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)


def recommend_songs(song_list, n_songs=10):
    """Main recommendation logic (offline-first)."""
    song_center = get_mean_vector(song_list)
    if song_center is None:
        return []

    # Use trained scaler
    scaler = song_cluster_pipeline.named_steps['scaler']
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = np.argsort(distances[0])[:n_songs]
    recs = spotify_data.iloc[index]

    return recs[['name', 'artists', 'year']].to_dict(orient='records')


# ===============================
# 6️⃣ FLASK ROUTES
# ===============================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    song_name = request.form.get('song', '').strip()
    song_year = request.form.get('year', '').strip()

    if not song_name or not song_year.isdigit():
        return render_template('results.html',
                               recommendations=[],
                               song=song_name,
                               message="⚠️ Please enter a valid song name and year.")

    input_song = [{'name': song_name, 'year': int(song_year)}]
    recs = recommend_songs(input_song)

    if not recs:
        return render_template('results.html',
                               recommendations=[],
                               song=song_name,
                               message=f"No recommendations found for '{song_name}'. Try another song.")
    
    return render_template('results.html',
                           recommendations=recs,
                           song=song_name,
                           message=None)


# 7️⃣ RUN FLASK APP
if __name__ == '__main__':
    app.run(debug=True)
