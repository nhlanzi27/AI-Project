**Music Recommender System**
📖 Overview

This project is a Music Recommendation Web Application built using Flask and Machine Learning (KMeans Clustering).
The app recommends songs based on a user’s input (song name and release year).
It first searches the song locally in a pre-collected Spotify dataset, and if unavailable, it attempts to fetch it via the Spotify API.

The system then computes the feature similarity between songs using numerical audio characteristics such as danceability, energy, valence, tempo, etc., and suggests similar tracks.

**Features**

✅ Web-based interface (Flask)
✅ Offline-first recommendation (works even without Spotify API)
✅ Fallback online search using the Spotify API
✅ Trained clustering model saved via joblib for faster startup
✅ Cosine similarity–based recommendations
✅ User-friendly results page displaying top recommended songs

**System Architecture**
1️⃣ Data Source

The app uses a CSV file (spotify_data.csv) containing song metadata and audio features.

Key features include:
valence, danceability, energy, tempo, popularity, year, speechiness, etc.

**2️⃣ Machine Learning**

A KMeans clustering pipeline groups similar songs together.

The features are standardized using StandardScaler to normalize the data.

A mean vector is computed for the user’s selected songs, and cosine distance is used to find the nearest songs.

**3️⃣ Backend (Flask)**

/ → Displays the input form (index page)

/recommend → Handles user requests and returns top song recommendations

**4️⃣ Frontend**

index.html: Song input form

results.html: Displays the recommended songs and messages

**Folder Structure**

MusicRecommender/
│
├── app.py                     # Flask application
├── spotify_data.csv           # Dataset with song features
├── song_cluster_pipeline.pkl  # Trained ML model (auto-generated)
├── .env                       # Spotify API credentials (not shared publicly)
│
├── templates/                 # HTML templates
│   ├── index.html
│   └── results.html
│
└── README.md                  # Project documentation

**Installation & Setup**

1️⃣ Clone the project
git clone https://github.com/yourusername/MusicRecommender.git
cd MusicRecommender

2️⃣ Create a virtual environment
python -m venv .venv

Activate it:

Windows: .\.venv\Scripts\activate
macOS/Linux: source .venv/bin/activate

3️⃣ Install dependencies

pip install -r requirements.txt

If you don’t have a requirements.txt, install manually:

pip install flask pandas numpy spotipy scikit-learn joblib python-dotenv scipy

4️⃣ Add Spotify credentials

Create a .env file in your project root and add:
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret

5️⃣ Run the Flask app

python app.py
Open your browser and visit 👉 http://127.0.0.1:5000

**🧾 Usage Instructions**

Enter a song name and release year on the homepage.

Click “Get Recommendations.”

The app will return a list of similar songs (based on clustering and feature similarity).

If the song isn’t found locally, the system will attempt to fetch it using the Spotify API (if API credentials are available).

**🧠 How It Works (Simplified Flow)**

Load dataset and trained ML model

User enters a song and year

App finds song data (locally or via Spotify API)

Compute the mean vector of the song’s features

Compare it to all songs in the dataset using cosine similarity

Return the top 10 most similar songs

**📊 Example Output**

Input:
Song: “Shape of You”
Year: 2017

Recommendations:

Song	Artist	Year
Castle on the Hill	Ed Sheeran	2017
Galway Girl	Ed Sheeran	2017
Perfect	Ed Sheeran	2017
Something Just Like This	The Chainsmokers & Coldplay	2017
...	...	...
**💡 Future Improvements**

Add user login & personalized recommendations

Integrate playlist creation using Spotify API

Improve UI with CSS and animations

Add popularity-based filtering

**👩‍💻 Authors**

MAMABU LANGA 
NHLANZEKO MSWELI
TALIFHANI NETSHIVHULANA
🎓 Bachelor in Information and Communication Technology (BICT)
📍 University of Mpumalanga



