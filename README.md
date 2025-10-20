**Music Recommender System**

**Overview**
This project is a Music Recommendation Web Application built using Flask and Machine Learning (KMeans Clustering).
The app recommends songs based on a user’s input (song name and release year).
It first searches the song locally in a pre-collected Spotify dataset, and if unavailable, it attempts to fetch it via the Spotify API.

The system then computes the feature similarity between songs using numerical audio characteristics such as danceability, energy, valence, tempo, etc., and suggests similar tracks.
Deployed by Render at : https://spotify-song-recommender-o6mb.onrender.com

**Introduction**

Music recommendation systems play a crucial role in modern streaming platforms by enhancing user experience through personalized suggestions. This project aims to design a Spotify Music Recommender System that leverages machine learning to analyze song features and cluster them for recommendation. The problem addressed is the need for efficient and intelligent music discovery. The objectives include building a clustering-based recommendation engine, performing exploratory analysis, and deploying a Flask web application for user interaction.

**Methodology**

The methodology involved several stages: data collection, preprocessing, model training, evaluation, and deployment.

• Data Source: The dataset was obtained from Kaggle’s Spotify dataset, containing song-level audio features and metadata.
• Data Preprocessing: Missing values were handled, irrelevant columns dropped, and numerical features normalized using StandardScaler.
• Exploratory Data Analysis: Correlation matrices, feature importance, and time-based feature trends were analyzed.
• Clustering: A KMeans model was trained with 20 clusters using audio features such as valence, energy, danceability, and loudness to group songs with similar acoustic profiles.
• Deployment: A Flask web app was built to serve recommendations based on user-inputted songs. The system can function offline using a pre-trained model or online via Spotify’s API for additional song data.Then   Deployed using Render.

**Discussion of Results or Findings**

The exploratory analysis revealed strong correlations among features such as energy and loudness, and a distinct clustering pattern emerged across genres and decades. Visualization tools such as Plotly, Seaborn, and Yellowbrick were used to display feature relationships and clustering behavior.

The KMeans clustering effectively grouped songs into meaningful categories that reflect their underlying musical properties. Dimensionality reduction using PCA and t-SNE confirmed clear separations among clusters.

When deployed, the Flask app successfully generated recommendations by computing the cosine similarity between user input and clustered songs. The system performed well with minimal latency, even without online access to Spotify’s API.


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

**Running the Application**

- **Locally**:  
  1. Make sure `.venv` is activated.  
  2. Run `python app.py`  
  3. Open http://127.0.0.1:5000 in your browser

- **Deployed on Render**:  
  Visit: [https://spotify-song-recommender-o6mb.onrender.com](https://spotify-song-recommender-o6mb.onrender.com)


**Production Deployment**
- The app is served with `gunicorn app:app` on Render or other cloud hosts.



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

**Conclusion**
The Spotify Music Recommender System achieved its goal of generating music recommendations based on song audio features using unsupervised learning. The combination of clustering and feature scaling allowed the system to learn meaningful song relationships.

**Future Improvements**
Add user login & personalized recommendations
Integrate playlist creation using Spotify API
Improve UI with CSS and animations
Add popularity-based filtering


**👩‍💻 Authors**

Msweli Nhlanzeko- 231067275
Mamabu Langa- 230317707
Talifhani Netshivhulana- 222421193
🎓 Bachelor in Information and Communication Technology (BICT)
📍 University of Mpumalanga



