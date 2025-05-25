import requests
import os
import json
import re
import yt_dlp
import librosa
import numpy as np
import threading
import concurrent.futures
import hashlib
import time
from flask import Flask, request, redirect, jsonify
from datetime import datetime

app = Flask(__name__)
TEMP_DIR = "/tmp/temp_audio"
SHARED_DIR = "/tmp/shared"
CACHE_DIR = "/tmp/audio_cache"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SHARED_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "d68f3bd56ef246aeac276470eb8926b5")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "569174440ac1449cb546a573d8a6310c")
REDIRECT_URI = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:5000/callback")
SCOPE = 'user-top-read user-read-email'

# Processing status tracking
processing_status = {}

# Thread pool for parallel processing
MAX_WORKERS = 4  # Adjust based on your server's CPU cores

def sanitize_filename(text):
    return re.sub(r'[\\/*?:"<>|()\']', "_", text)

def get_cache_key(track, artist):
    """Generate a unique cache key for track+artist combination"""
    return hashlib.md5(f"{track.lower()}_{artist.lower()}".encode()).hexdigest()

def get_cached_features(track, artist):
    """Check if features are already cached"""
    cache_key = get_cache_key(track, artist)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def cache_features(track, artist, features):
    """Cache the extracted features"""
    cache_key = get_cache_key(track, artist)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump(features, f)
    except:
        pass

def download_audio_optimized(track, artist):
    """Optimized audio download with better settings"""
    query = f"ytsearch1:{track} {artist} audio"
    filename = f"{sanitize_filename(track)}_{sanitize_filename(artist)}.%(ext)s"
    path = os.path.join(TEMP_DIR, filename)
    
    ydl_opts = {
        'format': 'bestaudio[ext=webm]/bestaudio/best',  # Prefer webm for faster download
        'quiet': True,
        'no_warnings': True,
        'outtmpl': path,
        'cachedir': False,
        'socket_timeout': 30,  # Prevent hanging
        'retries': 2,
        'fragment_retries': 2,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',  # Lower quality for faster processing
        }],
        'postprocessor_args': ['-ac', '1', '-ar', '22050'],  # Mono, lower sample rate
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
        return path.replace("%(ext)s", "mp3")
    except Exception as e:
        print(f"Download failed for {track} - {artist}: {e}")
        return None

def extract_features_fast(audio_path):
    """Faster feature extraction with optimized settings"""
    try:
        # Load with lower sample rate for faster processing
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)  # Only first 30 seconds
        
        # Use hop_length for faster processing
        hop_length = 1024
        
        # Extract features more efficiently
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        energy = np.mean(rms)
        
        # Compute loudness
        loudness = np.mean(librosa.amplitude_to_db(rms, ref=np.max))
        
        # Spectral features (simplified)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        danceability = min(1.0, np.mean(spectral_centroids) / 4000)  # Normalized
        
        # Simple valence estimation
        valence = (0.6 + 0.4 * min(1.0, tempo / 150)) / 2
        
        return {
            "tempo": float(tempo),
            "loudness": float(loudness),
            "energy": float(energy),
            "danceability": float(danceability),
            "valence": float(valence),
        }
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

def process_single_track(track_data):
    """Process a single track - designed for parallel execution"""
    name = track_data.get("name", "")
    artist = track_data.get("artist", "")
    
    # Check cache first
    cached_features = get_cached_features(name, artist)
    if cached_features:
        track_data["features"] = cached_features
        return track_data
    
    # Download and process if not cached
    audio_path = download_audio_optimized(name, artist)
    if audio_path and os.path.exists(audio_path):
        try:
            features = extract_features_fast(audio_path)
            os.remove(audio_path)  # Clean up immediately
            if features:
                cache_features(name, artist, features)  # Cache for future use
                track_data["features"] = features
            else:
                track_data["features"] = None
        except Exception as e:
            print(f"Error processing {name}: {e}")
            track_data["features"] = None
            if os.path.exists(audio_path):
                os.remove(audio_path)
    else:
        track_data["features"] = None
    
    return track_data

def analyze_user_async(username):
    """Parallel audio analysis using ThreadPoolExecutor"""
    try:
        processing_status[username] = {"status": "processing", "step": "downloading_audio", "progress": 0}
        
        input_path = os.path.join(SHARED_DIR, f"tracks_{username}.json")
        if not os.path.exists(input_path):
            processing_status[username] = {"status": "error", "message": "tracks file not found"}
            return

        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        tracks = data.get("tracks", [])
        total_tracks = len(tracks)
        
        if not tracks:
            processing_status[username] = {"status": "error", "message": "no tracks found"}
            return
        
        # Process tracks in parallel
        processed_tracks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tracks for processing
            future_to_track = {executor.submit(process_single_track, track.copy()): i 
                             for i, track in enumerate(tracks)}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_track):
                track_idx = future_to_track[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per track
                    processed_tracks.append((track_idx, result))
                    
                    # Update progress
                    progress = int((len(processed_tracks) / total_tracks) * 50)  # 50% for audio processing
                    processing_status[username]["progress"] = progress
                    processing_status[username]["step"] = f"processed_{len(processed_tracks)}_of_{total_tracks}"
                    
                except concurrent.futures.TimeoutError:
                    print(f"Timeout processing track {track_idx}")
                    processed_tracks.append((track_idx, tracks[track_idx]))  # Use original without features
                except Exception as e:
                    print(f"Error processing track {track_idx}: {e}")
                    processed_tracks.append((track_idx, tracks[track_idx]))  # Use original without features

        # Sort by original order and update data
        processed_tracks.sort(key=lambda x: x[0])
        data["tracks"] = [track for _, track in processed_tracks]

        # Save features data
        output_path = os.path.join(SHARED_DIR, f"features_{username}.json")
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=2)

        processing_status[username]["step"] = "finalizing_data"
        processing_status[username]["progress"] = 75

        # Finalize data
        finalize_data(username)
        
        processing_status[username] = {"status": "completed", "progress": 100}
        
        # Clean up any remaining temp files
        cleanup_temp_files()
        
    except Exception as e:
        processing_status[username] = {"status": "error", "message": str(e)}
        cleanup_temp_files()

def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        now = time.time()
        for filename in os.listdir(TEMP_DIR):
            filepath = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(filepath):
                # Remove files older than 1 hour
                if now - os.path.getctime(filepath) > 3600:
                    os.remove(filepath)
    except Exception as e:
        print(f"Cleanup error: {e}")
    
def finalize_data(username):
    features_path = os.path.join(SHARED_DIR, f"features_{username}.json")
    if not os.path.exists(features_path):
        return None

    with open(features_path, encoding="utf-8") as f:
        features_data = json.load(f)

    access_token = features_data.get("access_token")
    if not access_token:
        return None

    headers = {"Authorization": f"Bearer {access_token}"}
    final = []

    user_object = {
        "id": 2,
        "username": features_data.get("username"),
        "role": "user"
    }

    for i, track in enumerate(features_data.get("tracks", []), 1):
        spotify_id = track["spotifyId"]
        local_feats = track.get("features", {})

        r = requests.get(f"https://api.spotify.com/v1/audio-features/{spotify_id}", headers=headers)
        audio_features = r.json() if r.status_code == 200 else {}

        final.append({
            "id": i,
            "trackName": track["name"],
            "artist": track["artist"],
            "spotifyId": spotify_id,
            "coverPicture": track.get("coverPicture", ""),
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "features": {
                "id": i,
                "genre": ", ".join(track.get("genres", [])) or "Unknown",
                "acousticness": audio_features.get("acousticness", 0.0),
                "danceability": local_feats.get("danceability", 0.0),
                "energy": local_feats.get("energy", 0.0),
                "instrumentalness": audio_features.get("instrumentalness", 0.0),
                "key": audio_features.get("key", 0),
                "liveness": audio_features.get("liveness", 0.0),
                "loudness": local_feats.get("loudness", 0.0),
                "mode": audio_features.get("mode", 0),
                "speechiness": audio_features.get("speechiness", 0.0),
                "tempo": local_feats.get("tempo", 0.0),
                "timeSignature": audio_features.get("time_signature", 4),
                "valence": local_feats.get("valence", 0.0)
            },
            "user": user_object
        })

    output_path = os.path.join(SHARED_DIR, f"final_{username}.json")
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(final, out, ensure_ascii=False, indent=2)

    return output_path

@app.route('/')
def authorize():
    auth_url = (
        'https://accounts.spotify.com/authorize'
        f'?client_id={CLIENT_ID}&response_type=code'
        f'&redirect_uri={REDIRECT_URI}'
        f'&scope={SCOPE}&show_dialog=true'
    )
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_response = requests.post(
        'https://accounts.spotify.com/api/token',
        data={
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': REDIRECT_URI,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        },
        headers={'Content-Type': 'application/x-www-form-urlencoded'}
    )

    if token_response.status_code != 200:
        return "❌ Token request failed."

    access_token = token_response.json()['access_token']

    # Get user data quickly
    user_data = requests.get(
        'https://api.spotify.com/v1/me',
        headers={'Authorization': f'Bearer {access_token}'}
    ).json()
    username = user_data.get('display_name') or user_data.get('id')

    # Get top tracks quickly (just the basic info)
    tracks_response = requests.get(
        'https://api.spotify.com/v1/me/top/tracks?limit=5',
        headers={'Authorization': f'Bearer {access_token}'}
    )

    if tracks_response.status_code != 200:
        return "❌ Failed to fetch top tracks"

    items = tracks_response.json().get("items", [])
    tracks = []

    # Only get basic track info first - no artist details yet
    for track in items:
        name = track["name"]
        artist = track["artists"][0]["name"]
        artist_id = track["artists"][0]["id"]
        spotify_id = track["id"]
        cover = track["album"]["images"][0]["url"] if track["album"]["images"] else ""

        tracks.append({
            "name": name,
            "artist": artist,
            "artistId": artist_id,  # Store for later processing
            "spotifyId": spotify_id,
            "coverPicture": cover,
            "genres": []  # Will be filled during async processing
        })

    # Save basic track data immediately
    data = {
        "username": username,
        "access_token": access_token,
        "tracks": tracks
    }

    tracks_path = os.path.join(SHARED_DIR, f"tracks_{username}.json")
    with open(tracks_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Initialize processing status
    processing_status[username] = {"status": "starting", "progress": 0}

    # Start async processing in background thread
    thread = threading.Thread(target=process_user_data_async, args=(username, access_token))
    thread.daemon = True
    thread.start()

    # Return immediately with a better UI
    return f"""
        <html>
            <head>
                <title>Processing {username}</title>
                <meta http-equiv="refresh" content="3;url=/status/{username}">
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; }}
                    .container {{ max-width: 600px; margin: 0 auto; text-align: center; }}
                    .spinner {{ border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 20px auto; }}
                    @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
                    .links {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-top: 20px; }}
                    .links a {{ color: #fff; text-decoration: none; display: block; margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.2); border-radius: 5px; }}
                    .links a:hover {{ background: rgba(255,255,255,0.3); }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>🎵 Welcome {username}!</h2>
                    <div class="spinner"></div>
                    <p>✅ Your top 5 tracks have been loaded!</p>
                    <p>🔄 We're now analyzing your music in the background...</p>
                    <div class="links">
                        <p><strong>Available now:</strong></p>
                        <a href="/tracks/{username}">🎧 View Your Top Tracks</a>
                        <a href="/status/{username}">📊 Check Processing Status</a>
                        <p style="margin-top: 20px; font-size: 0.9em;">Audio feature analysis is running in the background. You'll be redirected to the status page in 3 seconds.</p>
                    </div>
                </div>
            </body>
        </html>
    """

def process_user_data_async(username, access_token):
    """Background processing of genres and audio features"""
    try:
        processing_status[username] = {"status": "processing", "step": "fetching_genres", "progress": 10}
        
        # Load the basic track data
        tracks_path = os.path.join(SHARED_DIR, f"tracks_{username}.json")
        with open(tracks_path, encoding="utf-8") as f:
            data = json.load(f)

        headers = {'Authorization': f'Bearer {access_token}'}
        
        # Get artist genres
        total_tracks = len(data["tracks"])
        for i, track in enumerate(data["tracks"]):
            artist_id = track.get("artistId")
            if artist_id:
                artist_response = requests.get(
                    f'https://api.spotify.com/v1/artists/{artist_id}',
                    headers=headers
                )
                if artist_response.status_code == 200:
                    track["genres"] = artist_response.json().get("genres", [])
            
            processing_status[username]["progress"] = 10 + int((i / total_tracks) * 20)  # 10-30%

        # Save updated track data with genres
        with open(tracks_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        processing_status[username]["step"] = "analyzing_audio"
        processing_status[username]["progress"] = 30

        # Now do the heavy audio processing
        analyze_user_async(username)
        
    except Exception as e:
        processing_status[username] = {"status": "error", "message": str(e)}

@app.route('/status/<username>')
def show_status(username):
    status = processing_status.get(username, {"status": "not_found"})
    
    if status["status"] == "completed":
        return f"""
            <html>
                <head>
                    <title>Analysis Complete - {username}</title>
                    <style>
                        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; min-height: 100vh; }}
                        .container {{ max-width: 600px; margin: 0 auto; text-align: center; }}
                        .success {{ font-size: 4em; margin: 20px 0; }}
                        .links {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-top: 20px; }}
                        .links a {{ color: #fff; text-decoration: none; display: block; margin: 10px 0; padding: 15px; background: rgba(255,255,255,0.2); border-radius: 5px; font-weight: bold; }}
                        .links a:hover {{ background: rgba(255,255,255,0.3); }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="success">✅</div>
                        <h2>Analysis Complete!</h2>
                        <p>All your music data has been processed and is ready to view.</p>
                        <div class="links">
                            <a href="/tracks/{username}">🎧 Raw Tracks Data</a>
                            <a href="/features/{username}">🧠 Extracted Audio Features</a>
                            <a href="/final/{username}">📦 Complete JSON Dataset</a>
                        </div>
                    </div>
                </body>
            </html>
        """
    elif status["status"] == "error":
        return f"""
            <html>
                <head><title>Error - {username}</title></head>
                <body style="font-family: sans-serif; padding: 40px; background: #ff6b6b; color: white;">
                    <h2>❌ Processing Error</h2>
                    <p>Error: {status.get('message', 'Unknown error')}</p>
                    <a href="/" style="color: white;">← Try Again</a>
                </body>
            </html>
        """
    else:
        progress = status.get("progress", 0)
        step = status.get("step", "starting")
        return f"""
            <html>
                <head>
                    <title>Processing {username}</title>
                    <meta http-equiv="refresh" content="5">
                    <style>
                        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; }}
                        .container {{ max-width: 600px; margin: 0 auto; text-align: center; }}
                        .progress-bar {{ width: 100%; height: 30px; background: rgba(255,255,255,0.2); border-radius: 15px; overflow: hidden; margin: 20px 0; }}
                        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #4CAF50, #45a049); transition: width 0.3s ease; }}
                        .spinner {{ border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 20px auto; }}
                        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
                        .links {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-top: 20px; }}
                        .links a {{ color: #fff; text-decoration: none; display: block; margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.2); border-radius: 5px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h2>🎵 Processing {username}'s Music</h2>
                        <div class="spinner"></div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {progress}%"></div>
                        </div>
                        <p>Step: {step.replace('_', ' ').title()}</p>
                        <p>Progress: {progress}% ({processing_status.get(username, {}).get('step', '').replace('_', ' ').title()})</p>
                        <div class="links">
                            <p><strong>Available now:</strong></p>
                            <a href="/tracks/{username}">🎧 View Your Tracks</a>
                            <p style="font-size: 0.9em; margin-top: 20px;">This page will refresh automatically every 5 seconds.</p>
                        </div>
                    </div>
                </body>
            </html>
        """

@app.route('/tracks/<username>')
def show_tracks(username):
    path = os.path.join(SHARED_DIR, f"tracks_{username}.json")
    if not os.path.exists(path):
        return jsonify({"error": "tracks file not found"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))

@app.route('/features/<username>')
def show_features(username):
    path = os.path.join(SHARED_DIR, f"features_{username}.json")
    if not os.path.exists(path):
        return jsonify({"error": "features file not ready - processing may still be in progress"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))

@app.route('/final/<username>')
def show_final(username):
    path = os.path.join(SHARED_DIR, f"final_{username}.json")
    if not os.path.exists(path):
        return jsonify({"error": "final data not ready - processing may still be in progress"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))

@app.route('/clear-cache')
def clear_cache():
    """Clear the audio features cache - useful for testing"""
    try:
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
        return "Cache cleared successfully", 200
    except Exception as e:
        return f"Error clearing cache: {e}", 500

@app.route('/cache-stats')
def cache_stats():
    """Show cache statistics"""
    try:
        cache_files = os.listdir(CACHE_DIR) if os.path.exists(CACHE_DIR) else []
        return jsonify({
            "cached_tracks": len(cache_files),
            "cache_size_mb": sum(os.path.getsize(os.path.join(CACHE_DIR, f)) 
                                for f in cache_files) / (1024*1024)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    print("🚀 Run: http://localhost:5000")
    print(f"🔧 Using {MAX_WORKERS} parallel workers for audio processing")
    print(f"💾 Cache directory: {CACHE_DIR}")
    app.run(host="0.0.0.0", port=5000, threaded=True)