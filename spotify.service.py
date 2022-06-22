import spotipy
import pandas as pd
import math
import json

PLAYLISTS_FILE_PATH = 'playlists.json'
AUDIO_FEATURES_PROPERTIES = [
    'acousticness', 
    'danceability', 
    'duration_ms',
    'energy',
    'instrumentalness',
    'liveness',
    'loudness',
    'speechiness',
    'tempo',
    'valence'
]

def authenticate_and_instantiate():
    token = spotipy.util.prompt_for_user_token(
        username='12165639406', 
        scope='user-library-read',
        client_id='a5646fd64ee74edabc2781812067f237',
        client_secret='f2633dc8c0c241c0a93b6688edaabdc6',
        redirect_uri='http://localhost:5000'
    )

    return spotipy.Spotify(auth=token, requests_timeout=20)

def get_track_ids_from_playlist(spotifyInstance, playlist_id):
    """
        Get the track ids from a playlist
        Arguments:
            spotifyInstance: instance of spotipy.Spotify
            playlist_id: playlist id
        Return:
            List containing the track ids from the playlist received by param
    """
    try:
        return spotifyInstance.playlist_tracks(playlist_id)
    except:
        print('failed to get track_ids from playlist: {}'.format(playlist_id))
    
def get_tracks_audio_features(spotifyInstance, track_list):
    """
        Get the track audio features from a playlist
        Arguments:
            spotifyInstance: instance of spotipy.Spotify
            track_list: list containing the track ids
        Return:
            Dataframe containing the track audio features from the playlist received by param
    """

    request_size = 50                                               # number of tracks per request
    request_num = int(math.ceil(len(track_list) / request_size))    # number of requests
    track_features_list = []

    for i in range(request_num):
        initial_index = i * request_size
        final_index = min(((i + 1) * request_size), len(track_list))
        request_ids = track_list[initial_index : final_index]
        features = spotifyInstance.audio_features(tracks=request_ids)
        track_features_list += features
    
    track_features_df = pd.DataFrame(track_features_list)
    # columns_to_remove = [i for i in track_features_df.columns if i not in AUDIO_FEATURES_PROPERTIES]
    # track_features_df.drop(columns_to_remove, index=1)
    return track_features_df[AUDIO_FEATURES_PROPERTIES]

def export_track_features_to_csv():
    """
        Export the audio features from the playlists to a csv file
    """
    df = pd.DataFrame()
    spotifyInstance = authenticate_and_instantiate()
    file = open(PLAYLISTS_FILE_PATH)
    playlists = json.load(file)

    for mood, urls in playlists.items():
        for url in urls:
            playlist_id = url[34:56]
            tracks = get_track_ids_from_playlist(spotifyInstance, playlist_id)
            track_ids = [i['track']['id'] for i in tracks['items']]
            features = get_tracks_audio_features(spotifyInstance, track_ids)
            features['id'] = track_ids
            features['mood'] = mood
            df = pd.concat([df, features])
    
    df.to_csv('track_features.csv')

export_track_features_to_csv()