import spotipy
import pandas as pd
import math
import json
import constants

class SpotifyService:
    def __init__(self):
        self.instance = self.authenticate_and_instantiate()

    def authenticate_and_instantiate(self):
        token = spotipy.util.prompt_for_user_token(
            username='12165639406', 
            scope='user-library-read',
            client_id='a5646fd64ee74edabc2781812067f237',
            client_secret='f2633dc8c0c241c0a93b6688edaabdc6',
            redirect_uri='http://localhost:8888/callback/'
        )

        return spotipy.Spotify(auth=token, requests_timeout=20)

    def get_track_ids_from_playlist(self, playlist_url):
        """
            Get the track ids from a playlist
            Arguments:
                playlist_id: playlist id
            Return:
                List containing the track ids from the playlist received by param
        """
        try:
            playlist_id = playlist_url[34:56]
            tracks = self.instance.playlist_tracks(playlist_id)
            return [i['track']['id'] for i in tracks['items']]
        except:
            print('failed to get track_ids from playlist: {}'.format(playlist_id))

    def get_tracks_audio_features(self, track_list):
        """
            Get the track audio features from a playlist
            Arguments:
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
            features = self.instance.audio_features(tracks=request_ids)
            track_features_list += features
        
        track_features_df = pd.DataFrame(track_features_list)
        return track_features_df[constants.AUDIO_FEATURES_PROPERTIES]

    def get_track_feature_from_user_playlist(self, playlistUrl):
        track_ids = self.get_track_ids_from_playlist(playlistUrl)
        features = self.get_tracks_audio_features(track_ids)
        features['id'] = track_ids
        return pd.DataFrame(features)

    def export_track_features_to_csv(self):
        """
            Export the audio features from the playlists to a csv file
        """
        df = pd.DataFrame()
        file = open(constants.PLAYLISTS_FILE_PATH)
        playlists = json.load(file)

        for mood, urls in playlists.items():
            for url in urls:
                track_ids = self.get_track_ids_from_playlist(url)
                features = self.get_tracks_audio_features(track_ids)
                features['id'] = track_ids
                features['mood'] = mood
                df = pd.concat([df, features])
        
        df.to_csv(constants.DATASET_FILE_NAME)