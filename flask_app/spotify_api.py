import spotipy.util as util
import pandas as pd
import numpy as np
import requests
import pickle


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity



#Import scaler for new spotify track data
scaler = pickle.load(open('data/scaler_PCA.pickle', 'rb'))
#Import PCA model for 7 components
pca_model = pickle.load(open('data/pca_scaled7.pickle', 'rb'))

def get_spotify_info(q, token, type = 'album', query = 'id'):
    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }
    params = [
        ('q', q),
        ('type', type),
        ('market', 'US')
    ]
    json_key = str(type + 's')
    try: 
        response = requests.get('https://api.spotify.com/v1/search',
                           headers = headers, params = params, timeout = 5)
        json = response.json()
        first_result = json[json_key]['items'][0][query]
        return first_result
    except:
        return "None Found"

def get_tracklist(album_id, token):
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer ' + token,
    }

    if album_id == 'None Found':
        return 'No Album ID'
    else:
        get_url = 'https://api.spotify.com/v1/albums/' + album_id + '/tracks'
    
    response = requests.get(get_url, headers=headers, params=None)
    json = response.json()
    first_result = []
    album_id_list = []
    for i in json['items']:
        first_result.append(i['id'])
        album_id_list.append(album_id)
    return pd.DataFrame(list(zip(album_id_list, first_result)), columns = ['album_id', 'track_id'])

def get_features(track_list, token):
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer ' + token,
    }

    ids = ','.join(track_list)
    
    params = [
        ('ids', ids),
        ('market', 'US')
    ]
    
    response = requests.get('https://api.spotify.com/v1/audio-features/', headers=headers, params=params)
    json = response.json()
    return pd.DataFrame(json['audio_features'])

def scale_PCA(track_features):
	track_features.drop(['id', 'uri', 'track_href', 'analysis_url', 'type'], axis = 1, inplace = True)
	scaled = scaler.transform(track_features)
	pca_applied = pca_model.transform(scaled)
	return pca_applied.mean(axis = 0)




# This section checks that the code runs properly
# To run, type "spotify_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
# if __name__ == '__main__':
#     from pprint import pprint
#     print("Checking to see if connection to API works")
#     q= 'album:Love And Theft artist:Bob Dylan'
#     #print('q is')
#     #pprint(q)
#     first_result = get_album_id(q, token)
#     #print(f'Album ID: {first_result}')
#     tracks = get_tracklist(first_result, token)
#     #print(f'track_list: {tracks}')
#     feature_list = get_features([tracks.iloc[i, 1] for i in range(tracks.shape[0])], token)
#     #print(f'features:{feature_list}')
#     feature_means = scale_PCA(feature_list)
#     #print(f'feature_means:{feature_means}')
#  #   dist = cosine_similarity(feature_means.reshape(1, -1), components_by_album)[0]
#     #print(f'distances:{dist}')
#     lengthdist = len(dist)
#     print(f'number of comparisons:{lengthdist}')
#     match = np.argsort(dist)[::-1][:10]
#     print(f'index of closest matches:{match}')
# #    better = components_by_album.reset_index()
#     album_name = better.iloc[match[0], 0]
#     print(f'closest albums are: {better.iloc[match[0], 0]}, {better.iloc[match[1], 0]}, {better.iloc[match[2], 0]}, {better.iloc[match[3], 0]}, {better.iloc[match[4], 0]}') 
