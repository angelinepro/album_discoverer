# minimal example from:
# http://flask.pocoo.org/docs/quickstart/

from flask import Flask, request
import pickle
import flask
import numpy as np
from spotify_api import get_spotify_info, get_tracklist, get_features, scale_PCA
import spotipy.util as util
from sklearn.metrics.pairwise import cosine_similarity
from credentials import token
import re

#Import components for each album
components_by_album = pickle.load(open('data/components_by_album.pickle', 'rb'))
#Import metacritic album ratings
critics_df_all = pickle.load(open('data/critics_df_all.pickle', 'rb'))

app = Flask(__name__)  # create instance of Flask class

@app.route('/home')
def index_home() -> str:
    return flask.render_template('home.html')

@app.route('/get_info',  methods = ["POST", "GET"])
def get_info():
	return flask.render_template('info.html')


@app.route('/recommend_api', methods = ["POST", "GET"])
def result():
	result = request.form
	q = 'album:' + str(result['album']) + ' artist:' + str(result['artist'])
	album_id = get_spotify_info(q = q, token = token)
	tracks = get_tracklist(album_id, token)
	list_tracks = [tracks.iloc[i, 1] for i in range(tracks.shape[0])]
	list_features = get_features([tracks.iloc[i, 1] for i in range(tracks.shape[0])], token)
	feature_means = scale_PCA(list_features)
	match = np.argsort(cosine_similarity(feature_means.reshape(1, -1), components_by_album)[0])[::-1][:10]
	better = components_by_album.reset_index()
	if better.iloc[match[0], 0] == q:
		album_rec = better.iloc[match[1:6], 0].values
	else:
		album_rec = better.iloc[match[:5], 0].values
	album_recs = []
	artist_recs = []
	for row in album_rec:
		album_recs.append(re.search('^.*?:(.*?) artist:(.+)', str(row)).group(1))
		artist_recs.append(re.search('^.*?:(.*?) artist:(.+)', str(row)).group(2))
	recs_df = list(zip(artist_recs, album_recs))
	
	return flask.render_template('recommend_api.html',
		q = q, 
		result = result,
		album_id = album_id, 
		list_tracks = list_tracks,
		match = match,
		album_rec = album_rec,
		album_recs = album_recs,
		artist_recs = artist_recs,
		recs_df = recs_df)

@app.route('/top_albums', methods = ["POST", "GET"])
def top_100():
	column_names = ['Album Title', 'Artist Name', 'Critic Rating', 'User Rating', 'Release Date']

	return flask.render_template('top_albums.html',
		column_names = column_names,
		row_data = list(critics_df_all.iloc[:100, :].values.tolist()),
		zip = zip

		)
@app.errorhandler(404)
def not_found_error(error):
    return flask.render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
#    db.session.rollback()
    return flask.render_template('500.html'), 500

if __name__ == '__main__':
    app.run()
