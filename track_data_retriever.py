import pprint
import re
import time
import pickle
import os.path
import argparse as ap
import spotipy.util as util
import sys
import spotipy
import random

USER_ID = 12100162381
MAX_ID_COUNT = 50
DICT_FILE = 'dict_large.pkl'
ARTIST_GENRE_FILE = 'artist_genre.pkl'
ARTIST_BLACKLIST_FILE = 'artist_blacklist.pkl'
TEST_DATA_FILE = 'NN/test_data.dat'
TRAIN_DATA_FILE = 'NN/train_data.dat'
ALL_TRACK_DATA_FILE = 'all_track_data.pkl'
PLAYLIST_ID = '3ak5YjddmeNcRrnTpypzv9'
APP_CLIENT_ID = "97fad7e7f7f44d6daabac8473b12809e"
APP_SECRET_KEY = "840212a600484c27959867cbc92a9643"
REDIRECT_URI = "http://google.com"

DEF_KEY_FILE = "NN/genres.key"
DEF_FEAT_FILE = "NN/features.key"


#removes all non alpha characters from the list
def clean_list(items):
    items = [re.sub('[^a-zA-Z]+', '', str(x)) for x in items]
    return items

#assigns an artist to the genres that we have already defined
def get_genre(genres, genre_ids, is_multiclass=False):
    genre_list = {x:{'aliases':genre_ids[x], 'count': 0} for x in genre_ids}

    for genre in genres:
        for item in genre_list:
            for alias in genre_list[item]['aliases']:
                if alias in genre:
                    genre_list[item]['count'] += 1
                    break
    temp = {}

    if is_multiclass:
        for genre in genre_list:
            if genre_list[genre]['count'] > 0:
                    if len(temp) == 3:
                        should_add = (len(temp) == 0)
                        remove_id = -1
                        for item in temp:
                            if genre_list[genre]['count'] > temp[item]['count']:
                                should_add = True
                                if remove_id != -1:
                                     if temp[remove_id]['count'] > temp[item]['count']:
                                        remove_id = item
                                else:
                                    remove_id = item

                        if should_add:
                            if remove_id != -1:
                                del temp[remove_id]
                            temp[genre] = genre_list[genre]
                    else:
                        temp[genre] = genre_list[genre]
    else:
        maxval = max(genre_list.items(), key=lambda x: x[1]['count'])
        if maxval[1] != 0:
            temp[maxval[0]] = maxval[1]

    # print "Genre List: "
    # pp.pprint(genre_list)
    #
    # print "\nTemp: "
    # pp.pprint(temp)

    if len(temp) == 0:
        return -1
    else:
        return temp.keys()



def get_tracks(sp, playlist_id, genre_ids):
    all_track_data = {}
    category_dict = sp.categories(limit=50)

    category_ids = [x["id"] for x in category_dict['categories']['items']]

    counter = 1
    total_count = len(category_ids)
    for category_id in category_ids:
        print str(counter) + "/" + str(total_count) + ' ID: ' + category_id
        results = sp.category_playlists(category_id, limit=50)
        results = results['playlists']
        category_playlists_list = results['items']
        while results['next']:
            results = sp.next(results)
            results = results['playlists']
            category_playlists_list.extend(results['items'])

        print "Playlist Count: " + str(len(category_playlists_list)) + "\n"

        for playlist in category_playlists_list:
            print "\tPlaylist: " + playlist['name']
            playlist_id = playlist['id']
            playlist_owner = playlist['owner']['id']

            results = sp.user_playlist(playlist_owner, playlist_id)
            results = results['tracks']
            playlist_songs = results['items']

            while results['next']:
                results = sp.next(results)
                playlist_songs.extend(results['items'])

            print "\tTrack Count: " + str(len(all_track_data)) + "\n"
            artist_ids = []
            for playlist_track in playlist_songs:
                if playlist_track['track'] is None:
                    continue
                if playlist_track['track']['id'] not in all_track_data:
                    new_track = {
                        'name': playlist_track['track']['name'],
                        'artist_id': playlist_track['track']['artists'][0]['id'],
                        'artist_name': playlist_track['track']['artists'][0]['name']
                    }

                    all_track_data[playlist_track['track']['id']] = new_track
        counter += 1
    with open(DICT_FILE, 'wb') as f:
        pickle.dump(all_track_data, f, pickle.HIGHEST_PROTOCOL)
    return all_track_data

def load_keys(key_file):
    f = open(key_file, 'r')
    lines = f.readlines()
    retval = {}
    for line in lines:
        splitline = line.split()
        retval[int(splitline[0])] = splitline[1:]
    return retval

def get_artist_data(all_track_data, genre_ids, is_multiclass=False):
    is_running = True

    artist_ids = []
    [artist_ids.append(all_track_data[x]['artist_id']) for x in all_track_data if all_track_data[x]['artist_id'] not in artist_ids and all_track_data[x]['artist_id'] is not None]

    artist_blacklist = []
    artist_genre_cache = {}

    while is_running:
        print "Getting artist info, " + str(len(artist_ids)) + " to go"
        if len(artist_ids) <= MAX_ID_COUNT:
            artists_info = sp.artists(artist_ids)
            is_running = False
        else:
            artists_info = sp.artists(artist_ids[0:MAX_ID_COUNT])

        for artist in artists_info['artists']:
            artist_genres = clean_list(artist["genres"])
            genre = get_genre(artist_genres, genre_ids, is_multiclass)

            if genre != -1:
                print_statement = "\t" + artist['name'] + " genre(s):"
                for genre_id in genre:
                    print_statement += " " + genre_ids[genre_id][0]
                print print_statement
                artist_genre_cache[artist['id']] = genre
            else:
                print "\t" + artist['name'] + ' genre not found, blacklisting...'
                artist_blacklist.append(artist['id'])

        if len(artist_ids) > MAX_ID_COUNT:
            del artist_ids[:MAX_ID_COUNT]

    with open(ARTIST_GENRE_FILE, 'wb') as f:
        pickle.dump(artist_genre_cache, f, pickle.HIGHEST_PROTOCOL)

    with open(ARTIST_BLACKLIST_FILE, 'wb') as f:
        pickle.dump(artist_blacklist, f, pickle.HIGHEST_PROTOCOL)

    return artist_genre_cache, artist_blacklist

def get_track_features(all_track_data, feature_list):
    song_ids = all_track_data.keys()

    is_running = True
    while is_running:
        print "Getting track features, " + str(len(song_ids)) + " to go"
        if len(song_ids) < MAX_ID_COUNT:
            tracks_features = sp.audio_features(song_ids)
            is_running = False
        else:
            tracks_features = sp.audio_features(song_ids[0:MAX_ID_COUNT])


        for features in tracks_features:
            if features is None:
                continue
            for feature_name in feature_list.values():
                all_track_data[features['id']][feature_name[0]] = features[feature_name[0]]
            # all_track_data[features['id']]['danceability'] = features['danceability']
            # all_track_data[features['id']]['energy'] = features['energy']
            # all_track_data[features['id']]['instrumentalness'] = features['instrumentalness']
            # all_track_data[features['id']]['key'] = features['key']
            # all_track_data[features['id']]['loudness'] = features['loudness']
            # all_track_data[features['id']]['mode'] = features['mode']
            # all_track_data[features['id']]['speechiness'] = features['speechiness']
            # all_track_data[features['id']]['tempo'] = features['tempo']
            # all_track_data[features['id']]['time_signature'] = features['time_signature']
            # all_track_data[features['id']]['valence'] = features['valence']
            # all_track_data[features['id']]['acousticness'] = features['acousticness']

        if(len(song_ids) > MAX_ID_COUNT):
            del song_ids[:MAX_ID_COUNT]

def write_to_file(all_track_data, train_filename, test_filename, feature_list):
    print "Writing to file..."

    f_test = open(test_filename, 'w')
    f_train = open(train_filename, 'w')

    for track in all_track_data.keys():
        if feature_list[feature_list.keys()[0]][0] not in all_track_data[track]:
            print "Skipping " + all_track_data[track]['name']
            del all_track_data[track]
            continue
        line = ''
        for feature_name in feature_list.values():
            line += str(all_track_data[track][feature_name[0]]) + ' '

        for g in all_track_data[track]['genre']:
            line += str(g) + ' '

        line += "\n"

        if random.randint(0,3) == 0:
            f_test.write(line)
        else:
            f_train.write(line)

    f_test.close()
    f_train.close()

def bind_genres_to_tracks(all_track_data, genre_ids, artist_blacklist):
    genre_count = {id:0 for id in genre_ids}

    for track in all_track_data.keys():
        if all_track_data[track]['artist_id'] is None or all_track_data[track]['artist_id'] in artist_blacklist or all_track_data[track]['artist_id'] not in artist_genre_cache:
            del all_track_data[track]
        else:
            all_track_data[track]['genre'] = artist_genre_cache[all_track_data[track]['artist_id']]
            for genre in all_track_data[track]['genre']:
                genre_count[genre] += 1
    return genre_count

def equalize(all_track_data, genre_count):
    min_genre = min(genre_count, key=genre_count.get)
    min_genre_count = genre_count[min_genre]
    print "min genre: " + min_genre + ", with " + str(min_genre_count)
    for track in all_track_data.keys():
        should_break = True
        if genre_count[all_track_data[track]['genre'][0]] > min_genre_count:
            genre_count[all_track_data[track]['genre'][0]] -= 1
            del all_track_data[track]
            for genre in genre_count:
                if genre_count[genre] != min_genre_count:
                    should_break = False
                    break
            if should_break:
                break

def normalize(all_track_data, features, newmax, newmin):
    feature_vars = {val[0]:{'min': sys.maxint, "max":(-sys.maxint - 1)} for val in features.values()}
    count = 1
    for track in all_track_data.keys():
        print str(count) + '/' + str(len(all_track_data)) + ' Normalizing ' + all_track_data[track]['name']
        count += 1
        for feature in feature_vars:
            if track not in all_track_data.keys():
                continue
            if feature not in all_track_data[track] or all_track_data[track][feature] is None:
                del all_track_data[track]
                continue

            if all_track_data[track][feature] > feature_vars[feature]['max']:
                feature_vars[feature]['max'] = all_track_data[track][feature]
            if all_track_data[track][feature] < feature_vars[feature]['min']:
                feature_vars[feature]['min'] = all_track_data[track][feature]

    for track in all_track_data.keys():
        for feature in feature_vars:
            minval = feature_vars[feature]['min']
            maxval = feature_vars[feature]['max']
            all_track_data[track][feature] = (((all_track_data[track][feature] - minval) * (newmax - newmin)) / (maxval - minval)) + newmin



#Start of main method
if __name__ == "__main__":
    #parse arguments
    start_time = time.time()
    parser = ap.ArgumentParser(description="Retrieve data from the Spotify API regarding tracks and their genres")
    parser.add_argument("--keys", default=DEF_KEY_FILE,  help='The filename to retrieve genre keys from')
    parser.add_argument("--features",default=DEF_FEAT_FILE, help='The filename to load the list of features from')
    parser.add_argument("--multiclass", action='store_true', help='Retrieve multiple genres for each track')
    parser.add_argument("--data", default=ALL_TRACK_DATA_FILE,  help='The .pkl file containing preretrieved data about the tracks')
    parser.add_argument("--outtrain", default=TEST_DATA_FILE,  help="The filename to write the train data out to")
    parser.add_argument("--outtest", default=TRAIN_DATA_FILE,  help="The filename to write the test data out to")
    parser.add_argument("--equalize", action='store_true', help="Enable equalizing of test data (all classes equally represented, only available if multiclass is not enabled)")
    args = parser.parse_args()

    #setup spotipy
    token = util.prompt_for_user_token(str(USER_ID), scope='playlist-modify-public', client_id=APP_CLIENT_ID, client_secret=APP_SECRET_KEY, redirect_uri=REDIRECT_URI)
    sp = spotipy.Spotify(auth=token)

    #load keys from file
    keys = load_keys(args.keys)

    try:
        with open(DICT_FILE, 'rb') as f:
            all_track_data = pickle.load(f)
    except:
        all_track_data = get_tracks(sp, PLAYLIST_ID, keys)

    print "Total Track count before genre retrieval: " + str(len(all_track_data))

    try:
        with open(ARTIST_GENRE_FILE, 'rb') as f:
            artist_genre_cache = pickle.load(f)
        with open(ARTIST_BLACKLIST_FILE, 'rb') as f:
            artist_blacklist = pickle.load(f)
    except:
        artist_genre_cache, artist_blacklist = get_artist_data(all_track_data, keys, args.multiclass)

    genre_count = bind_genres_to_tracks(all_track_data, keys, artist_blacklist)

    print genre_count

    if args.equalize and not args.multiclass:
        equalize(all_track_data, genre_count)

    features = load_keys(args.features)

    get_track_features(all_track_data, features)

    normalize(all_track_data, features, 1, 0)

    write_to_file(all_track_data, args.outtrain, args.outtest, features)

    print "Elapsed time: " + str(time.time() - start_time)
    print 'Track Count: ' + str(len(all_track_data))
    for key in genre_count:
        print "\t" + keys[key][0] + ': ' + str(genre_count[key])
