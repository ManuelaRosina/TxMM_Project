import os
import datetime
import pickle
import pandas as pd
from jodel_api import jodel_api

file_recent = 'posts_recent.pkl'
file_popular = 'posts_popular.pkl'

if os.path.isfile(file_recent):
    with open(file_recent, 'rb') as f:
        data_recent = pickle.load(f)
else:
    data_recent = []

if os.path.isfile(file_popular):
    with open(file_popular, 'rb') as f:
        data_popular = pickle.load(f)
else:
    data_popular = []

url = "file:///home/manuela/Git_Repos/JodelJS/dist/index.html"
lat, lng, city = 51.833, 5.849, "Nijmegen"
print(jodel_api.JodelAccount.device_uid)
j = jodel_api.JodelAccount(lat=lat, lng=lng, city=city, is_legacy=False)
print(j.get_account_data())

date = datetime.datetime.now()
print(date)

rep_r, payload_r = j.get_posts_recent(skip=0, limit=100, after=None, mine=False, hashtag=None, channel=None)
rep_p, payload_p = j.get_posts_popular(skip=0, limit=100, after=None, mine=False, hashtag=None, channel=None)
if rep_r == 200 and rep_p == 200:
    print('Number r: ' + str(len(payload_r['posts'])))
    print('Number p: ' + str(len(payload_p['posts'])))
    print(payload_r['posts'][0])

    posts_recent = payload_r['posts']

    for post in posts_recent:
        post['collection Time'] = date
    data_recent.extend(posts_recent)

    posts_popular = payload_p['posts']
    for post in posts_popular:
        post['collection Time'] = date
    data_popular.extend(posts_popular)

    with open(file_recent, 'wb+') as f:
        pickle.dump(data_recent, f, 0)
    with open(file_popular, 'wb+') as f:
        pickle.dump(data_popular, f, 0)

    """df_recent = pd.DataFrame(posts_recent)
    df_recent['collection Time'] = date

    if not set(['image_url', 'image_headers', 'thumbnail_url']).issubset(df_recent.columns):
        df_recent['image_url'] = float('NaN')
        df_recent['image_headers'] = float('NaN')
        df_recent['thumbnail_url'] = float('NaN')

    if not set(['channel']).issubset(df_recent.columns):
        df_recent['channel'] = float('NaN')

    if not set(['poll_options', 'poll_votes', 'poll_id']).issubset(df_recent.columns):
        df_recent['poll_options'] = float('NaN')
        df_recent['poll_votes'] = float('NaN')
        df_recent['poll_id'] = float('NaN')


    df_popular = pd.DataFrame(posts_popular)
    df_popular['collection Time'] = date

    if not set(['image_url', 'image_headers', 'thumbnail_url']).issubset(df_popular.columns):
        df_popular['image_url'] = float('NaN')
        df_popular['image_headers'] = float('NaN')
        df_popular['thumbnail_url'] = float('NaN')

    if not set(['channel']).issubset(df_popular.columns):
        df_popular['channel'] = float('NaN')

    if not set(['poll_options', 'poll_votes', 'poll_id']).issubset(df_popular.columns):
        df_popular['poll_options'] = float('NaN')
        df_popular['poll_votes'] = float('NaN')
        df_popular['poll_id'] = float('NaN')

    with open(file_recent, 'a') as f:
        df_recent.to_csv(f, index=False, header=f.tell()==0)

    with open(file_popular, 'a') as f:
        df_popular.to_csv(f, index=False, header=f.tell()==0)"""

        #time.sleep(300)
#else:
#    print(rep_p)
#    print(rep_r)
#    j = jodel_api.JodelAccount(lat=lat, lng=lng, city=city, is_legacy=False)
