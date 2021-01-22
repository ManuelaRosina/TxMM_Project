import jodel_api
import pickle
import os
import time
import datetime
import pandas as pd

file_recent = 'posts_recent3.csv'
file_popular = 'posts_popular3.csv'

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
    posts_popular = payload_p['posts']

    df_recent = pd.DataFrame(posts_recent)
    df_recent['collection Time'] = date
    df_popular = pd.DataFrame(posts_popular)
    df_popular['collection Time'] = date

    with open(file_recent, 'a') as f:
        df_recent.to_csv(f, header=f.tell() == 0)

    with open(file_popular, 'a') as f:
        df_popular.to_csv(f, header=f.tell() == 0)

        #time.sleep(300)
#else:
#    print(rep_p)
#    print(rep_r)
#    j = jodel_api.JodelAccount(lat=lat, lng=lng, city=city, is_legacy=False)
