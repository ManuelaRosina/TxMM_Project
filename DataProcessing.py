import os
import pickle
import datetime
from dateutil import tz
import pandas as pd
import matplotlib.pyplot
import matplotlib.dates
from pandas.plotting import register_matplotlib_converters


def convert_server_timestamp(timestr):
    #print(timestr)
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    converted = datetime.datetime.strptime(timestr, '%Y-%m-%dT%H:%M:%S.%fZ')
    converted = converted.replace(tzinfo=from_zone)
    converted = converted.astimezone(to_zone)
    converted = converted.replace(tzinfo=None)
    return converted

def convert_collection_timestamp(timestr):
    converted = datetime.datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S.%f')
    return converted

def delete_pictures(dataframe):
    # delete image posts
    dataframe = dataframe[dataframe['image_url'].isna()]
    dataframe = dataframe.drop(['image_url', 'image_headers', 'thumbnail_url', 'video_url'], 1)
    return dataframe

def delete_polls(dataframe):
    dataframe = dataframe[dataframe['poll_id'].isna()]
    dataframe = dataframe.drop(['poll_options', 'poll_votes', 'poll_id'], 1)
    return dataframe

# TODO: loop
data = []
for filename in sorted(os.listdir('Data/recent')):
    file = 'Data/recent/'+filename
    if os.path.isfile(file):
        with open(file, 'rb') as f:
            d = pickle.load(f)
        data.extend(d)
        print("File: " + filename + " was added")
data = pd.DataFrame(data)

"""file = '/home/manuela/Dropbox/Jodel_project/posts_recent.pkl'
with open(file, 'rb') as f:
    data2 = pickle.load(f)
data2 = pd.DataFrame(data2)
print("File: dropbox was added")

data = pd.concat([data, data2],ignore_index=True, sort=True)"""

#data = pd.read_csv('Data/train_data.csv')
data['collection Time'] = pd.to_datetime(data['collection Time'])
data['created_at'] = pd.to_datetime(data['created_at'])
#data_filtered = data[10001:] #data[:10000]
#data['created_at'] = data['created_at'].map(lambda a: convert_server_timestamp(a))
#data['updated_at'] = data['updated_at'].map(lambda a: convert_server_timestamp(a))

# delete polls and pictures
data_filtered = delete_pictures(data)
data_filtered = delete_polls(data_filtered)

print(data_filtered.info())
data_filtered.to_csv('Data/test_data.csv', index=False)
print(data_filtered.head())
#print(data_filtered[['message', 'vote_count', 'collection Time']])
"""posts_data = data_filtered.groupby(['post_id'])

groups = dict(list(posts_data))
keys = list(groups.keys())
register_matplotlib_converters()
print(sorted(set(data_filtered['collection Time']),reverse=True))
creation_times = []
last_upvote = []
for key in keys:
    #print(key)
    #print(groups[key][[ 'vote_count', 'collection Time']])
    creation_times.append(list(groups[key]['created_at'])[0].time())
    last_upvote.append(list(groups[key]['vote_count'])[-1])
    dates = matplotlib.dates.date2num(groups[key]['collection Time'])
    matplotlib.pyplot.plot_date(dates, groups[key]['vote_count'],'-')
matplotlib.pyplot.gcf().autofmt_xdate()
matplotlib.pyplot.show()
#print(creation_times)
matplotlib.pyplot.scatter(creation_times, last_upvote)
matplotlib.pyplot.gcf().autofmt_xdate()
matplotlib.pyplot.show()"""


