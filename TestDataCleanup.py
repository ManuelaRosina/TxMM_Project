import pickle
import datetime
from dateutil import tz
import pandas as pd
import numpy as np
import json

def convert_loc(locationdict):
    dictionary = json.loads(locationdict.replace("'","\""))
    return dictionary['name']

data = pd.read_csv('Data/test_data.csv')
train_data = pd.read_csv('Data/train_data_cleaned.csv')
ids = list(train_data['post_id'])
overlap = data[data['post_id'].isin(ids)]
data = data[~data['post_id'].isin(ids)]

print(data.info())
data['collection Time'] = pd.to_datetime(data['collection Time'])
data['created_at'] = pd.to_datetime(data['created_at'])

overlap['collection Time'] = pd.to_datetime(overlap['collection Time'])
overlap['created_at'] = pd.to_datetime(overlap['created_at'])

# image_approved, replier, discovered_by, postblock, got_thanks, promoted, was_promoted
# promoted_type, cta_type, user_handle, view_count, post_own, children, (post_type?)
data = data.drop(['post_type', 'channel', 'share_count', 'pin_count', 'color', 'from_home', 'distance', 'image_approved', 'replier', 'discovered_by', 'postblock', 'got_thanks', 'promoted', 'was_promoted', 'promoted_type', 'cta_type', 'user_handle', 'view_count', 'post_own', 'children'], axis=1)
data['location'] = data['location'].map(lambda a: convert_loc(a))
data = data[data['location']!='Kleve']
data = data[data['location']!='Kranenburg']

overlap = overlap.drop(['post_type', 'channel', 'share_count', 'pin_count', 'color', 'from_home', 'distance', 'image_approved', 'replier', 'discovered_by', 'postblock', 'got_thanks', 'promoted', 'was_promoted', 'promoted_type', 'cta_type', 'user_handle', 'view_count', 'post_own', 'children'], axis=1)
overlap['location'] = overlap['location'].map(lambda a: convert_loc(a))
overlap = overlap[overlap['location']!='Kleve']
overlap = overlap[overlap['location']!='Kranenburg']

#data['channel'] = data['channel'].astype(str)
#print(set(data['channel']))

"""posts_data = data.groupby(['post_id']).last()
posts_data['channel'] = posts_data['channel'].astype(str)
#print(data['post_id'].value_counts())
print(posts_data[['message', 'vote_count']])
print()
print(set(posts_data['channel']))
"""

print(data.info())

# get last collection time of each post
posts_group = data.groupby(['post_id'])['collection Time'].transform(max) == data['collection Time']
cleaned_posts = data[posts_group]
print(cleaned_posts.info())

#cleaned_posts.to_csv('Data/test_data_cleaned.csv', index=False)

posts_group_o = overlap.groupby(['post_id'])['collection Time'].transform(max) == overlap['collection Time']
cleaned_posts_o = overlap[posts_group_o]
print(cleaned_posts_o.info())

#cleaned_posts_o.to_csv('Data/overlap_data_cleaned.csv', index=False)
ids = list(overlap['post_id'])
comp = train_data[train_data['post_id'].isin(ids)]
comp.to_csv('Data/train_overlap.csv', index=False)
old_vote = list(comp['vote_count'])
new_vote = list(overlap['vote_count'])
print(old_vote)
print(new_vote)
#diff = [i == j for i, j in zip(list(comp['vote_count']), list(overlap['vote_count']))]

for i, v in enumerate(old_vote):
    if v == new_vote[i]:
        print("index: ", i)
        print("new: ", overlap['vote_count'].iloc[[i]])
        print("old: ", comp['vote_count'].iloc[[i]])
        print()