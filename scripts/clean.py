import json

import pandas as pd

import functions


### Constants
LEVEL_IDS = list(range(1, 7))


### Read data
points = pd.read_csv('../data/raw/points.psv', sep='|', index_col='id')
users = pd.read_csv('../data/raw/users.psv', sep='|', index_col='id')

### Merge data
points.index = points['id']
users.index = users['id']
del points['id']
del users['id']

### Clean data
## Points
# Sort points by user_id and timestamp
points.sort_values(by=['user_id', 'timestamp'], inplace=True)
points['timestamp'] = pd.to_datetime(points['timestamp'])

# Remove duplicated (user_id, level) and take the last
points = points[~points.duplicated(['user_id', 'level'], keep='last')]

# Filter only for users that completed the game (# plays == 6)
user_play_counts = points.groupby('user_id')['level'].nunique()
users_lt_6_plays = user_play_counts[user_play_counts < 6].index
points = points[~points['user_id'].isin(users_lt_6_plays)]
assert all(points.groupby('user_id').size() == 6)


## Users
# Remove users who didn't play or didn't play all levels
users = users[~users.index.isin(users_lt_6_plays)]
users = users[users.index.isin(points['user_id'].unique())]

# Get OS from browser agent string
users['timestamp'] = pd.to_datetime(users['timestamp'])
users['OS'] = users['browser'].map(functions.browser_to_os)
users['OS_generic'] = users['OS'].map(functions.os_to_generic)
users['is_mobile'] = users['OS_generic'].map(functions.is_mobile)


### Write cleaned data
points.to_csv('../data/clean/points.psv', sep='|')
users.to_csv('../data/clean/users.psv', sep='|')
