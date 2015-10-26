import json

import pandas as pd

import functions


TRUTH_ID = int(open('../data/truth_ID').read())
EXPERT_ID = int(open('../data/expert_ID').read())

points = pd.read_csv('../data/clean/points.psv', sep='|', index_col='id')
users = pd.read_csv('../data/clean/users.psv', sep='|', index_col='id')
levels = json.load(open('../data/levels.json', 'r'))['maps']
rects = {level['level']: functions.get_rect(level['polygon']) for level in levels}

points = points[~points.user_id.isin([TRUTH_ID, EXPERT_ID])].reset_index(drop=True)

for level_id in LEVEL_IDS:
    functions.plot_scatter(points, rects, level_id, grid_area=None)
for level_id in LEVEL_IDS:
    functions.plot_heatmap(points, rects, level_id, grid_area=192 if level_id != 6 else 384)
