import pandas as pd

import scripts.functions as functions


points = pd.read_csv('data/clean/points.psv', sep='|', index_col='id')
users = pd.read_csv('data/clean/users.psv', sep='|', index_col='id')
levels = json.load(open('data/levels.json', 'r'))['maps']

rects = {level['level']: functions.get_rect(level['polygon']) for level in levels}

for level_id in LEVEL_IDS:
    functions.plot_scatter(points, rects, level_id)
for level_id in LEVEL_IDS:
    functions.plot_heatmap(points, rects, level_id)
