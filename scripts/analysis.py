import json

import pandas as pd
from scipy.stats import gaussian_kde
import scripts.functions as functions


LEVEL_IDS = list(range(1, 7))

NUM_BINS = 10
COL_POINTS = 'steelblue'
COL_EXPERT = 'orange'
LW_EXPERT = 4
TITLE = 'Distribusi jarak pemain'
XLABEL = 'Jarak'
YLABEL = 'Jumlah pemain'
IQR = 2.0


def dist(p_ref, points):
	""" Compute distance between a reference point and a set of points.

		p_ref  : (float, float)
			lat, lng coordinates of the reference point
		points : pd.DataFrame with (lat, lng) columns
			Points to be computed

		Returns : pd.Series with same index as points
			Distance between points and p_ref
	"""
	return ((points.lat - p_ref[0])**2 + (points.lng - p_ref[1])**2)**0.5


TRUTH_ID = int(open('data/truth_ID').read())
EXPERT_ID = int(open('data/expert_ID').read())

points = pd.read_csv('data/clean/points.psv', sep='|', index_col='id')
users = pd.read_csv('data/clean/users.psv', sep='|', index_col='id')
levels = json.load(open('data/levels.json', 'r'))['maps']
rects = {level['level']: functions.get_rect(level['polygon']) for level in levels}

points_truth = points[points.user_id == TRUTH_ID].reset_index(drop=True)
points_expert = points[points.user_id == EXPERT_ID].reset_index(drop=True)
points = points[~points.user_id.isin([TRUTH_ID, EXPERT_ID])].reset_index(drop=True)


def get_dist_level(points_ref, points_expert, points, level_id):
	p_ref = points_ref[points_ref.level == level_id]
	p_ref = (p_ref.lat.iloc[0], p_ref.lng.iloc[0])
	p_expert = points_expert[points_expert.level == level_id]
	ps = points[points.level == level_id]
	return dist(p_ref, ps), dist(p_ref, p_expert).iloc[0]

dists = {level_id: get_dist_level(points_truth, points_expert, points, level_id) for level_id in LEVEL_IDS}
dists_points = {level_id: d[0] for level_id, d in dists.items()}
dists_expert = {level_id: d[1] for level_id, d in dists.items()}

def plot_level(dists_points, dists_expert, level_id, is_hist=True, bw=None, num_bins=NUM_BINS, num_iqr=None,
	col_points=COL_POINTS, col_expert=COL_EXPERT, lw_expert=LW_EXPERT,
	title=None, xlabel=XLABEL, ylabel=None):
	def filter_iqr(X, num_iqr):
		median = X.median()
		iqr = X.quantile(0.75) - X.quantile(0.25)
		return X[(X >= median - iqr*num_iqr) & (X <= median + iqr*num_iqr)]

	dist_points = dists_points[level_id]
	dist_expert = dists_expert[level_id]

	if num_iqr:
		dist_points = filter_iqr(dist_points, num_iqr)

	if is_hist:
		ax = dist_points.plot.hist(bins=num_bins, color=col_points)
	else:
		if bw:
			ax = sns.kdeplot(dist_points, bw=bw, color=col_points)
		else:
			ax = sns.kdeplot(dist_points, color=col_points)

	plt.axvline(dist_expert, 0, len(dist_points), color=col_expert, lw=LW_EXPERT)
	if not title:
		plt.title('Distribusi jarak pemain - Level %s' % level_id)
	plt.xlabel(xlabel)
	if is_hist and (not ylabel):
		plt.ylabel("Jumlah pemain")
	elif is_hist and (not ylabel):
		plt.ylabel("Distribusi")
	elif ylabel:
		plt.ylabel(ylabel)
	return ax
