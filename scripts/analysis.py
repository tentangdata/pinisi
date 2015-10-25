import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy as scp
import seaborn as sns

import functions as functions


# Constants
LEVEL_IDS = list(range(1, 7))

ALPHA = 0.05
DIRECTION = 'unequal'
VALID_DIRECTIONS = ['unequal', 'greater', 'less']
NUM_BINS = 10
COL_POINTS = 'tan'
COL_EXPERT = 'saddlebrown'
LW_EXPERT = 4
TITLE = 'Distribusi jarak pemain'
XLABEL = 'Jarak'
YLABEL = 'Jumlah pemain'
NUM_IQR = 2.0

sns.set_style('white')

# Functions
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


def get_dist_level(points_ref, points_expert, points, level_id):
	p_ref = points_ref[points_ref.level == level_id]
	p_ref = (p_ref.lat.iloc[0], p_ref.lng.iloc[0])
	p_expert = points_expert[points_expert.level == level_id]
	ps = points[points.level == level_id]
	return dist(p_ref, ps), dist(p_ref, p_expert).iloc[0]


def plot_level(dists_points, dists_expert, level_id, is_hist=True, bw=None, num_bins=NUM_BINS, num_iqr=None,
	col_points=COL_POINTS, col_expert=COL_EXPERT, lw_expert=LW_EXPERT,
	title=None, xlabel=XLABEL, ylabel=None, save_to=None):
	dist_points = dists_points[level_id]
	dist_expert = dists_expert[level_id]

	if num_iqr:
		dist_points = functions.filter_iqr(dist_points, num_iqr)

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

	if save_to:
		try:
			os.makedirs(save_to)
		except:
			pass
		plt.savefig(os.path.join(save_to, 'level{}.png'.format(level_id)))

	return ax


def calc_test_stat_and_pvalue(dists_points, dists_expert, level_id, ttest=False, num_iqr=None):
	""" Calculate test statistic and its corresponding p-value for a set of points, compared to expert's point for a given level.

		dists_points : dict of pandas.DataFrame
		    Distances between data points and true point, indexed by level_id
		dists_expert : dict of float
		    Distances between expert's point and true point, indexed by level_id
		level_id     : object
		    ID of the level
		ttest        : bool (default: False)
		    If True, use t-test instead of z-test
		num_iqr      : float (default: None)
		    If specified, filter data points outside num_iqr * IQR

		Returns : float, float
		    Statistic and p-value of the test
	"""
	dist_points = dists_points[level_id]
	dist_expert = dists_expert[level_id]

	if num_iqr:
		dist_points = functions.filter_iqr(dist_points, num_iqr)
	n = len(dist_points)

	if ttest:
		test_stat, pvalue = scp.stats.ttest_1samp(dist_points, dist_expert)
	else:
		mean, se = dist_points.mean(), dist_points.std()/math.sqrt(n)
		test_stat = (mean - dist_expert)/se
		pvalue = scp.stats.norm.pdf(test_stat)
	return test_stat, pvalue


def test_signif(test_stat, pvalue, alpha=0.05, alternative='unequal'):
	""" Decides whether test statistic and p-value is significance for the given significance level and alternative hypothesis.

		test_stat   : float
		    Test statistic of the result
		pvalue      : float
		    P-value of the test statistic
		alpha       : float (default: 0.05)
		    Significance level
		alternative : str (default: 'unequal')
		    Direction of alternative hypothesis. One of 'unequal', 'greater', or 'less'
	"""
	if alternative == 'unequal':
		return pvalue < alpha/2
	elif alternative == 'greater':
		return (pvalue < alpha) and (test_stat > 0)
	elif alternative == 'less':
		return (pvalue < alpha) and (test_stat < 0)
	else:
		raise AssertionError("Alternative must be one of 'unequal', 'greater', or 'less'")


def parse_and_assert_args():
	arg_parser = argparse.ArgumentParser(prog='pinisi', description="Statistical analysis for Pinisi data", usage="")

	arg_parser.add_argument('-a', dest='alpha', help='Significance level of the z-test (default: {})'.format(ALPHA))
	arg_parser.add_argument('-d', dest='direction', help="Direction of alternative hypothesis of the z-test. One of 'unequal', 'greater', or 'less' (default '{}')".format(DIRECTION))
	arg_parser.add_argument('-i', dest='num_iqr', help='Filter data points outside num_iqr * IQR. Set to 0 to prevent filtering (default: {})'.format(NUM_IQR))
	arg_parser.add_argument('-t', dest='ttest', action='store_true', help='Use t-test instead of z-test (default: False)')
	arg_parser.add_argument('-s', dest='save_to', help='Save distance histograms to directory (default: None)')
	
	args = arg_parser.parse_args()
	alpha = args.alpha
	num_iqr = args.num_iqr
	direction = args.direction
	if alpha is not None:
		alpha = float(alpha)
		assert (alpha >= 0.0) and (alpha <= 1.0), "alpha must be between 0 and 1"
	if num_iqr is not None:
		num_iqr = float(num_iqr)
		assert num_iqr >= 0.0, "num_iqr must be >= 0"
	if direction is not None:
		assert direction in VALID_DIRECTIONS, "direction must be one of {}".format(VALID_DIRECTIONS)
	return args


if __name__ == '__main__':
	args = parse_and_assert_args()
	alpha = float(args.alpha) if args.alpha else ALPHA
	direction = args.direction or DIRECTION
	num_iqr = float(args.num_iqr) if args.num_iqr else NUM_IQR
	ttest = args.ttest or False
	save_to = args.save_to

	TRUTH_ID = int(open('../data/truth_ID').read())
	EXPERT_ID = int(open('../data/expert_ID').read())

	points = pd.read_csv('../data/clean/points.psv', sep='|', index_col='id')
	users = pd.read_csv('../data/clean/users.psv', sep='|', index_col='id')
	levels = json.load(open('../data/levels.json', 'r'))['maps']
	rects = {level['level']: functions.get_rect(level['polygon']) for level in levels}

	points_truth = points[points.user_id == TRUTH_ID].reset_index(drop=True)
	points_expert = points[points.user_id == EXPERT_ID].reset_index(drop=True)
	points = points[~points.user_id.isin([TRUTH_ID, EXPERT_ID])].reset_index(drop=True)

	dists = {level_id: get_dist_level(points_truth, points_expert, points, level_id) for level_id in LEVEL_IDS}
	dists_points = {level_id: d[0] for level_id, d in dists.items()}
	dists_expert = {level_id: d[1] for level_id, d in dists.items()}

	for level_id in LEVEL_IDS:
		plot_level(dists_points, dists_expert, level_id, num_iqr=NUM_IQR, save_to=save_to)
		plt.clf()
	
	stats_pvalues = {level_id: calc_test_stat_and_pvalue(dists_points, dists_expert, level_id, num_iqr=num_iqr, ttest=ttest) for level_id in LEVEL_IDS}
	is_signifs = {level_id: test_signif(test_stat, pvalue, alpha=alpha, alternative=direction) for level_id, (test_stat, pvalue) in stats_pvalues.items()}
	results = pd.DataFrame({
		'Test statistic': {level_id: test_stat for level_id, (test_stat, pvalue) in stats_pvalues.items()},
		'P-value': {level_id: pvalue for level_id, (test_stat, pvalue) in stats_pvalues.items()},
		'Significant?': {level_id: is_signif for level_id, is_signif in is_signifs.items()},
	}, index=LEVEL_IDS)[['Test statistic', 'P-value', 'Significant?']]
	results.index.name = 'Level'

	print("Pinisi data analysis with parameters:")
	print("Alpha       = {}".format(alpha))
	print("Alternative = {}".format(direction))
	print("Filter dist = {}".format("{}x IQR".format(num_iqr) if num_iqr else 'None'))
	print("Test        = {}".format('t-test' if ttest else 'z-test'))
	print()

	print("Results")
	print("-------")
	print(results)
