import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy as scp
import seaborn as sns

import functions


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
def plot_level(dist_points, dist_expert, level_id, is_hist=True, bw=None, num_bins=NUM_BINS,
	col_points=COL_POINTS, col_expert=COL_EXPERT, lw_expert=LW_EXPERT,
	title=None, xlabel=None, ylabel=None, save_to=None):
	if is_hist:
		ax = dist_points.plot.hist(bins=num_bins, color=col_points)
	else:
		if bw:
			ax = sns.kdeplot(dist_points, bw=bw, color=col_points)
		else:
			ax = sns.kdeplot(dist_points, color=col_points)

	plt.axvline(dist_expert, 0, len(dist_points), color=col_expert, lw=LW_EXPERT)

	plt.title(title or "Distribusi jarak pemain - Level {}".format(level_id))
	plt.xlabel(xlabel or XLABEL)
	if ylabel:
		plt.ylabel(ylabel)
	else:
		plt.ylabel("Jumlah pemain" if is_hist else "Distribusi")

	if save_to:
		try:
			os.makedirs(save_to)
		except:
			pass
		plt.savefig(os.path.join(save_to, 'level{}.png'.format(level_id)))

	return ax


def calc_test_stat_and_pvalue(X, ref, alternative='unequal'):
	""" Calculate test statistic and its corresponding p-value for a set of points, compared to expert's point for a given level.

		Returns : float, float
		    Statistic and p-value of the test
	"""
	n = len(X)
	mean, se = X.mean(), X.std()/math.sqrt(n)
	test_stat = (mean - ref)/se
	
	if alternative == 'unequal':
		pvalue = scp.stats.norm.pdf(test_stat)
	elif alternative == 'greater':
		pvalue = 1.0 - scp.stats.norm.cdf(test_stat)
	elif alternative == 'less':
		pvalue = scp.stats.norm.cdf(test_stat)
	
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
	elif (alternative == 'greater') or (alternative == 'less'):
		return pvalue < alpha
	else:
		raise AssertionError("Alternative must be one of 'unequal', 'greater', or 'less'")


def format_pvalue(pvalue):
	if pvalue < 1e-3:
		return "~0"
	return ("%.3f" % pvalue)


def parse_and_assert_args():
	arg_parser = argparse.ArgumentParser(prog='pinisi', description="Statistical analysis for Pinisi data", usage="")

	arg_parser.add_argument('-a', dest='alpha', help='Significance level of the z-test (default: {})'.format(ALPHA))
	arg_parser.add_argument('-d', dest='direction', help="Direction of alternative hypothesis of the z-test. One of 'unequal', 'greater', or 'less' (default '{}')".format(DIRECTION))
	arg_parser.add_argument('-r', dest='filter_rect', action='store_true', help='Filter data points only inside rectangle. Takes precedence over num_iqr')
	arg_parser.add_argument('-i', dest='num_iqr', help='Filter data points outside num_iqr * IQR. Set to 0 to prevent filtering (default: {})'.format(NUM_IQR))
	arg_parser.add_argument('-p', dest='is_format_pvalue', action='store_true', help='Format p-value to be more readable (default: False)')
	arg_parser.add_argument('-s', dest='save_to', help='Directory for saving plots (default: None)')
	
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
	filter_rect = args.filter_rect
	num_iqr = float(args.num_iqr) if args.num_iqr else NUM_IQR
	is_format_pvalue = args.is_format_pvalue
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

	dists = {level_id: functions.get_dist_level(points_truth, points_expert, points, level_id) for level_id in LEVEL_IDS}
	dists_points = {level_id: d[0] for level_id, d in dists.items()}
	dists_expert = {level_id: d[1] for level_id, d in dists.items()}

	for level_id, dist_points in dists_points.items():
		if filter_rect:
			points_idx = functions.get_points_level(points, rects, level_id).index
			dist_points = dist_points[points_idx]
			dists_points[level_id] = dist_points
		elif num_iqr:
			dist_points = functions.filter_iqr(dist_points, num_iqr)
			dists_points[level_id] = dist_points

		dist_expert = dists_expert[level_id]
		plot_level(dist_points, dist_expert, level_id, save_to=save_to)
		plt.clf()
	
	stats_pvalues = {level_id: calc_test_stat_and_pvalue(dist_points, dists_expert[level_id], alternative=direction) for level_id, dist_points in dists_points.items()}
	is_signifs = {level_id: test_signif(test_stat, pvalue, alpha=alpha, alternative=direction) for level_id, (test_stat, pvalue) in stats_pvalues.items()}
	results = pd.DataFrame({
		'Test statistic': {level_id: test_stat for level_id, (test_stat, pvalue) in stats_pvalues.items()},
		'P-value': {level_id: pvalue for level_id, (test_stat, pvalue) in stats_pvalues.items()},
		'Significant?': {level_id: is_signif for level_id, is_signif in is_signifs.items()},
	}, index=LEVEL_IDS)[['Test statistic', 'P-value', 'Significant?']]
	
	results.index.name = 'Level'
	results['Test statistic'] = results['Test statistic'].round(3)
	if is_format_pvalue:
		results['P-value'] = results['P-value'].apply(format_pvalue)

	print("Pinisi data analysis with parameters:")
	print("Alpha       = {}".format(alpha))
	print("Alternative = {}".format(direction))
	if filter_rect:
		print("Filter      = level rectangle")
	else:
		print("Filter      = {}".format("{}x IQR".format(num_iqr) if num_iqr else 'None'))
	print()

	print("Results")
	print("-------")
	print(results)
