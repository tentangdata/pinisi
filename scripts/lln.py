import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns

import functions


# Constants
LEVEL_IDS = list(range(1, 7))

LINE_COL = 'saddlebrown'
LW = 3
BG_COLOR = 'tan'
XLABEL = 'Jumlah pemain'
YLABEL = 'Rata-rata jarak'
NUM_IQR = 2.0

sns.set_style('white')

# Functions
def plot_cum_dist(dist_points, level_id, line_col=None, lw=None, bg_color=None, show_mean=False, title=None, xlabel=None, ylabel=None, save_to=None):
    cum_dist = calc_cum_dist(dist_points)
    ax = cum_dist.plot.line(color=line_col or LINE_COL, lw=lw or LW)
    ax.set_axis_bgcolor(bg_color or BG_COLOR)

    if show_mean:
        plt.axhline(dist_points.mean(), 0, len(dist_points), linestyle='--', color=LINE_COL)

    plt.title(title or "Rata-rata jarak pemain terhadap jumlah pemain - Level {}".format(level_id))
    plt.xlabel(xlabel or XLABEL)
    plt.ylabel(ylabel or YLABEL)

    if save_to:
        try:
            os.makedirs(save_to)
        except:
            pass
        plt.savefig(os.path.join(save_to, 'level{}.png'.format(level_id)))

    return ax


def calc_cum_dist(dist_points):
    return dist_points.cumsum() / pd.Series(range(1, len(dist_points) + 1))


def parse_and_assert_args():
    arg_parser = argparse.ArgumentParser(prog='pinisi', description="Law of large numbers simulation with Pinisi data", usage="")

    arg_parser.add_argument('-r', dest='filter_rect', action='store_true', help='Filter data points only inside rectangle. Takes precedence over num_iqr (default: False)')
    arg_parser.add_argument('-i', dest='num_iqr', help='Filter data points outside num_iqr * IQR. Set to 0 to prevent filtering (default: {})'.format(NUM_IQR))
    arg_parser.add_argument('-m', dest='show_mean', action='store_true', help='Show mean line (default: False)')
    arg_parser.add_argument('-s', dest='save_to', help='Directory for saving plots (default: None)')
    
    args = arg_parser.parse_args()
    num_iqr = args.num_iqr
    if num_iqr is not None:
        num_iqr = float(num_iqr)
        assert num_iqr >= 0.0, "num_iqr must be >= 0"
    return args


if __name__ == '__main__':
    args = parse_and_assert_args()
    filter_rect = args.filter_rect
    num_iqr = float(args.num_iqr) if args.num_iqr else NUM_IQR
    show_mean = args.show_mean
    save_to = args.save_to

    TRUTH_ID = int(open('../data/truth_ID').read())
    EXPERT_ID = int(open('../data/expert_ID').read())

    points = pd.read_csv('../data/clean/points.psv', sep='|', index_col='id')
    users = pd.read_csv('../data/clean/users.psv', sep='|', index_col='id')
    levels = json.load(open('../data/levels.json', 'r'))['maps']
    rects = {level['level']: functions.get_rect(level['polygon']) for level in levels}
    areas = {level_id: functions.calc_rect_area(rect) for level_id, rect in rects.items()}

    points_truth = points[points.user_id == TRUTH_ID].reset_index(drop=True)
    points_expert = points[points.user_id == EXPERT_ID].reset_index(drop=True)
    points = points[~points.user_id.isin([TRUTH_ID, EXPERT_ID])].reset_index(drop=True)

    dists = {level_id: functions.get_dist_level(points_truth, points_expert, points, level_id) for level_id in LEVEL_IDS}
    dists_points = {level_id: d[0] for level_id, d in dists.items()}
    
    for level_id, dist_points in dists_points.items():
        if filter_rect:
            points_idx = functions.get_points_level(points, rects, level_id).index
            dist_points = dist_points[points_idx]
        elif num_iqr:
            dist_points = functions.filter_iqr(dist_points, num_iqr)

        dist_points = dist_points.reset_index(drop=True)
        plot_cum_dist(dist_points, level_id, show_mean=show_mean, save_to=save_to)
        plt.clf()
