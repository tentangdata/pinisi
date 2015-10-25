import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FIG_AREA = 48
GRID_AREA = 96

COLOR_GRID = 'lightgrey'
GRID_LW = 0.5


# Data processing functions
def filter_iqr(X, num_iqr):
    median = X.median()
    iqr = X.quantile(0.75) - X.quantile(0.25)
    return X[(X >= median - iqr*num_iqr) & (X <= median + iqr*num_iqr)]

# Plotting functions
def plot_scatter(points, rects, level_id, fig_area=FIG_AREA, grid_area=GRID_AREA, with_axis=False, with_img=True, img_alpha=1.0):
    rect = rects[level_id]
    top_lat, top_lng, bot_lat, bot_lng = get_rect_bounds(rect)

    plevel = get_points_level(points, rects, level_id)
    ax = plevel.plot('lng', 'lat', 'scatter')
    plt.xlim(left=top_lng, right=bot_lng)
    plt.ylim(top=top_lat, bottom=bot_lat)

    if with_img:
        img = plt.imread('/data/images/level%s.png' % level_id)
        plt.imshow(img, zorder=0, alpha=img_alpha, extent=[top_lng, bot_lng, bot_lat, top_lat])

    width, height = get_rect_width_height(rect)
    fig_width, fig_height = get_fig_width_height(width, height, fig_area)
    plt.gcf().set_size_inches(fig_width, fig_height)

    if grid_area:
        grid_horiz, grid_vertic = get_grids(rects, level_id, grid_area, fig_area)
        for lat in grid_horiz:
            plt.axhline(lat, color=COLOR_GRID, lw=GRID_LW)
        for lng in grid_vertic:
            plt.axvline(lng, color=COLOR_GRID, lw=GRID_LW)

    if not with_axis:
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    return ax


def plot_heatmap(points, rects, level_id, grid_area=GRID_AREA, fig_area=FIG_AREA, cmap=None):
    rect = rects[level_id]
    top_lat, top_lng, bot_lat, bot_lng = get_rect_bounds(rect)

    bins = get_bins(points, rects, level_id, grid_area, fig_area)
    
    ax = sns.heatmap(bins, cbar=False, xticklabels=False, yticklabels=False, cmap=cmap)
    width, height = get_rect_width_height(rect)
    fig_width, fig_height = get_fig_width_height(width, height, fig_area)
    plt.gcf().set_size_inches(fig_width, fig_height)

    return ax


# Helper functions
def get_points_level(points, rects, level_id):
    rect = rects[level_id]
    top_lat, top_lng = rect[0][0], rect[0][1]
    bot_lat, bot_lng = rect[1][0], rect[1][1]

    plevel = points[points['level'] == level_id]
    plevel = plevel[(plevel['lng'] >= top_lng) & (plevel['lng'] <= bot_lng) & (plevel['lat'] >= bot_lat) & (plevel['lat'] <= top_lat)]
    return plevel


def get_fig_width_height(width, height, fig_area):
    c_area = math.sqrt(fig_area / (width*height))
    fig_width  = width*c_area
    fig_height = height*c_area
    return fig_width, fig_height
    

def get_rect_width_height(rect):
    top_lat, top_lng, bot_lat, bot_lng = get_rect_bounds(rect)
    width  = bot_lng - top_lng
    height = top_lat - bot_lat
    return width, height


def get_rect(polygons):
    top_lat = max([coord[0] for coord in polygons])
    bot_lat = min([coord[0] for coord in polygons])
    top_lng = min([coord[1] for coord in polygons])
    bot_lng = max([coord[1] for coord in polygons])
    return (top_lat, top_lng), (bot_lat, bot_lng)


def get_rect_bounds(rect):
    top_lat, top_lng = rect[0][0], rect[0][1]
    bot_lat, bot_lng = rect[1][0], rect[1][1]
    return top_lat, top_lng, bot_lat, bot_lng


def get_grids(rects, level_id, grid_area, fig_area):
    rect = rects[level_id]
    top_lat, top_lng, bot_lat, bot_lng = get_rect_bounds(rect)
    width, height = get_rect_width_height(rect)
    fig_width, fig_height = get_fig_width_height(width, height, fig_area)
    
    factor = math.sqrt(grid_area/fig_area)
    n_grid_horiz, n_grid_vertic = round(grid_area/fig_width/factor), round(grid_area/fig_height/factor)
    grid_horiz  = [top_lat - i*height/n_grid_horiz for i in range(n_grid_horiz + 1)]
    grid_vertic = [bot_lng - i*width/n_grid_vertic for i in range(n_grid_vertic + 1)]
    return grid_horiz, grid_vertic


def get_bins(points, rects, level_id, grid_area=GRID_AREA, fig_area=FIG_AREA):
    plevel = get_points_level(points, rects, level_id)
    bins = get_grids(rects, level_id, grid_area, fig_area)
    bins_lat = pd.Series(pd.Categorical(pd.cut(plevel.lat, sorted(bins[0]), include_lowest=True), ordered=True))
    bins_lng = pd.Series(pd.Categorical(pd.cut(plevel.lng, sorted(bins[1]), include_lowest=True), ordered=True))

    coord_bins = pd.DataFrame(0, index=bins_lat.values.categories, columns=bins_lng.values.categories)
    coord_counts = (bins_lat.astype(str) + '|' + bins_lng.astype(str)).value_counts()
    for coord in coord_counts.index:
        lat, lng = coord.split('|')
        coord_bins.loc[lat, lng] += coord_counts[coord]
    assert coord_bins.sum().sum() == len(plevel)

    coord_bins.index = [float(i[1:i.index(',')]) for i in coord_bins.index]
    coord_bins.columns = [float(i[1:i.index(',')]) for i in coord_bins.columns]
    coord_bins = coord_bins.loc[coord_bins.index[::-1]]    # reverse latitude (positive should be upper)
    return coord_bins

# Dataset-specific functions
def browser_to_os(browser):
    browser = browser.lower()

    # Ubuntu
    if "ubuntu" in browser:
        return "Ubuntu"

    # Windows Phone
    elif "windows phone 8" in browser:
        return "Windows Phone 8"

    # Windows
    elif ("windows nt 10" in browser):
        return "Windows 10"
    elif ("windows nt 6.2" in browser) or ("windows nt 6.3" in browser):
        return "Windows 8"
    elif ("windows nt 6.1" in browser):
        return "Windows 7"
    elif ("windows nt 6.0" in browser):
        return "Windows Vista"
    elif ("windows nt 5" in browser):
        return "Windows XP"

    # OS X
    elif ("intel mac os x" in browser):
        return "OS X"
    
    # iPhone
    elif ("cpu iphone" in browser) or ("wp-iphone" in browser):
        if ("os 9" in browser):
            return "iPhone iOS 9"
        elif ("os 8" in browser):
            return "iPhone iOS 8"
        elif ("os 7" in browser):
            return "iPhone iOS 7"

    # Android
    elif "android 5" in browser:
        return "Android 5"
    elif "android 4" in browser:
        return "Android 4"
    elif "android 3" in browser:
        return "Android 3"
    elif "android 2" in browser:
        return "Android 2"
    elif ("android" in browser) and ("tablet" in browser):
        return "Android tablet"
    elif "android" in browser:
        return "Android"

    # Other Linuxes
    elif "linux" in browser:
        return "Other Linux"

    else:
        raise ValueError("Unknown agent string: %s" % browser)


def os_to_generic(os_name):
    os_name = os_name.lower()

    if "ubuntu" in os_name:
        return "Ubuntu"
    elif "linux" in os_name:
        return "Linux"
    elif "android" in os_name:
        return "Android"
    elif "os x" in os_name:
        return "OS X"
    elif "ios" in os_name:
        return "iOS"
    elif "windows phone" in os_name:
        return "Windows Phone"
    elif "windows" in os_name:
        return "Windows"
    else:
        raise ValueError("Unknown os: %s" % os_name)


def is_mobile(os_generic):
    return os_generic.lower() in ["android", "ios", "windows phone"]
