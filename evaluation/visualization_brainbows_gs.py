from pathlib import Path
import re
from typing import Any
from evaluation_utils import extend_path
import readline
extend_path()
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from utils.training_analysis.visualizer import aggregate_strings, filter_list_by_regexes, which_match_regex
from collections import deque
from enum import Enum

"""
For the y_axis array: Dimension 1 is the x_axis, dimension 0 are the bars
"""
    
class State(Enum):
    VIS_PARAMS = 1
    REORDER_X = 2

class CliRunner:

    def __init__(self, df):
        self.df = df
        self.states = deque()

    def check_special_action(self, string):
        """
        Checks input for help and quit. Returns true, if the prompt should be repeated.
        """
        if string.lower() == 'help' or string == '-h':
            self.print_help()
            return True
        elif string.lower() == 'quit' or string.lower() == 'exit' or string == '-q':
            exit()
        elif string.lower() == '-list':
            print('The metrics that can be visualized are:', ', '.join(list(self.df.data_vars.keys())))
            print('The coordinates that can be chosen are:', ', '.join(list(self.df.coords.keys())))
            return True
        elif string == '':
            return True

        return False

    def get_visualization_params(self):
        self.states.append(State.VIS_PARAMS)

        s = input('Visualization params [type "help" for help]: ')
        while self.check_special_action(s):
            s = input('Visualization params [type "help" for help]: ')

        params: dict[Any, Any] = {'bar': False}

        metrics = s.split(' ')
        metrics = [x for x in metrics if x != ' ' and x != '']
        assert '--' in metrics
        i = metrics.index('--')
        params['x_axis'] = metrics[:i]
        if '--' in metrics[i+1:]:
            j = metrics[i+1:].index('--') + i + 1
            params['metrics1'] = metrics[i+1:j]
            params['metrics2'] = metrics[j+1:]
        else:
            params['metrics1'] = metrics[i+1:]
            params['metrics2'] = []

        for metrics in ['metrics1', 'metrics2']:
            if '-b' in params[metrics]:
                params['bar'] = True
                params[metrics].remove('-b')
            if '-no-y' in params[metrics]:
                params['no_y'] = True
                params[metrics].remove('-no-y')
            if '-f' in params[metrics]:
                i = params[metrics].index('-f') + 1
                fig_size_string = params[metrics][i]
                fig_size = [int(x) for x in fig_size_string.split(',')]
                params['fig_size'] = fig_size
                params[metrics].remove('-f')
                params[metrics].remove(fig_size_string)
            if '-p' in params[metrics]:
                params['print'] = True
                params[metrics].remove('-p')
            if '-x' in params[metrics]:
                i = params[metrics].index('-x') + 1
                custom_x = params[metrics][i]
                params[metrics].remove('-x')
                params[metrics].remove(custom_x)
                params['custom_x'] = custom_x
            if not '-a' in params[metrics]:
                continue
            i = params[metrics].index('-a') + 1
            aggregate_string = params[metrics][i]
            params[metrics].remove('-a')
            params[metrics].remove(aggregate_string)

            if '=' in aggregate_string:
                l = aggregate_string.split(',')
                keys = []
                values = []
                for item in l:
                    key, value = item.split('=')
                    keys.append(key)
                    values.append(value)
                params[f'{metrics}_aggregate'] = dict(zip(keys, values))
            else:
                params[f'{metrics}_aggregate'] = aggregate_string

        try:
            assert 1 <= len(params['x_axis']) <= 2, "Only 1 or 2 x arguments are supported"
            for metrics in ['metrics1', 'metrics2']:
                if len(params[metrics]) == 0:
                    continue
                elif f'{metrics}_aggregate' not in params:
                    dims = max([len(self.df[x].dims) for x in params[metrics]])
                elif type(params[f'{metrics}_aggregate']) == str:
                    continue
                else:
                    dims = len(self.df[params[metrics]].sel(params[f'{metrics}_aggregate']).dims) 
                assert dims == len(params['x_axis']), f"Specified aggregation yields {dims} dimension(s), " \
                    + f"but {len(params['x_axis'])} dimension(s) have been specified!"
                assert dims <= 2, "Only 1 or 2 coords can be visualized"
                if dims == 2:
                    assert len(params['metrics1']) == 1, "When showing 2 coords, only one metric can be selected"
                    assert len(params['metrics2']) == 0, "When showing 2 coords, only first axis can be used"
        except AssertionError as e:
            print(e)
            self.states.pop()
            return self.get_visualization_params()

        self.states.pop()
        return params

    def print_help(self):
        constant_help_string = """
Type 'help' to show help (which you just did, lol).
Type one of 'quit', 'exit', '-q' to exit the program.
Type '-list' to list all metrics and coordinates that can be selected.
        """
        if self.states[-1] == State.VIS_PARAMS:
            print("""
Interactively plot metrics from a grid search analysis!
    If asked for 'Visualization params', first type the coordinates that you want to be on the x-axis. 
    After that, type the metrics you want to visualize on the first axis, separated by '--' from the x_axis coords.
    After another '--' you can type metrics for the second axis. All metrics and coords can be specified as regex.
    Add -a to aggregate metrics accross not specified coords. To do this, type either a numpy function after the '-a'
      which'll be used or specify concrete values for the parameters like 'bg_threshold=0.1,bias_cut=-0.3'.
    Add -b to force a bar plot. This can be useful if a line plot isn't sensible (e.g. few x values)
    Add -p to print the values that are visualized.
    Add -x followed by a string representing a pything iterable that is used as the x-axis. E.g. [1,2,3] to use these
      number as values for the x-axis.
    Add -no-y to omit that the metrics that are displayed on an axis are listed as the label for that axis
    Add -f followed by the figure size to change the size of the figure, e.g. -f 7,6
                  """)
        else:
            raise ValueError("Invalid state encountered!")
        print(constant_help_string)

    def get_visualization_df(self, params):
        if 'custom_x' in params:
            x_axis = np.array(eval(params['custom_x']))
        else:
            x_axis = self.df[params['x_axis'][0]].values

        metrics1 = filter(lambda metric: True if metric in self.df.keys() else print(f"Metric {metric} is not in dataset"), params['metrics1'])
        metrics2 = filter(lambda metric: True if metric in self.df.keys() else print(f"Metric {metric} is not in dataset"), params['metrics2'])

        y_axis = [[], []]
        labels = [[], []]
        for i in range(2):
            for metric in [metrics1, metrics2][i]:
                agg_str = f'metrics{i+1}_aggregate'
                if agg_str not in params:
                    sub_df = self.df[metric]
                elif type(params[agg_str]) == str:
                    aggregation_metrics = list(set(self.df[metric].coords.keys()) - set(params['x_axis']))
                    sub_df = eval(f"self.df[metric].{params[agg_str]}({aggregation_metrics})")
                else:
                    sub_df = self.df[metric].sel(params[agg_str])
                
                y_axis[i].append(sub_df.values.T if sub_df.dims[0] == params['x_axis'][0] else sub_df.values)
                label = metric.replace('_', ' ')
                digit_match = re.search(r'\d', label)
                if digit_match is not None and label[digit_match.start() - 1] != ' ' and digit_match.start() > 0:
                    label = label[:digit_match.start()] + ' ' + label[digit_match.start():]
                labels[i].append(label)

        y_axis1 = np.array(y_axis[0])
        y_axis2 = np.array(y_axis[1])
        if y_axis1.ndim == 3:
            assert y_axis1.shape[0] == 1, "Here one metric with 2 coords should be selected"
            y_axis1 = y_axis1[0]
        if y_axis2.ndim == 3:
            assert y_axis2.shape[0] == 1, "Here one metric with 2 coords should be selected"
            y_axis2 = y_axis2[0]

        if len(params['x_axis']) == 2:
            labels[0] = [labels[0][0] + f" {x}" for x in self.df[params['x_axis'][1]].values]

        return x_axis, y_axis1, y_axis2, labels[0], labels[1]


def plot_bars(ax, x_axis, y_axis, labels, color_offset=0, base_offset=0):
    if len(y_axis) == 0:
        return
    if y_axis.shape[0] == 1 or True:
        for i in range(y_axis.shape[0]):
            offset = width * ((-column_count) / 2 + 0.5 + i + base_offset)
            ax.bar(np.arange(len(x_axis)) + offset, y_axis[i], width, label=labels[i], color=f'C{color_offset + i}')
    else:
        for i in range(y_axis.shape[1]):
            offset = width * ((-column_count) / 2 + 0.5 + i + base_offset)
            ax.bar(np.arange(len(x_axis)) + offset, y_axis[:, i], width, label=labels[i], color=f'C{color_offset + i}')

def plot_graphs(ax, x_axis, y_axis, labels, color_offset=0):
    for i in range(y_axis.shape[0]):
        ax.plot(x_axis, y_axis[i], label=labels[i], color=f'C{color_offset + i}')

def set_size(w,h, ax):
    """ w, h: width, height in inches """
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_nc', help='Path to the evaluation netCDF file.')
    parser.add_argument('--style', default="style_1", help='The style that shall be applied')

    args = parser.parse_args()
    all_data = xr.load_dataset(args.eval_nc)
    runner = CliRunner(all_data)

    plt.style.use(Path(__file__).parent.parent / 'utils' / 'training_analysis' / 'styles' / f'{args.style}.mplstyle')
    fig = plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.subplots_adjust(left=0.07, right=0.93, top=0.97, bottom=0.08)
    fig.canvas.mpl_connect('close_event', lambda _: exit())
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    plt.ion()
    fig.show()


    while True:
        params = runner.get_visualization_params()
        x_axis, y_axis1, y_axis2, labels1, labels2 = runner.get_visualization_df(params)
        
        ax1.clear()
        ax2.clear()

        if 'fig_size' in params:
            set_size(params['fig_size'][0], params['fig_size'][1], ax1)

        c1 = y_axis1.shape[0]
        c2 = y_axis2.shape[0]
        column_count = c1 + c2
        width = 1 / (column_count + 1)

        if len(params['x_axis']) == 2 or all_data[params['x_axis'][0]].values.dtype == object or params['bar']:
            plot_bars(ax1, x_axis, y_axis1, labels1)
            plot_bars(ax2, x_axis, y_axis2, labels2, y_axis1.shape[0], c1)
            ax1.set_xticks(np.arange(len(x_axis)), x_axis)
        else:
            plot_graphs(ax1, x_axis, y_axis1, labels1)
            plot_graphs(ax2, x_axis, y_axis2, labels2, c1)

        if c1 == 0:
            ax1.set_axis_off()
        if c2 == 0:
            ax2.set_axis_off()

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
        ax1.legend(lines, labels)
        ax1.set_xlabel(params['x_axis'][0])
        ax2.set_xlim(*ax1.get_xlim())

        if 'print' in params and params['print']:
            print(x_axis)
            if c1 != 0:
                print(y_axis1)
            if c2 != 0:
                print(y_axis2)

        if len(params['x_axis']) == 2:
            ax1_label = params['metrics1'][0] + f"_[{params['x_axis'][1]}]"
        else:
            ax1_label = ', '.join(labels1)

        if 'no_y' not in params or not params['no_y']:
            ax1.set_ylabel(ax1_label)
            ax2.set_ylabel(', '.join(labels2))
        plt.axes(ax1)
