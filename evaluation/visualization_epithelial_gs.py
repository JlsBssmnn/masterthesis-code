from pathlib import Path
from evaluation_utils import extend_path
extend_path()
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.training_analysis.visualizer import aggregate_strings, which_match_regex
from collections import deque
from enum import Enum
    
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
            print('The metrics that can be visualized are:', ', '.join(self.df.columns))
            return True
        elif string == '':
            return True

        return False

    def get_visualization_params(self):
        self.states.append(State.VIS_PARAMS)

        s = input('Aggregation metrics [type "help" for help]: ')
        while self.check_special_action(s):
            s = input('Aggregation metrics [type "help" for help]: ')

        params = {}

        metrics = s.split(' ')
        metrics = [x for x in metrics if x != ' ' and x != '']
        if '--' in metrics:
            params['metrics1'] = metrics[:metrics.index('--')]
            params['metrics2'] = metrics[metrics.index('--') + 1:]
        else:
            params['metrics1'] = metrics
            params['metrics2'] = []

        reorderX = False
        for metrics in ['metrics1', 'metrics2']:
            params[f'{metrics}_aggregate'] = '-a' in params[metrics]
            if params[f'{metrics}_aggregate']:
                params[metrics].remove('-a')
            if '-reorderX' in params[metrics]:
                params[metrics].remove('-reorderX')
                reorderX = True
            elif '-r' in params[metrics]:
                params[metrics].remove('-r')
                reorderX = True

        if reorderX:
            x_axis = self.get_x_axis()
            if x_axis is not None:
                params['x_axis'] = x_axis

        self.states.pop()
        return params

    def get_x_axis(self):
        self.states.append(State.REORDER_X)
        s = input('How to reorder the x-axis [type "help" for help]: ')
        while self.check_special_action(s):
            s = input('How to reorder the x-axis [type "help" for help]: ')

        numbers = s.split(' ')
        numbers = [x for x in numbers if x != ' ' and x != '']

        if '-abort' in numbers:
            self.states.pop()
            return None

        step = int(numbers[0])
        n_groups = int(numbers[1]) if len(numbers) > 1 else 2

        experiments = np.array(self.df.index).reshape(-1, step)
        group_size = experiments.size // n_groups

        order = np.empty((n_groups, group_size), dtype=int)
        for i in range(n_groups):
            order[i] = experiments[range(i, experiments.shape[0], n_groups)].flatten()

        self.states.pop()
        return order

    def print_help(self):
        constant_help_string = """
Type 'help' to show help (which you just did, lol).
Type one of 'quit', 'exit', '-q' to exit the program.
Type '-list' to list all metrics that can be selected.
        """
        if self.states[-1] == State.VIS_PARAMS:
            print("""
Interactively plot metrics from a grid search analysis!
    If asked for 'Aggregation metrics', type regexes, seperated by space for the metrics you want to visualize.
    Metrics after a double dash '--' will be splotted on the second axis.
    Add -a to aggregate all metrics that match one regex. This will only aggregate metrics for that axis.
                  """)
        elif self.states[-1] == State.REORDER_X:
            print("""
You are reordering the x axis, meaning you are changing the order in which experiments are going to be displayed.
This is done by splitting the experiments into groups.
To abort this, type '-abort'.
To reorder, first type after how many experiments an experiment belong to the next group. E.g. to group experiments into
    an even and an odd group this would be 1. Optionnaly after that specify the number of groups, which defaults to 2.
                  """)
        else:
            raise ValueError("Invalid state encountered!")
        print(constant_help_string)

    def get_visualization_df(self, params):
        df = [pd.DataFrame(), pd.DataFrame()]
        for i in range(1, 3):
            for regex in params[f'metrics{i}']:
                metrics = which_match_regex(regex, all_data.keys())
                sub_df = all_data[metrics]

                if params[f'metrics{i}_aggregate']:
                    sub_df = pd.DataFrame({aggregate_strings(metrics): sub_df.mean(1)})
                df[i - 1] = pd.concat((df[i - 1], sub_df), axis=1)
        return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_csv', help='Path to the evaluation csv.')
    parser.add_argument('--style', default="style_1", help='The style that shall be applied')

    args = parser.parse_args()
    all_data = pd.read_csv(args.eval_csv, index_col=0)
    runner = CliRunner(all_data)

    plt.style.use(Path(__file__).parent.parent / 'utils' / 'training_analysis' / 'styles' / f'{args.style}.mplstyle')
    fig = plt.figure()
    fig.subplots_adjust(left=0.07, right=0.93, top=0.97, bottom=0.04)
    fig.canvas.mpl_connect('close_event', lambda _: exit())
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    # fig.tight_layout()
    plt.ion()
    fig.show()


    while True:
        params = runner.get_visualization_params()
        df1, df2 = runner.get_visualization_df(params)
        
        ax1.clear()
        ax2.clear()

        c1 = df1.shape[1]
        c2 = df2.shape[1]
        column_count = df1.shape[1] + df2.shape[1]
        width = 1 / (column_count + 1)

        color_value = 0
        for i, column in enumerate(df1):
            offset = width * ((-column_count) / 2 + 0.5 + i)
            x_axis = df1.index if 'x_axis' not in params else params['x_axis'].flatten()[df1.index]
            rects = ax1.bar(df1.index + offset, df1[column][x_axis], width, label=column, color=f'C{color_value}')
            color_value += 1
            # ax.bar_label(rects, padding=3)
        for i, column in enumerate(df2):
            offset = width * ((-column_count) / 2 + 0.5 + i + c1)
            x_axis = df2.index if 'x_axis' not in params else params['x_axis'].flatten()[df2.index]
            rects = ax2.bar(df2.index + offset, df2[column][x_axis], width, label=column, color=f'C{color_value}')
            color_value += 1

        if c1 == 0:
            ax1.set_axis_off()
        if c2 == 0:
            ax2.set_axis_off()

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
        ax1.legend(lines, labels)

        if 'x_axis' in params:
            ax1.set_xticks(df1.index, params['x_axis'].flatten(), fontsize='large')
            for i in range(1, params['x_axis'].shape[0]):
                offset = width * ((-column_count) / 2 + 0.5 + i + column_count)
                spacing = params['x_axis'].shape[1]
                ax1.axline((i * spacing - 0.5, 0), slope=float('inf'), color='black')
        ax1.set_ylabel(', '.join(df1.columns))
        ax2.set_ylabel(', '.join(df2.columns))
        # ax.set_xticks(df.index + width, df.index)
