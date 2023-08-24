import numpy as np
import importlib
import matplotlib.pyplot as plt
import os
import re
import pathlib

def matches_any_regex(string, regex_list) -> bool:
    return any([re.search(regex, string) is not None for regex in regex_list])

def which_match_regex(regex, string_list) -> list[str]:
    return [s for s in string_list if re.search(regex, s) is not None]

def filter_list_by_regexes(string_list, regex_list) -> list[str]:
    return [s for s in string_list if matches_any_regex(s, regex_list)]

def aggregate_strings(strings):
    i = -1
    while i >= -min([len(x) for x in strings]):
        if len(set(x[i] for x in strings)) != 1:
            break
        i -= 1
    string = strings[0][i+1:]
    if string[0] == '_':
        return string[1:]
    else:
        return string

def aggregate_columns(df, regex_list):
    for regex in regex_list:
        metrics = which_match_regex(regex, df.keys())
        mean = np.nanmean(df[metrics].to_numpy(), axis=1)

        df = df.drop(columns=metrics)
        df[aggregate_strings(metrics)] = mean
    return df

def plot_losses(losses, options):
    if options.show_only is not None:
        first_axis_data = losses[[x for x in losses.keys()
                                  if matches_any_regex(x, options.show_only) and x not in options.omit]]
    else:
        first_axis_data = losses[[x for x in losses.keys()
                                  if matches_any_regex(x, losses.columns) and x not in options.omit]]

    if options.show_2nd_axis is not None:
        second_axis_data = losses[[x for x in losses.keys()
                                  if matches_any_regex(x, options.show_2nd_axis) and x not in options.omit]]
    else:
        second_axis_data = None

    fixed_order_cols = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
    reindex = []
    for col in fixed_order_cols:
        if col in first_axis_data.columns:
            reindex.append(col)
    reindex.extend([x for x in first_axis_data.columns if x not in fixed_order_cols])
    first_axis_data = first_axis_data[reindex]

    first_axis_data = first_axis_data.rename(columns={'D_A': 'D_Y', 'G_A': 'G', 'cycle_A': 'cycle X→Y→X'})

    if options.aggregate:
        assert options.show_only is not None, "Connot aggregate without a regex"
        first_axis_data = aggregate_columns(first_axis_data, options.show_only)
        
        if second_axis_data is not None:
            second_axis_data = aggregate_columns(second_axis_data, options.show_2nd_axis)
    
    if options.style is not None:
        plt.style.use(os.path.join(pathlib.Path(__file__).parent, "styles", f'{options.style}.mplstyle'))

    fig, ax = plt.subplots()

    if options.show_2nd_axis is None:
        axes = [ax]
    else:
        axes = [ax, ax.twinx()]

    i = 0
    for metric in first_axis_data:
        column = first_axis_data[metric].dropna()
        axes[0].plot(column.index, column.values, f"C{i}", label=metric)
        i += 1

    if second_axis_data is not None:
        for metric in second_axis_data:
            column = second_axis_data[metric].dropna()
            axes[1].plot(column.index, column.values, f"C{i}", label=metric)
            i += 1

    if options.setting is not None:
        importlib.import_module(f'settings.{options.setting}').setting(fig, ax)

    lines, labels = axes[0].get_legend_handles_labels()
    if len(axes) > 1:
        axes[1].set_ylabel(options.text_2nd_axis)
        lines2, labels2 = axes[1].get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax.legend(lines, labels)

    if options.output_file is not None:
        save_dir = pathlib.Path(options.output_file).parent
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(options.output_file)

    if options.print_last is not None:
        for loss in losses.keys():
            if matches_any_regex(loss, options.print_last):
                print(f"last {loss}: {losses[loss].dropna().iloc[-1]}")
    if options.print_min is not None:
        for loss in losses.keys():
            if matches_any_regex(loss, options.print_min):
                print(f"min {loss}: {losses[loss].dropna().min()}")

    plt.show()
