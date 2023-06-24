from collections import defaultdict
import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.parasite_axes import HostAxes
import os
import re
import pathlib

def matches_any_regex(string, regex_list) -> bool:
    return any([re.search(regex, string) is not None for regex in regex_list])

def plot_losses(losses: list[dict[str, float]], options):
    data = defaultdict(lambda: ([], []))
    for i, entry in enumerate(losses):
        for key in entry:
            if options.iteration_variable in entry:
                data[key][0].append(entry[options.iteration_variable])
            else:
                data[key][0].append(i)
            data[key][1].append(entry[key])

    if options.iteration_variable in data:
        del data[options.iteration_variable]
    
    if options.style is not None:
        plt.style.use(os.path.join(pathlib.Path(__file__).parent, "styles", f'{options.style}.mplstyle'))

    fig, ax = plt.subplots()

    if options.show_2nd_axis is None:
        axes = [ax]
    else:
        axes = [ax, ax.twinx()]
    
    i = 0
    for loss in data:
        if loss in options.omit:
            continue

        if options.show_only is not None and matches_any_regex(loss, options.show_only):
            axes[0].plot(data[loss][0], data[loss][1], f"C{i}", label=loss)
            i += 1
        elif options.show_2nd_axis is not None and matches_any_regex(loss, options.show_2nd_axis):
            axes[1].plot(data[loss][0], data[loss][1], f"C{i}", label=loss)
            i += 1
        elif options.show_only is None and options.show_2nd_axis is None:
            axes[0].plot(data[loss][0], data[loss][1], f"C{i}", label=loss)
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
        for loss in data.keys():
            if matches_any_regex(loss, options.print_last):
                print(f"{loss}: {data[loss][1][-1]}")

    plt.show()
