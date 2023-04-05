from collections import defaultdict
import importlib
import matplotlib.pyplot as plt
import os
import pathlib

def plot_losses(losses: list[dict[str, float]], options):
    data = defaultdict(lambda: [])
    for entry in losses:
        for key in entry:
            data[key].append(entry[key])

    if options.iteration_variable in data:
        x = data[options.iteration_variable]
        del data[options.iteration_variable]
    else:
        x = range(len(data[list(data.keys())[0]]))

    
    if options.style is not None:
        plt.style.use(os.path.join(pathlib.Path(__file__).parent, "styles", f'{options.style}.mplstyle'))
    fig, ax = plt.subplots()
    
    for loss in data:
        show_loss = loss not in options.omit
        if options.show_only is not None:
            show_loss &= loss in options.show_only
        if show_loss:
            ax.plot(x, data[loss], label=loss)
    ax.legend()

    if options.setting is not None:
        importlib.import_module(f'settings.{options.setting}').setting(fig, ax)

    if options.output_file is not None:
        plt.savefig(options.output_file)
    plt.show()
