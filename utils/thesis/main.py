import argparse

import xarray as xr
from collections import defaultdict

def map_param_values(name, values):
    new = []
    if name == 'dataset':
        for value in values:
            if value == 'v_1_1_1':
                new.append('dataset 1')
            elif value == 'v_1_2_0':
                new.append('dataset 2')
            elif value == 'v_1_3_0':
                new.append('dataset 3')
            else:
                raise ValueError(f'{value} is invalid dataset name')
        return new
    elif name == 'lr':
        for value in values:
            if value == 0.0002:
                new.append('$2\\cdot 10^{-4}$')
            elif value == 0.00001:
                new.append('$1\\cdot 10^{-5}$')
            else:
                raise ValueError(f'{value} is invalid lr')
    elif name == 'lambda':
        new = [str(int(x)) for x in values]
    elif name == 'network':
        new = values
    elif name == 'disc_transform':
        new = values
    else:
        raise NotImplementedError(f'param {name} not implemented')
    return new

def print_gs_tables(gs_result):
    ds = xr.load_dataset(gs_result)
    ds = ds.assign(avg_VI=lambda x: (x.slice1_VI + x.slice2_VI + x.slice3_VI) / 3)
    hyperparams = ['dataset', 'lr', 'network', 'lambda', 'disc_transform']

    for param in hyperparams:
        if param == 'disc_transform':
            continue
        print(param)
        param_values = ds[param].values
        values = defaultdict(lambda: [])

        for val in param_values:
            d = ds.sel({param: val})
            amin = d['avg_VI'].argmin(...)
            sel = {}
            for hyperparam in [x for x in hyperparams if x != param]:
                value = d[hyperparam][amin[hyperparam].values].item()
                values[hyperparam].append(value)
                sel[hyperparam] = value
            pinned = d.sel(sel)
            values['slice1_VI'].append(pinned['slice1_VI'].item())
            values['slice2_VI'].append(pinned['slice2_VI'].item())
            values['slice3_VI'].append(pinned['slice3_VI'].item())
            values['avg_VI'].append(pinned['avg_VI'].item())

            sel['disc_transform'] = 'None' if sel['disc_transform'] == 'RandomPixelModifier' else 'RandomPixelModifier'
            values['avg_VI_diff'].append(d.sel(sel)['avg_VI'].item() - pinned['avg_VI'].item())

        for key, value in values.items():
            match key:
                case 'lambda' |  'dataset' | 'lr' | 'network' | 'disc_transform':
                    values[key] = map_param_values(key, value)
                case _:
                    values[key] = ['%.3f' % x for x in value]

        print(values)
        print(f'''
        \\hline
        \\multicolumn{{1}}{{|c|}}{{Metric/Parameter}} & {' & '.join([chr(92) + 'multicolumn{1}{|c|}{' + str(x) + '}' for
                                                                     x in map_param_values(param, param_values)])} \\\\
        \\hline
        {'' if param == 'dataset' else 'Dataset & ' + ' & '.join(values['dataset']) + ' ' + chr(92) + chr(92)}
        {'' if param == 'network' else 'Network & ' + ' & '.join(values['network']) + ' ' + chr(92) + chr(92)}
        {'' if param == 'lr' else 'Discriminator learning rate & ' + ' & '.join(values['lr']) + ' ' + chr(92) + chr(92)}
        {'' if param == 'lambda' else 'cycle consistency $' + chr(92) + 'lambda$ & ' + ' & '.join(values['lambda']) + ' ' + chr(92) + chr(92)}
        \\hline
        slice 1 VI & {' & '.join(values['slice1_VI'])} \\\\
        slice 2 VI & {' & '.join(values['slice2_VI'])} \\\\
        slice 3 VI & {' & '.join(values['slice3_VI'])} \\\\
        average VI & {' & '.join(values['avg_VI'])} \\\\
        average VI difference & {' & '.join(values['avg_VI_diff'])} \\\\
        \\hline
            ''')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help='Which function shall be run')
    parser.add_argument('--gs-result', type=str, help='The path to the grid search NetCDF file')

    args = parser.parse_args()

    if args.function == 'gs-tables':
        print_gs_tables(args.gs_result)
    else:
        raise NotImplementedError(f"Function {args.function} is not implemented")


if __name__ == '__main__':
    main()
