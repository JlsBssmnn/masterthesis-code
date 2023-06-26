import re
import pandas as pd
from collections import defaultdict

def parse_losses(file):
    f = open(file, 'r')
    data = defaultdict(lambda: ([], []))

    for line in f:
            line = line.strip()
            if re.search(r'epoch: \d*, iters: \d*, time: \d*\.\d*, data: \d*\.\d*', line) is not None:
                iteration = re.search(r'iters: \d*', line)
                assert iteration is not None
                iteration = int(iteration.group()[7:])
                losses = line[line.find(')') + 1:]
                matches = re.findall(r'[\w,_]*: -?\d*\.\d*', losses)

                for match in matches:
                    key = re.search(r'[\w,_]*', match)
                    value = re.search(r'-?\d*\.\d*', match)
                    assert key is not None and value is not None
                    key = key.group()
                    value = value.group()
                    data[key][0].append(iteration)
                    data[key][1].append(float(value))
    f.close()
    return pd.DataFrame({key: pd.Series(value[1], value[0]) for key, value in data.items()})
