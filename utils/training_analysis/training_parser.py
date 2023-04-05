import re

def parse_losses(file) -> list[dict[str, float]]:
    f = open(file, 'r')
    data = []
    for line in f:
            line = line.strip()
            if 'visualizer.py' in line and \
                    re.search('epoch: \d*, iters: \d*, time: \d*\.\d*, data: \d*\.\d*', line) is not None:
                iteration = int(re.search('iters: \d*', line).group()[7:])
                losses = line[line.find(')') + 1:]
                matches = re.findall('[\w,_]*: \d*\.\d*', losses)
                parsed = {'iters': iteration}


                for match in matches:
                    key = re.search('[\w,_]*', match)
                    value = re.search('\d*\.\d*', match)
                    assert key is not None and value is not None
                    key = key.group()
                    value = value.group()
                    parsed[key] = float(value)
                data.append(parsed)
    f.close()
    return data
