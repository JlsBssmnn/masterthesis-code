import re

def parse_losses(file) -> list[dict[str, float]]:
    f = open(file, 'r')
    data = []
    for line in f:
            line = line.strip()
            if re.search(r'epoch: \d*, iters: \d*, time: \d*\.\d*, data: \d*\.\d*', line) is not None:
                iteration = re.search(r'iters: \d*', line)
                assert iteration is not None
                iteration = int(iteration.group()[7:])
                losses = line[line.find(')') + 1:]
                matches = re.findall(r'[\w,_]*: -?\d*\.\d*', losses)
                parsed: dict[str, int | float] = {'iters': iteration}


                for match in matches:
                    key = re.search(r'[\w,_]*', match)
                    value = re.search(r'-?\d*\.\d*', match)
                    assert key is not None and value is not None
                    key = key.group()
                    value = value.group()
                    parsed[key] = float(value)
                data.append(parsed)
    f.close()
    return data
