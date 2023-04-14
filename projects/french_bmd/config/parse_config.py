import yaml
import collections
from pathlib import Path
ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "./default.yaml"

def _recursive_update(d, u):
    if d is None:
        return u
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = _recursive_update(d.get(k, {}), v)
        elif k not in d:
            d[k] = v
    return d
def parse_config(yaml_file, default_values_file=DEFAULT_CONFIG):
    with open(default_values_file, "r") as f:
        default_values = yaml.safe_load(f)
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    out = _recursive_update(data, default_values)
    return out

if __name__ == '__main__':
    config = 'default.yaml'
    data = parse_config(config)
