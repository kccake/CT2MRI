import yaml

def load_yaml(config_path, encoding='utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.safe_load(f)