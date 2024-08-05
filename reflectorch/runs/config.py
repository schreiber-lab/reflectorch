import yaml

from pathlib import Path
from reflectorch.paths import CONFIG_DIR


def load_config(config_name: str, config_dir: str = None) -> dict:
    """Loads a configuration dictionary from a YAML configuration file located in the configuration directory

    Args:
        config_name (str): name of the YAML configuration file
        config_dir (str): path of the configuration directory

    Returns:
        dict: the configuration dictionary
    """
    if not config_name.endswith('.yaml'):
        config_name = f'{config_name}.yaml'
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    path = config_dir / config_name
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    config['config_path'] = str(path.absolute())

    return config
