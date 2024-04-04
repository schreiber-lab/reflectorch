# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import yaml

from reflectorch.paths import CONFIG_DIR


def load_config(config_name: str) -> dict:
    """Initializes a configuration dictionary from a configuration file in the configuration directory

    Args:
        config_name (str): name of the configuration file

    Returns:
        dict: the configuration dictionary
    """
    if not config_name.endswith('.yaml'):
        config_name = f'{config_name}.yaml'
    path = CONFIG_DIR / config_name
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    config['config_path'] = str(path.absolute())
    
    return config
