# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import yaml

from reflectorch.paths import CONFIG_DIR


def load_config(config_name: str) -> dict:
    if not config_name.endswith('.yaml'):
        config_name = f'{config_name}.yaml'
    path = CONFIG_DIR / config_name
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    config['config_path'] = str(path.absolute())
    return config
