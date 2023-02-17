# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
from pathlib import Path

__all__ = [
    'ROOT_DIR',
    'SAVED_MODELS_DIR',
    'SAVED_LOSSES_DIR',
    'RUN_SCRIPTS_DIR',
    'CONFIG_DIR',
    'listdir',
]

ROOT_DIR: Path = Path(__file__).parents[1]
SAVED_MODELS_DIR: Path = ROOT_DIR / 'saved_models'
SAVED_LOSSES_DIR: Path = ROOT_DIR / 'saved_losses'
RUN_SCRIPTS_DIR: Path = ROOT_DIR / 'runs'
CONFIG_DIR: Path = ROOT_DIR / 'configs'


def listdir(path: Union[Path, str], pattern: str = '*', recursive: bool = False, *, sort_key=None, reverse=False):
    path = Path(path)
    func = path.rglob if recursive else path.glob
    return sorted(list(func(pattern)), key=sort_key, reverse=reverse)
