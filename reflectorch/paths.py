from typing import Union
from pathlib import Path

__all__ = [
    'ROOT_DIR',
    'SAVED_MODELS_DIR',
    'SAVED_LOSSES_DIR',
    'RUN_SCRIPTS_DIR',
    'CONFIG_DIR',
    'TESTS_PATH',
    'TEST_DATA_PATH',
    'listdir',
]

ROOT_DIR: Path = Path(__file__).parents[1]
SAVED_MODELS_DIR: Path = ROOT_DIR / 'saved_models'
SAVED_LOSSES_DIR: Path = ROOT_DIR / 'saved_losses'
RUN_SCRIPTS_DIR: Path = ROOT_DIR / 'runs'
CONFIG_DIR: Path = ROOT_DIR / 'configs'
TESTS_PATH: Path = ROOT_DIR / 'tests'
TEST_DATA_PATH = TESTS_PATH / 'data'


def listdir(path: Union[Path, str], pattern: str = '*', recursive: bool = False, *, sort_key=None, reverse=False):
    path = Path(path)
    func = path.rglob if recursive else path.glob
    return sorted(list(func(pattern)), key=sort_key, reverse=reverse)
