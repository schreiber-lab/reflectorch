import click

from reflectorch.runs.slurm_utils import save_sbatch_and_run
from reflectorch.runs.utils import train_from_config
from reflectorch.runs.config import load_config

__all__ = [
    'run_train',
    'run_train_on_cluster',
    'run_test_config',
]


@click.command()
@click.argument('config_name', type=str)
def run_train(config_name: str):
    """Runs the training from the command line interface
       Example: python -m reflectorch.train 'conf_name'

    Args:
        config_name (str): name of the YAML configuration file
    """
    config = load_config(config_name)
    train_from_config(config)


@click.command()
@click.argument('config_name', type=str)
@click.argument('batch_size', type=int, default=512)
@click.argument('num_iterations', type=int, default=10)
def run_test_config(config_name: str, batch_size: int, num_iterations: int):
    """Run for the purpose of testing the configuration file.
       Example: python -m reflectorch.test_config 'conf_name.yaml' 512 10

    Args:
        config_name (str): name of the YAML configuration file
        batch_size (int): overwrites the batch size in the configuration file
        num_iterations (int): overwrites the number of iterations in the configuration file
    """
    config = load_config(config_name)
    config = _change_to_test_config(config, batch_size=batch_size, num_iterations=num_iterations)
    train_from_config(config)


@click.command()
@click.argument('config_name')
def run_train_on_cluster(config_name: str):
    config = load_config(config_name)
    name = config['general']['name']
    slurm_conf = config['slurm']

    res = save_sbatch_and_run(
        name,
        config_name,
        time=slurm_conf['time'],
        partition=slurm_conf['partition'],
        reservation=slurm_conf.get('reservation', False),
        chdir=slurm_conf.get('chdir', '~/maxwell_output'),
        run_dir=slurm_conf.get('run_dir', None),
        confirm=slurm_conf.get('confirm', True),
    )
    if not res:
        print('Aborted.')
        return
    out, err = res

    if err:
        print('Error occurred: ', err)
    else:
        print('Success!', out)


def _change_to_test_config(config, batch_size: int, num_iterations: int):
    config = dict(config)
    config['training']['num_iterations'] = num_iterations
    config['training']['batch_size'] = batch_size
    config['training']['update_tqdm_freq'] = 1
    return config
