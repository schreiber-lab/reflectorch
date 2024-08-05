from typing import Tuple, Union
from pathlib import Path
import subprocess

from reflectorch.paths import RUN_SCRIPTS_DIR


def save_sbatch_and_run(
        name: str,
        args: str,
        time: str,
        partition: str = None,
        reservation: bool = False,
        chdir: str = '~/maxwell_output',
        run_dir: Path = None,
        confirm: bool = False,
) -> Union[Tuple[str, str], None]:
    run_dir = Path(run_dir) if run_dir else RUN_SCRIPTS_DIR
    sbatch_path = run_dir / f'{name}.sh'

    if sbatch_path.is_file():
        import warnings
        warnings.warn(f'Sbatch file {str(sbatch_path)} already exists!')
        if confirm and not confirm_input('Continue?'):
            return

    file_content = _generate_sbatch_str(
        name,
        args,
        time=time,
        reservation=reservation,
        partition=partition,
        chdir=chdir,
    )

    if confirm:
        print(f'Generated file content: \n{file_content}\n')
        if not confirm_input(f'Save to {str(sbatch_path)} and run?'):
            return

    with open(str(sbatch_path), 'w') as f:
        f.write(file_content)

    res = submit_job(str(sbatch_path))
    return res


def _generate_sbatch_str(name: str,
                         args: str,
                         time: str,
                         partition: str = None,
                         reservation: bool = False,
                         chdir: str = '~/maxwell_output',
                         entry_point: str = 'python -m reflectorch.train',
                         ) -> str:
    chdir = str(Path(chdir).expanduser().absolute())
    partition_keyword = 'reservation' if reservation else 'partition'

    return f'''#!/bin/bash
#SBATCH --chdir {chdir}
#SBATCH --{partition_keyword}={partition}
#SBATCH --constraint=P100
#SBATCH --nodes=1
#SBATCH --job-name {name}
#SBATCH --time={time}
#SBATCH --output {name}.out
#SBATCH --error {name}.err

{entry_point} {args}
'''


def confirm_input(message: str) -> bool:
    yes = ('y', 'yes')
    no = ('n', 'no')
    res = ''
    valid_results = list(yes) + list(no)
    message = f'{message} Y/n: '

    while res not in valid_results:
        res = input(message).lower()
    return res in yes


def submit_job(sbatch_path: str) -> Tuple[str, str]:
    process = subprocess.Popen(
        ['sbatch', str(sbatch_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode()
