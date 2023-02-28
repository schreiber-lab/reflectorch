import numpy as np

from reflectorch import *

from tests.fixtures.data import TEST_DATA_PATH


def generate_test_data_by_trainer(file_name: str, model_name: str = 'l2q64_new_sub_1'):
    trainer = get_trainer_by_name(model_name, load_weights=False)
    data = trainer.loader.get_batch(1)
    noisy_curve = trainer.loader.curves_scaler.restore(data['scaled_noisy_curves'])[0].cpu().numpy()
    params = data['params'].as_tensor(add_bounds=False)[0].cpu().numpy()
    priors = np.stack([
        data['params'].min_bounds[0].cpu().numpy(), data['params'].max_bounds[0].cpu().numpy()
    ], -1)
    np.savez(TEST_DATA_PATH / f'{file_name}.npz', preprocessed_curve=noisy_curve, params=params, priors=priors)


if __name__ == '__main__':
    generate_test_data_by_trainer('test_preprocessed_curve_1')
