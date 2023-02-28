import numpy as np

from reflectorch.inference import standard_preprocessing, StandardPreprocessing


def test_preprocessing(raw_data_with_preprocessing_params):
    raw_data, preprocessing_params = raw_data_with_preprocessing_params
    res = standard_preprocessing(**raw_data, **preprocessing_params)

    assert "curve_interp" in res
    assert "curve" in res
    assert "q_values" in res
    assert "q_interp" in res

    assert res["curve_interp"].shape == res["q_interp"].shape
    assert res["curve"].shape == res["q_values"].shape
    assert len(res["curve_interp"].shape) == 1
    assert len(res["curve"].shape) == 1


def test_preprocessing_obj(raw_data_with_preprocessing_params):
    raw_data, preprocessing_params = raw_data_with_preprocessing_params
    func_res = standard_preprocessing(**raw_data, **preprocessing_params)
    st_p = StandardPreprocessing()

    obj_res = st_p.preprocess(**raw_data, **preprocessing_params)

    for key in func_res:
        assert np.all(func_res[key] == obj_res[key])

    st_p.set_parameters(**preprocessing_params)
    obj_res = st_p(**raw_data)

    for key in func_res:
        assert np.all(func_res[key] == obj_res[key])


