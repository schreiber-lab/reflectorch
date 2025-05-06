import numpy as np
from functools import reduce
from operator import or_

from reflectorch.inference.inference_model import EasyInferenceModel
from reflectorch import BasicParams

import refnx
from refnx.dataset import ReflectDataset, Data1D
from refnx.analysis import Transform, CurveFitter, Objective, Model, Parameter
from refnx.reflect import SLD, Slab, ReflectModel

def covert_reflectorch_prediction_to_refnx_structure(inference_model: EasyInferenceModel, pred_params_object: BasicParams,  prior_bounds: np.array):
    assert inference_model.trainer.loader.prior_sampler.param_model.__class__.__name__ == 'StandardModel'

    n_layers = inference_model.trainer.loader.prior_sampler.max_num_layers
    init_thicknesses = pred_params_object.thicknesses.squeeze().tolist()
    init_roughnesses = pred_params_object.roughnesses.squeeze().tolist()
    init_slds = pred_params_object.slds.squeeze().tolist()

    sld_objects = []

    for sld in init_slds:
        sld_objects.append(SLD(value=sld))

    layer_objects = [SLD(0)()]
    for i in range(n_layers):
        layer_objects.append(sld_objects[i](init_thicknesses[i], init_roughnesses[i]))

    layer_objects.append(sld_objects[-1](0, init_roughnesses[-1]))

    thickness_bounds = prior_bounds[:n_layers]
    roughness_bounds = prior_bounds[n_layers:2*n_layers+1]
    sld_bounds = prior_bounds[2*n_layers+1:]

    for i, layer in enumerate(layer_objects):
        if i == 0:
            print("Ambient (air)")
            print(80 * '-')
        elif i < n_layers+1:
            layer.thick.setp(bounds=thickness_bounds[i-1], vary=True)
            layer.rough.setp(bounds=roughness_bounds[i-1], vary=True)
            layer.sld.real.setp(bounds=sld_bounds[i-1], vary=True)

            print(f'Layer {i}')
            print(f'Thickness: value {layer.thick.value}, vary {layer.thick.vary}, bounds {layer.thick.bounds}')
            print(f'Roughness: value {layer.rough.value}, vary {layer.rough.vary}, bounds {layer.rough.bounds}')
            print(f'SLD: value {layer.sld.real.value}, vary {layer.sld.real.vary}, bounds {layer.sld.real.bounds}')
            print(80 * '-')
        else: #substrate
            layer.rough.setp(bounds=roughness_bounds[i-1], vary=True)
            layer.sld.real.setp(bounds=sld_bounds[i-1], vary=True)

            print(f'Substrate')
            print(f'Thickness: value {layer.thick.value}, vary {layer.thick.vary}, bounds {layer.thick.bounds}')
            print(f'Roughness: value {layer.rough.value}, vary {layer.rough.vary}, bounds {layer.rough.bounds}')
            print(f'SLD: value {layer.sld.real.value}, vary {layer.sld.real.vary}, bounds {layer.sld.real.bounds}')
    
    refnx_structure = reduce(or_, layer_objects)

    return refnx_structure


###Example usage:
# refnx_structure = covert_reflectorch_prediction_to_refnx_structure(inference_model, pred_params_object, prior_bounds)

# refnx_reflect_model = ReflectModel(refnx_structure, bkg=1e-10, dq=0.0)
# refnx_reflect_model.scale.setp(bounds=(0.8, 1.2), vary=True)
# refnx_reflect_model.q_offset.setp(bounds=(-0.01, 0.01), vary=True)
# refnx_reflect_model.bkg.setp(bounds=(1e-10, 1e-8), vary=True)


# data = Data1D(data=(q_model, exp_curve_interp))

# refnx_objective = Objective(refnx_reflect_model, data, transform=Transform("logY"))
# fitter = CurveFitter(refnx_objective)
# fitter.fit('least_squares')