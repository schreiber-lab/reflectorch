from reflectorch.data_generation.priors.parametric_models import NuisanceParamsWrapper
from reflectorch.inference import inference_model


class Layer():
    def __init__(self, thickness_bounds, roughness_bounds, sld_bounds, imag_sld_bounds=None):
        self.thickness_bounds = thickness_bounds
        self.roughness_bounds = roughness_bounds
        self.sld_bounds = sld_bounds
        self.imag_sld_bounds = imag_sld_bounds

class Backing():
    def __init__(self, roughness_bounds, sld_bounds, imag_sld_bounds=None):
        self.thickness_bounds = None
        self.roughness_bounds = roughness_bounds
        self.sld_bounds = sld_bounds
        self.imag_sld_bounds = imag_sld_bounds

class Structure():
    def __init__(self, layers, q_shift_bounds=None, r_scale_bounds=None, log10_background_bounds=None):
        self.layers=layers
        self.q_shift_bounds=q_shift_bounds
        self.r_scale_bounds = r_scale_bounds
        self.log10_background_bounds = log10_background_bounds
        self.thicknesses_bounds = []
        self.roughnesses_bounds = []
        self.slds_bounds = []
        self.imag_slds_bounds = []

        for layer in layers:
            if layer.thickness_bounds is not None:
                self.thicknesses_bounds.append(layer.thickness_bounds)
            self.roughnesses_bounds.append(layer.roughness_bounds)
            self.slds_bounds.append(layer.sld_bounds)
            if layer.imag_sld_bounds is not None:
                self.imag_slds_bounds.append(layer.imag_sld_bounds)
        
        self.prior_bounds = self.thicknesses_bounds + self.roughnesses_bounds + self.slds_bounds + self.imag_slds_bounds

        if q_shift_bounds is not None:
            self.prior_bounds += [q_shift_bounds]
        if r_scale_bounds is not None:
            self.prior_bounds += [r_scale_bounds]
        if log10_background_bounds is not None:
            self.prior_bounds += [log10_background_bounds]
    
    def validate_parameters_and_ranges(self, inference_model):
        if len(self.layers) - 1 != inference_model.trainer.loader.prior_sampler.max_num_layers:
            raise ValueError(f'Number of layers mismatch: this model expects {inference_model.trainer.loader.prior_sampler.max_num_layers} layers (backing not included) but you provided {len(self.layers) - 1}')
        

        thickness_ranges = inference_model.trainer.loader.prior_sampler.param_ranges['thicknesses']
        roughness_ranges = inference_model.trainer.loader.prior_sampler.param_ranges['roughnesses']
        sld_ranges = inference_model.trainer.loader.prior_sampler.param_ranges['slds']
        
        def layer_name(i):
            if i == inference_model.trainer.loader.prior_sampler.max_num_layers:
                return 'the backing medium'
            else:
                return f'layer {i+1}'

        for i, layer in enumerate(self.layers):
            if layer.thickness_bounds is not None:
                if layer.thickness_bounds[0] < thickness_ranges[0] or layer.thickness_bounds[1] > thickness_ranges[1]:
                    raise ValueError(f"The provided prior bounds for the thickness of layer {i+1} are outside the training range of the network: {thickness_ranges}")
            if layer.roughness_bounds[0] < roughness_ranges[0] or layer.roughness_bounds[1] > roughness_ranges[1]:
                raise ValueError(f"The provided prior bounds for the roughness of {layer_name(i)} are outside the training range of the network: {roughness_ranges}")
            if layer.sld_bounds[0] < sld_ranges[0] or layer.sld_bounds[1] > sld_ranges[1]:
                raise ValueError(f"The provided prior bounds for the (real) SLD of {layer_name(i)} are outside the training range of the network: {sld_ranges}")
            
            max_sld_bounds_width = inference_model.trainer.loader.prior_sampler.bound_width_ranges['slds'][1]
            if layer.sld_bounds[1] - layer.sld_bounds[0] > max_sld_bounds_width:
                raise ValueError(f"The provided prior bounds for the (real) SLD of {layer_name(i)} have a width (max - min) exceeding the maximum width used for training: {max_sld_bounds_width}")
            
        param_model = inference_model.trainer.loader.prior_sampler.param_model
        if isinstance(param_model, NuisanceParamsWrapper):
            nuisance_params_config = inference_model.trainer.loader.prior_sampler.shift_param_config
                                    
        if self.q_shift_bounds is not None:
            if 'q_shift' not in nuisance_params_config:
                raise ValueError(f'Prior bounds for the q_shift parameter were provided but this parameter is not supported by this model.')
            q_shift_range = inference_model.trainer.loader.prior_sampler.param_ranges['q_shift']
            if self.q_shift_bounds[0] < q_shift_range[0] or self.q_shift_bounds[1] > q_shift_range[1]:
                raise ValueError(f"The provided prior bounds for the q_shift are outside the training range of the network: {q_shift_range}")
            
        if self.r_scale_bounds is not None:
            if 'r_scale' not in nuisance_params_config:
                raise ValueError(f'Prior bounds for the r_scale parameter were provided but this parameter is not supported by this model.')
            r_scale_range = inference_model.trainer.loader.prior_sampler.param_ranges['r_scale']
            if self.r_scale_bounds[0] < r_scale_range[0] or self.r_scale_bounds[1] > r_scale_range[1]:
                raise ValueError(f"The provided prior bounds for the r_scale are outside the training range of the network: {r_scale_range}")
            
        if self.log10_background_bounds is not None:
            if 'log10_background' not in nuisance_params_config:
                raise ValueError(f'Prior bounds for the log10_background parameter were provided but this parameter is not supported by this model.')
            log10_background_range = inference_model.trainer.loader.prior_sampler.param_ranges['log10_background']
            if self.log10_background_bounds[0] < log10_background_range[0] or self.log10_background_bounds[1] > log10_background_range[1]:
                raise ValueError(f"The provided prior bounds for the r_scale are outside the training range of the network: {log10_background_range}")
        
        if isinstance(param_model, NuisanceParamsWrapper):            
            if 'q_shift' in nuisance_params_config and self.q_shift_bounds is None:
                raise ValueError(f'Prior bounds for the q_shift parameter are expected by this model but were not provided.')

            if 'r_scale' in nuisance_params_config and self.r_scale_bounds is None:
                raise ValueError(f'Prior bounds for the r_scale parameter are expected by this model but were not provided.')

            if 'log10_background' in nuisance_params_config and self.log10_background_bounds is None:
                raise ValueError(f'Prior bounds for the log10_background parameter are expected by this model but were not provided.')
        
        print("All checks passed.")

    def get_huggingface_filtering_query(self):
        query = {'dset.prior_sampler.kwargs.max_num_layers': len(self.layers) - 1}
        
        query['dset.prior_sampler.kwargs.param_ranges.thicknesses'] = [min(sl[0] for sl in self.thicknesses_bounds), max(sl[1] for sl in self.thicknesses_bounds)]
        query['dset.prior_sampler.kwargs.param_ranges.roughnesses'] = [min(sl[0] for sl in self.roughnesses_bounds), max(sl[1] for sl in self.roughnesses_bounds)]
        query['dset.prior_sampler.kwargs.param_ranges.slds'] = [min(sl[0] for sl in self.slds_bounds), max(sl[1] for sl in self.slds_bounds)]

        if len(self.imag_slds_bounds) > 0:
            query['dset.prior_sampler.kwargs.model_name'] = 'model_with_absorption'
            query['dset.prior_sampler.kwargs.param_ranges.islds'] = [min(sl[0] for sl in self.imag_slds_bounds), max(sl[1] for sl in self.imag_slds_bounds)]
        else:
            query['dset.prior_sampler.kwargs.model_name'] = 'standard_model'

        if self.q_shift_bounds is not None:
            query['dset.prior_sampler.kwargs.shift_param_config.q_shift'] = True
            query['dset.prior_sampler.kwargs.param_ranges.q_shift'] = self.q_shift_bounds

        if self.r_scale_bounds is not None:
            query['dset.prior_sampler.kwargs.shift_param_config.r_scale'] = True
            query['dset.prior_sampler.kwargs.param_ranges.r_scale'] = self.r_scale_bounds

        if self.log10_background_bounds is not None:
            query['dset.prior_sampler.kwargs.shift_param_config.log10_background'] = True
            query['dset.prior_sampler.kwargs.param_ranges.log10_background'] = self.log10_background_bounds

        return query
    

if __name__ == '__main__':
    from reflectorch.inference.query_matcher import HuggingfaceQueryMatcher
    from reflectorch import EasyInferenceModel

    layer1 = Layer(thickness_bounds=[1, 1000], roughness_bounds=[0, 60], sld_bounds=[-2, 2])
    layer2 = Layer(thickness_bounds=[1, 50], roughness_bounds=[0, 10], sld_bounds=[1, 4])
    backing = Backing(roughness_bounds=[0, 15], sld_bounds=[0, 3])

    structure = Structure(
        layers=[layer1, layer2, backing], 
        r_scale_bounds=[0.9, 1.1],
        log10_background_bounds=[-8, -5],
    )

    print(structure.prior_bounds)

    query_matcher = HuggingfaceQueryMatcher(repo_id='valentinsingularity/reflectivity')
    filtering_query = structure.get_huggingface_filtering_query()
    print(filtering_query)

    matching_configs = query_matcher.get_matching_configs(filtering_query)
    print(f'Matching configs: {matching_configs}')


    inference_model = EasyInferenceModel(config_name=matching_configs[0])
    structure.validate_parameters_and_ranges(inference_model)