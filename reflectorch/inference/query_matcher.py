import os
import tempfile
import yaml
from huggingface_hub import hf_hub_download, list_repo_files

class HuggingfaceQueryMatcher:
    """Downloads the available configurations files to a temporary directory and provides functionality for filtering those configuration files matching user specified queries.
    
    Args:
        repo_id (str): The Hugging Face repository ID.
        config_dir (str): Directory within the repo where YAML files are stored.
    """
    def __init__(self, repo_id='valentinsingularity/reflectivity', config_dir='configs'):
        self.repo_id = repo_id
        self.config_dir = config_dir
        self.cache = {
            'parsed_configs': None,
            'temp_dir': None
        }
        self._renew_cache()

    def _renew_cache(self):
        temp_dir = tempfile.mkdtemp()
        print(f"Temporary directory created at: {temp_dir}")

        repo_files = list_repo_files(self.repo_id, repo_type='model')
        config_files = [file for file in repo_files if file.startswith(self.config_dir) and file.endswith('.yaml')]

        downloaded_files = []
        for file in config_files:
            file_path = hf_hub_download(repo_id=self.repo_id, filename=file, local_dir=temp_dir, repo_type='model')
            downloaded_files.append(file_path)
        
        parsed_configs = {}
        for file_path in downloaded_files:
            with open(file_path, 'r') as file:
                config_data = yaml.safe_load(file)
                file_name = os.path.basename(file_path)
                parsed_configs[file_name] = config_data
        
        self.cache['parsed_configs'] = parsed_configs
        self.cache['temp_dir'] = temp_dir

    def get_matching_configs(self, query):
        """retrieves configuration files that match the user specified query.
        
        Args:
            query (dict): Dictionary of key-value pairs to filter configurations, e.g. ``query = {'dset.prior_sampler.kwargs.max_num_layers': 3, 'dset.prior_sampler.kwargs.param_ranges.slds': [0., 100.]}``.
                          For keys containing the ``param_ranges`` subkey a configuration is selected if the value of the query (i.e. desired parameter range) 
                          is a subrange of the parameter range in the configuration, in all other cases the values must match exactly.
        
        Returns:
            list: List of file names that match the query.
        """
        
        filtered_configs = []
        
        for file_name, config_data in self.cache['parsed_configs'].items():
            if self.matches_query(config_data, query):
                filtered_configs.append(file_name)
        
        return filtered_configs
    
    def matches_query(self, config_data, query):
        for q_key, q_value in query.items():
            keys = q_key.split('.')
            value = self.deep_get(config_data, keys)
            if 'param_ranges' in keys:
                if q_value[0] < value[0] or q_value[1] > value[1]:
                    return False
            else: 
                if value != q_value:
                    return False
                
        return True

    def deep_get(self, d, keys):
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key, None)

        return d