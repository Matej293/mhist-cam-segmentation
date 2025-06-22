import os
import yaml

class ConfigManager:
    def __init__(self, config_path="config/default_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def __str__(self):
        return yaml.dump(self.config, default_flow_style=False)
    
    def _load_config(self):
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file {self.config_path} not found.")
            return {}
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        
        if isinstance(value, str):
            try:
                if '.' in value or 'e' in value.lower():
                    return float(value)
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    return int(value)
            except (ValueError, TypeError):
                pass
                
        return value