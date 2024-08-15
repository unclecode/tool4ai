import os
import yaml
from typing import Any, Dict

class ConfigManager:
    DEFAULT_CONFIG = {
        'llm': {
            # 'model': 'groq/llama3.1-70b-versatile',
            # 'model': 'claude-3-5-sonnet-20240620',
            # 'model': 'gpt-4o',
            # 'model': 'gpt-4o-2024-08-06',
            # 'model': 'gpt-4o-mini',
            'model': "gpt-4o-mini-2024-07-18",
            # 'model': 'claude-3-haiku-20240307',
            # 'model': 'claude-3-sonnet-20240229',
            # 'model': 'groq/llama3-groq-70b-8192-tool-use-preview',
            # 'model': 'mistral/mistral-large-latest',
            # 'model': 'groq/llama-3.1-70b-versatile',
            # 'model': 'gemini/gemini-1.5-flash',
            # 'model': 'gemini/gemini-pro',
            # 'model': 'ollama/gemma2',
            # 'model': 'openrouter/qwen/qwen-2-72b-instruct',
            'max_tokens': 1000,
        },
        'router': {
            'strategy': 'default',
        },
        'toolkit': {
            'max_tools': 100,
        },
        'dependency_graph': {
            'visualization_engine': 'graphviz',
        }
    }

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.expanduser('~/.tool4ai/config.yaml')
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            return self.merge_configs(self.DEFAULT_CONFIG, user_config)
        return self.DEFAULT_CONFIG.copy()

    def merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in user.items():
            if isinstance(value, dict) and key in default:
                default[key] = self.merge_configs(default[key], value)
            else:
                default[key] = value
        return default

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        return value

    def save(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

config_manager = ConfigManager()