import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()
    print(cfg)