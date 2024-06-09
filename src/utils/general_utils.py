import yaml

def read_config(config_path: str) -> dict:
    """Reads a YAML configuration file and returns it as a dictionary.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        The configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)  # Use safe_load for security
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    return config

# Example usage:
if __name__ == "__main__":
    config_file = "config.yaml"  # Replace with your actual config file path
    config = read_config(config_file)

    print(f"Number of classes: {config['num_classes']}")
    print(f"Image size: {config['img_height'], config['img_width']}")
    print(f"Learning rate: {config['learning_rate']}")
