from types import SimpleNamespace

def parse_config(config: dict):
    """
    parse a configuration dictionary
    """
    parse_result = SimpleNamespace()

    for key, value in config.items():
        if isinstance(value, dict):
            setattr(parse_result, key, parse_config(value))
        else:
            setattr(parse_result, key, value)
    
    return parse_result