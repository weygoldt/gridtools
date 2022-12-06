import yaml


class ConfLoader:
    def __init__(self, path: str) -> None:
        
        with open(path) as file:
            
            try:
                conf = yaml.safe_load(file)
                for key in conf:
                    setattr(self, key, conf[key])
            
            except yaml.YAMLError as error:
                # logger.error("Failed to open yaml file!")
                raise error


conf = ConfLoader("datacleaner_conf.yml")
    