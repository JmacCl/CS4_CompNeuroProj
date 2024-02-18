import configparser
import os
import yaml
from pathlib import Path
# from src.logging_conf import logger

def path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path

def reformat_ini_dict(ini_dict):
    pass

class BraTS2020Configuration:

    def __init__(self, config):
        # Check if path exists
        data = self.__determine_file(config)
        # Process data into a python dictionary, depending on if it is accepted
        self.data_creation = data["data_creation"]
        self.training = data["training"]
        self.testing = data["testing"]
        self.graphing = data["graphing"]

    def __determine_file(self, path):
        """
        This function process the file, it determines if the path exists and then process
        the configuration file if it is a supported file format.
        It will process the file into a python dictionary
        :param path:
        :return:
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = {}
        # Check if data is of a YAML format
        if path.lower().endswith((".yaml", ".yml")):
            with open(path, "r") as file:
                yaml_data = yaml.safe_load_all(file)
                for d in yaml_data:
                    for key, val in d.items():
                        data[key] = val
            return data
        # elif path.lower().endswith(".ini"):
        #     ini_config = configparser.ConfigParser()
        #     ini_config.read(path)
        #
        #     for section in ini_config.sections():
        #         data[section] = dict(ini_config.items(section))
        #     return data
        else:
            raise IOError("Invalid configuration file format, see READEME for details")


    def __set_up_trains_params(self):
        # First define how training_utils parameters should be set up
        return_dic = {}

        config = self.config
        return_dic["main"] = config["main"]
        return_dic["training"] = config["training"]
        return_dic["alg_set_up"] = config["alg_set_up"]

        return return_dic

    def data_creation_params(self):
        pass

    def get_training_params(self):
        return self.trains_params

    def get_testing_params(self):
        pass

    def get_graphing_params(self):
        pass


