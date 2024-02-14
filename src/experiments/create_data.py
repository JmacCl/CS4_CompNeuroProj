import sys
import os

from src.data.data_processors.data_processors import BratsDataProcessor
from src.experiments.configs.config import BraTS2020Configuration

def create_data(config):
    BratsDataProcessor(config)



if __name__ == "__main__":
    print(sys.argv)
    cwd = os.getcwd()
    config_path = os.path.join(cwd, sys.argv[1])
    config = BraTS2020Configuration(config_path)
    create_data(config.data_creation)
