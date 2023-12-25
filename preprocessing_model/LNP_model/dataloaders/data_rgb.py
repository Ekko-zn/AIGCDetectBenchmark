import os
from .dataset_rgb import DataLoaderVal , DataLoaderTest, DataLoader_NoisyData

def get_validation_data(rgb_dir,noise_type):

    return DataLoaderVal(rgb_dir,noise_type)


def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir)

def get_rgb_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoader_NoisyData(rgb_dir)