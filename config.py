import os


def root_path():
    return os.path.dirname(__file__)


def dataset_path():
    return os.path.join(root_path(),"dataset")


def src_path():
    return os.path.join(root_path(),"src")

def output_path():
    return os.path.join(root_path(),"output")

def weight_path():
    return os.path.join(root_path(),"weight")
