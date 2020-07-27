"""
This file contains all the required files for loading and setting hyper paramaters
indicated wiht the @HP_ tag in configuration files
"""
import copy
import pathlib
import itertools
import yaml
import numpy as np


def _get_from_dict(data_dict, map_list):
    for k in map_list:
        data_dict = data_dict[k]
    return data_dict


def _set_in_dict(data_dict, map_list, value):
    _get_from_dict(data_dict, map_list[:-1])[map_list[-1]] = value


def _replace_hparam(hyper_param_dict, dict_item, dictionary=None, path=None):
    if path is None:
        path = []

    if dictionary is None:
        dictionary = dict_item

    if isinstance(dict_item, dict):
        for key, value in dict_item.items():
            new_path = copy.copy(path)
            new_path.append(key)
            _replace_hparam(hyper_param_dict, value, dictionary, new_path)

    elif isinstance(dict_item, list):
        for index, item in enumerate(dict_item):
            new_path = copy.copy(path)
            new_path.append(index)
            _replace_hparam(hyper_param_dict, item, dictionary, new_path)

    elif isinstance(dict_item, str):
        if str.startswith(dict_item, "@HP_"):
            key = dict_item[1:]

            if hyper_param_dict.__contains__(key):
                _set_in_dict(dictionary, path, hyper_param_dict.get(key))
                print("{} set to {}".format(key, hyper_param_dict.get(key)))

            else:
                raise KeyError("HParam dictionary does not contain key {}".format(key))


def _generate_hparam_sets(hparam_set):
    """
    Generate all sets of possible hyper paramaters
    """
    items = list(hparam_set.values())
    hparam_sets = list(itertools.product(*items))
    hparam_sets = [dict(zip(hparam_set.keys(), x)) for x in hparam_sets]

    return hparam_sets


def load_hparam_set(path: str) -> dict:
    """
    Load hyper paramaters from yaml file
    """
    with open(path) as file:
        hparam_set = yaml.full_load(file)

    hparam_dict = {}

    for param in hparam_set["hparams"]:
        if param["type"] == "range":
            param_list = list(np.arange(param["min"], param["max"], param["interval"]))

        elif param["type"] == "list":
            param_list = param["items"]

        hparam_dict[param["hparam"]] = param_list

    return _generate_hparam_sets(hparam_dict)


def set_hparams(hparam_dict, param_config):
    """
    Update config file replacing items marked with @HP_ tag
    """
    tmp_config = copy.deepcopy(param_config)

    _replace_hparam(hparam_dict, tmp_config, None, None)

    return tmp_config


if __name__ == "__main__":
    _hparam_path = pathlib.Path("conf\\hparam.yaml")
    _hparam_sets = load_hparam_set(_hparam_path)

    _model_path = pathlib.Path("conf\\model.yaml")
    with open(_model_path) as _file:
        _config = yaml.full_load(_file)

    for _param_set in _hparam_sets:
        _tmp_config = set_hparams(_param_set, copy.deepcopy(_config))
