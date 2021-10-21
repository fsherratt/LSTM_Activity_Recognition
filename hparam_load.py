"""
This file contains all the required files for loading and setting hyper paramaters
indicated wiht the @HP_ tag in configuration files
"""
import copy
import itertools
import yaml
import numpy as np
import sys


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

            if key in hyper_param_dict:
                _set_in_dict(dictionary, path, hyper_param_dict.get(key))
                print("{} set to {}".format(key, hyper_param_dict.get(key)))

            else:
                print(
                    "HParam dictionary does not contain key {}".format(key),
                    file=sys.stderr,
                )


def _generate_hparam_sets(hparam_set):
    """
    Generate all sets of possible hyper paramaters
    """
    items = list(hparam_set.values())
    product_hparam_set = list(itertools.product(*items))
    return [dict(zip(hparam_set.keys(), x)) for x in product_hparam_set]


def _parse_hparam_dict(hparam_set) -> dict:
    if hparam_set is None:
        return None

    hparam_dict = {}

    for param in hparam_set:
        if param["type"] == "range":
            if param["dtype"] == "float" or param["dtype"] != "int":
                dtype = np.float
            else:
                dtype = np.int
            if param["steps"]:
                param_list = list(
                    np.linspace(param["min"], param["max"], param["steps"], dtype=dtype)
                )
            else:
                param_list = list(
                    np.arange(
                        param["min"], param["max"], param["interval"], dtype=dtype
                    )
                )

        elif param["type"] == "const":
            param_list = [param["value"]]

        elif param["type"] in ["set", "list"]:
            param_list = []
            for item in param["items"]:
                if (
                    not isinstance(item, dict)
                    and isinstance(item, list)
                    and isinstance(item[0], dict)
                ):
                    param_list.append(tuple(_parse_hparam_dict(item)))
                elif (
                    not isinstance(item, dict)
                    and isinstance(item, list)
                    or not isinstance(item, dict)
                ):
                    param_list.append(item)
                else:
                    param_list.append(tuple(_parse_hparam_dict([item])))
        else:
            continue

        hparam_dict[param["hparam"]] = param_list

    hparam_list = _generate_hparam_sets(hparam_dict)

    # Flatten any boolean hyper paramater sets
    new_hparam_list = []
    for item in hparam_list:
        flattened_list = _flatten_bool_dict(item)

        if not flattened_list:
            new_hparam_list.append(item)
        else:
            new_hparam_list.extend(flattened_list)

    return new_hparam_list


def _flatten_bool_dict(item):
    new_list = []
    dict_list = {key: value for key, value in item.items() if isinstance(value, tuple)}

    if dict_list:
        expanded_list = _generate_hparam_sets(dict_list)
        for hp_row in expanded_list:

            new_dict = copy.deepcopy(item)
            for hp_key, hp_item in hp_row.items():
                new_dict = {**new_dict, **hp_item}
                del new_dict[hp_key]

            new_list.append(new_dict)

    return new_list


def load_hparam_set(path: str) -> dict:
    """
    Load hyper paramaters from yaml file
    """
    with open(path) as file:
        hparam_set = yaml.full_load(file)

    if "hparams" not in hparam_set:
        return [None]

    hparam_dict = _parse_hparam_dict(hparam_set["hparams"])

    if hparam_dict is None:
        return [None]

    return hparam_dict


def set_hparams(hparam_dict, param_config):
    """
    Update config file replacing items marked with @HP_ tag
    """
    tmp_config = copy.deepcopy(param_config)

    _replace_hparam(hparam_dict, tmp_config, None, None)

    return tmp_config
