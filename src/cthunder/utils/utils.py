import glob
import importlib
import json
import os
from typing import Any, Dict, List, Optional, Union, Tuple

import yaml
from loguru import logger


def open_yml(path: str) -> Optional[Dict]:
    """Open a .yml or .yaml file.

    :param path: the path where is stored the .yml/.yaml file.
    """
    if path.endswith(".yml") or path.endswith(".yaml"):
        with open(path) as file:
            try:
                result = yaml.safe_load(file)
                return result
            except yaml.YAMLError as exc:
                logger.debug(f"ERROR IN READING .YML FILE : {exc}")
                return None
    else:
        logger.debug(f"NOT A .YML or .YAML FILE : {path}")
        return None


def save_to_yaml(path: str, document: Dict) -> None:
    """
    Save a dictionary to yaml file.
    :param document: the json file to save
    :param path: path where to save the file.
    """
    if not (path.endswith(".yml") or path.endswith(".yaml")):
        path = path + ".yml"
    with open(path, "w") as yaml_file:
        yaml.dump(document, yaml_file)


def listdir_or_py(
    path: str, to_remove: List[str] = ["__pycache__", "__init__.py", "wandb"]
):
    """List the directory or python files.
    Args
        :param path: the path to get the directories from
    """
    # Get all not hidden python files.
    dirs = glob.glob(os.path.join(path, "**/*.py"), recursive=True)
    # Remove not wanted files or directories
    new_dirs = []
    for file in dirs:
        conditions = [file.endswith(x) for x in to_remove]
        if not any(conditions):
            new_dirs.append(file)
    return new_dirs


def instantiate_class_from_name(path: str, class_name: str) -> Optional[Any]:
    """Return the class from the name.

    :param path: the directory where to search the class
    :param class_name: the name of the class to instantiate
    """
    files = listdir_or_py(path)
    files_from_helper = listdir_or_py("thunder")
    files = files + files_from_helper
    for file in files:
        class_inst = get_class_from_file(file, class_name)
        if class_inst is not None:
            return class_inst
    return None


def get_class_from_file(python_file: str, class_name: str) -> Optional[Any]:
    """Return the instance class from the python file.

    :param python_file: the path to a python file
    :param class_name: the name of the class to instantiate
    :returns the class that is inside this file if it exists
    Example:
        if there is a class A in a file : example/folder/file.py
        the inputs are the following :
            python_file: example/folder/file.py
            class_name: "A"
    """
    module = python_file.replace(os.sep, ".").replace(".py", "")
    try:
        class_inst = getattr(importlib.import_module(module), class_name)
        return class_inst
    except (ImportError, AttributeError):
        return None


def read_json(path: str) -> Dict:
    """Read the json file.
    Args
    :path the path to the file
    :return a dictionary of the json file
    """
    if path.endswith(".json"):
        with open(path, "rb") as f:
            content = json.loads(f.read())
            return content
    else:
        logger.debug("Not a json file")
        return {}


def save_json(object: Dict, path: str):
    """Save the dictionary into a json file.
    Args
    :param object: the object to save
    :param path: the path where to save. Could have .json or not in the path
    """
    if path.endswith(".json"):
        path_to_save = path
    else:
        path_to_save = path + ".json"
    with open(path_to_save, "w") as file:
        json.dump(object, file)


def instantiate_class_from_init(init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.
    Code original : "from lightning.pytorch.cli import instantiate_class"

    Args:
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(**kwargs)
