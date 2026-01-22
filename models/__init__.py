import glob
import importlib
import os

# Get the current directory of the models package
models_dir = os.path.dirname(__file__)

# Find all Python files in the models directory, excluding __init__.py
model_files = glob.glob(os.path.join(models_dir, "*.py"))
model_modules = [os.path.basename(f)[:-3] for f in model_files if os.path.isfile(f) and not f.endswith("__init__.py")]

# Import all modules dynamically
for module in model_modules:
    object_list = dir(importlib.import_module(f".{module}", package="models"))
    for obj in object_list:
        if obj.lower() == module:
            # from module import obj
            exec(f"from .{module} import {obj}")


#####
# import packages in subfolders
#####

dir_list = os.listdir(models_dir)

for d in dir_list:
    if not os.path.isdir(os.path.join(models_dir, d)):
        continue

    _models_dir = os.path.join(models_dir, d)
    _files = glob.glob(os.path.join(_models_dir, "*.py"))
    _modules = [os.path.basename(f)[:-3] for f in _files if os.path.isfile(f) and not f.endswith("__init__.py")]

    for module in _modules:
        object_list = dir(importlib.import_module(f".{d}.{module}", package="models"))
        for obj in object_list:
            if obj.lower() == module:
                # from module import obj
                exec(f"from .img.{module} import {obj}")
