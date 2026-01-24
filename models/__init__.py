import importlib
import pkgutil

from core.model import Model

# Auto-import all Model children
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if module_name in ["__init__", "base"]:
        continue

    module = importlib.import_module(f".{module_name}", package=__name__)

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isinstance(attribute, type) and issubclass(attribute, Model) and attribute is not Model:
            globals()[attribute_name] = attribute
