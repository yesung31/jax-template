import importlib
import pkgutil

from core.data import DataModule

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    if any(part.startswith(("_", "base")) for part in module_name.split(".")):
        continue

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        continue

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if (
            isinstance(attribute, type)
            and issubclass(attribute, DataModule)
            and attribute is not DataModule
        ):
            globals()[attribute_name] = attribute
