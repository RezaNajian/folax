import importlib
import pkgutil

# Dynamically import all submodules in the subpackage
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = module

# Optionally, define what should be accessible when * is used
__all__ = [name for _, name, _ in pkgutil.iter_modules(__path__)]
