import importlib
import pkgutil
from typing import Type, Dict
from .geometry import Geometry
from .hexahedra_3d_8 import Hexahedra3D8
from .tetrahedra_3d_4 import Tetrahedra3D4
from .triangle_2d_3 import Triangle2D3
from .quadrilateral_2d_4 import Quadrilateral2D4

# Dynamically import all submodules in the subpackage
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = module

# Optionally, define what should be accessible when * is used
__all__ = [name for _, name, _ in pkgutil.iter_modules(__path__)]

fe_element_dict: Dict[str, Geometry] = {"hexahedron":Hexahedra3D8("hex"),
                                        "tetra":Tetrahedra3D4("tet"),
                                        "triangle":Triangle2D3("tri"),
                                        "quad":Quadrilateral2D4("quad")}