"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: December, 2024
 License: FOL/LICENSE
"""
from abc import ABC, abstractmethod
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *
from .data_io import DataIO
import zarr
import dask.array as da

class ZarrIO(DataIO):
    """
    A class to handle import and export of datasets in Zarr format.

    This class provides methods for importing datasets from Zarr files and exporting
    datasets to Zarr files, with support for optional parallel processing and custom
    compression settings.
    """

    def __init__(self, io_name: str) -> None:
        """
        Initializes a ZarrIO instance.

        Parameters:
            io_name (str): A name identifier for the ZarrIO instance, typically used to describe the purpose or 
                source of the data being handled.

        Returns:
            None
        """

        super().__init__(io_name)

    @print_with_timestamp_and_execution_time  
    def Import(self,file_name:str,file_directory:str=".",parallel_lazy_import:bool=False) -> dict[str, np.ndarray]:
        """
        Imports datasets from a Zarr file.

        Parameters:
            file_name (str): The name of the Zarr file to import. If the ".zarr" extension is not provided, 
                it will be appended automatically.
            file_directory (str): The directory containing the Zarr file. Defaults to the current directory (".").
            parallel_lazy_import (bool): If True, datasets are lazily imported using Dask arrays; otherwise, 
                they are loaded as NumPy arrays. Defaults to False.

        Returns:
            dict: A dictionary containing the imported datasets. Keys are dataset names, and values are the 
                datasets themselves (as NumPy arrays or Dask arrays depending on the `parallel_lazy_import` setting).

        Raises:
            FileNotFoundError: If the specified Zarr file does not exist.
        """

        if ".zarr" not in file_name:
            file_name += ".zarr"
        file_path = os.path.join(file_directory, file_name)
        # Open the Zarr group
        store = zarr.DirectoryStore(file_path)
        root_group = zarr.open_group(store, mode='r')

        # Dictionary to store the loaded datasets
        datasets = {}

        # Iterate through datasets in the group
        for name in root_group:
            if parallel_lazy_import:
                datasets[name] = da.from_zarr(root_group[name])
            else:
                datasets[name] = np.array(root_group[name])
            fol_info(f"imported dataset '{name}' with shape {datasets[name].shape}")

        return datasets

    @print_with_timestamp_and_execution_time  
    def Export(self,data_dict:dict[str, np.ndarray],file_name:str,file_directory:str=".",zarr_export_settings:dict={}) -> None:
        """
        Exports datasets to a Zarr file.

        Parameters:
            data_dict (dict): A dictionary where keys are dataset names and values are datasets (NumPy arrays).
            file_name (str): The name of the Zarr file to create or update. If the ".zarr" extension is not provided,
                it will be appended automatically.
            file_directory (str): The directory to save the Zarr file. Defaults to the current directory (".").
            zarr_export_settings (dict): A dictionary of export settings for chunking, compression, and shuffling. 
                Default settings are:
                    - "chunk_size": (1000, 1000)
                    - "compression": "zstd"
                    - "compression_level": 5
                    - "shuffle": 2

        Returns:
            None

        Raises:
            ValueError: If any dataset in `data_dict` is not a NumPy array.
        """

        default_settings = {"chunk_size":(1000, 1000),
                            "compression":'zstd',
                            "compression_level":5,
                            "shuffle":2}
        
        zarr_export_settings = UpdateDefaultDict(default_settings,
                                                 zarr_export_settings)
        
        compressor = zarr.Blosc(cname=zarr_export_settings["compression"], 
                                clevel=zarr_export_settings["compression_level"], 
                                shuffle=zarr_export_settings["shuffle"])
        if ".zarr" not in file_name:
            file_name += ".zarr"
        file_path = os.path.join(file_directory, file_name)

        # Create or open the Zarr group
        store = zarr.DirectoryStore(file_path)
        root_group = zarr.group(store=store)

        for name, data in data_dict.items():
            # Validate data
            if not isinstance(data, np.ndarray):
                raise ValueError(f"The dataset '{name}' must be a NumPy array.")
            
            # Create a dataset within the group
            root_group.create_dataset(
                name,
                data=data,
                chunks=zarr_export_settings["chunk_size"],
                dtype=data.dtype,
                compressor=compressor,
                overwrite=True  # Overwrite if the dataset already exists
            )
            fol_info(f"exported dataset '{name}' with shape {data.shape} to Zarr file at '{file_path}'")
