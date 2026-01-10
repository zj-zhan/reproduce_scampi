import os
import h5py
from pathlib import Path
from typing import Union
from fastmri.data import SliceDataset

def check_file_type(root, file_ext):
    """return True if all files within root have extension of file_ext
    """
    if os.path.exists(root) and os.path.isdir(root):
        for rt, dirs, files in os.walk(root):
            for f in files:
                if Path(f).suffix != file_ext:
                    return False
    else:
        return False
    return True

class FastMRISliceH5PY(SliceDataset):
    def __init__(self,
                 root: Union[str, Path],
                 challenge: str = "multicoil",
                 use_dataset_cache: bool = False,
                 ):
        assert check_file_type(root, ".h5")
        super().__init__(root=Path(root), challenge=challenge, use_dataset_cache=use_dataset_cache)

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, index: int):
        fname, dataslice, metadata = self.raw_samples[index]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            target = hf["reconstruction_rss"][dataslice]
            attrs = dict(hf.attrs)
            attrs.update(metadata)
            max_value = attrs["max"]
        return kspace, target, max_value