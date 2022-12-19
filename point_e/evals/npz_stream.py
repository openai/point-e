import glob
import io
import os
import re
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class NumpyArrayInfo:
    """
    Information about an array in an npz file.
    """

    name: str
    dtype: np.dtype
    shape: Tuple[int]

    @classmethod
    def infos_from_first_file(cls, glob_path: str) -> Dict[str, "NumpyArrayInfo"]:
        paths, _ = _npz_paths_and_length(glob_path)
        return cls.infos_from_file(paths[0])

    @classmethod
    def infos_from_file(cls, npz_path: str) -> Dict[str, "NumpyArrayInfo"]:
        """
        Extract the info of every array in an npz file.
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"batch of samples was not found: {npz_path}")
        results = {}
        with open(npz_path, "rb") as f:
            with zipfile.ZipFile(f, "r") as zip_f:
                for name in zip_f.namelist():
                    if not name.endswith(".npy"):
                        continue
                    key_name = name[: -len(".npy")]
                    with zip_f.open(name, "r") as arr_f:
                        version = np.lib.format.read_magic(arr_f)
                        if version == (1, 0):
                            header = np.lib.format.read_array_header_1_0(arr_f)
                        elif version == (2, 0):
                            header = np.lib.format.read_array_header_2_0(arr_f)
                        else:
                            raise ValueError(f"unknown numpy array version: {version}")
                        shape, _, dtype = header
                        results[key_name] = cls(name=key_name, dtype=dtype, shape=shape)
        return results

    @property
    def elem_shape(self) -> Tuple[int]:
        return self.shape[1:]

    def validate(self):
        if self.name in {"R", "G", "B"}:
            if len(self.shape) != 2:
                raise ValueError(
                    f"expecting exactly 2-D shape for '{self.name}' but got: {self.shape}"
                )
        elif self.name == "arr_0":
            if len(self.shape) < 2:
                raise ValueError(f"expecting at least 2-D shape but got: {self.shape}")
            elif len(self.shape) == 3:
                # For audio, we require continuous samples.
                if not np.issubdtype(self.dtype, np.floating):
                    raise ValueError(
                        f"invalid dtype for audio batch: {self.dtype} (expected float)"
                    )
            elif self.dtype != np.uint8:
                raise ValueError(f"invalid dtype for image batch: {self.dtype} (expected uint8)")


class NpzStreamer:
    def __init__(self, glob_path: str):
        self.paths, self.trunc_length = _npz_paths_and_length(glob_path)
        self.infos = NumpyArrayInfo.infos_from_file(self.paths[0])

    def keys(self) -> List[str]:
        return list(self.infos.keys())

    def stream(self, batch_size: int, keys: Sequence[str]) -> Iterator[Dict[str, np.ndarray]]:
        cur_batch = None
        num_remaining = self.trunc_length
        for path in self.paths:
            if num_remaining is not None and num_remaining <= 0:
                break
            with open_npz_arrays(path, keys) as readers:
                combined_reader = CombinedReader(keys, readers)
                while num_remaining is None or num_remaining > 0:
                    read_bs = batch_size
                    if cur_batch is not None:
                        read_bs -= _dict_batch_size(cur_batch)
                    if num_remaining is not None:
                        read_bs = min(read_bs, num_remaining)

                    batch = combined_reader.read_batch(read_bs)
                    if batch is None:
                        break
                    if num_remaining is not None:
                        num_remaining -= _dict_batch_size(batch)
                    if cur_batch is None:
                        cur_batch = batch
                    else:
                        cur_batch = {
                            # pylint: disable=unsubscriptable-object
                            k: np.concatenate([cur_batch[k], v], axis=0)
                            for k, v in batch.items()
                        }
                    if _dict_batch_size(cur_batch) == batch_size:
                        yield cur_batch
                        cur_batch = None
        if cur_batch is not None:
            yield cur_batch


def _npz_paths_and_length(glob_path: str) -> Tuple[List[str], Optional[int]]:
    # Match slice syntax like path[:100].
    count_match = re.match("^(.*)\\[:([0-9]*)\\]$", glob_path)
    if count_match:
        raw_path = count_match[1]
        max_count = int(count_match[2])
    else:
        raw_path = glob_path
        max_count = None
    paths = sorted(glob.glob(raw_path))
    if not len(paths):
        raise ValueError(f"no paths found matching: {glob_path}")
    return paths, max_count


class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        pass


class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]:
            return None

        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs

        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)

        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str):
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]:
            return None

        res = self.arr[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return res


@contextmanager
def open_npz_arrays(path: str, arr_names: Sequence[str]) -> List[NpzArrayReader]:
    if not len(arr_names):
        yield []
        return
    arr_name = arr_names[0]
    with open_array(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        header = None
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)

        if header is None:
            reader = MemoryNpzArrayReader.load(path, arr_name)
        else:
            shape, fortran, dtype = header
            if fortran or dtype.hasobject:
                reader = MemoryNpzArrayReader.load(path, arr_name)
            else:
                reader = StreamingNpzArrayReader(arr_f, shape, dtype)

        with open_npz_arrays(path, arr_names[1:]) as next_readers:
            yield [reader] + next_readers


class CombinedReader:
    def __init__(self, keys: List[str], readers: List[NpzArrayReader]):
        self.keys = keys
        self.readers = readers

    def read_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        batches = [r.read_batch(batch_size) for r in self.readers]
        any_none = any(x is None for x in batches)
        all_none = all(x is None for x in batches)
        if any_none != all_none:
            raise RuntimeError("different keys had different numbers of elements")
        if any_none:
            return None
        if any(len(x) != len(batches[0]) for x in batches):
            raise RuntimeError("different keys had different numbers of elements")
        return dict(zip(self.keys, batches))


def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Copied from: https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/format.py#L788-L886

    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.
    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


@contextmanager
def open_array(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist():
                raise ValueError(f"missing {arr_name} in npz file")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f:
                yield arr_f


def _dict_batch_size(objs: Dict[str, np.ndarray]) -> int:
    return len(next(iter(objs.values())))
