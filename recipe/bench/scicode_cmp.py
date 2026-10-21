"""SciCode's test helpers, copied into the grading sandbox as `scicode/compare/cmp.py`: the comparisons its test cases
import from `scicode.compare.cmp` and the target loader its harness imports from `scicode.parse.parse`, as in
github.com/scicode-bench/SciCode, src/scicode/. Only the sandbox imports this module."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import scipy.sparse
import sympy


def are_dicts_close(dict1: dict[Any, Any], dict2: dict[Any, Any], atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    dict1 = process_symbol_in_dict(dict1)
    dict2 = process_symbol_in_dict(dict2)
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]
        if isinstance(value1, sympy.Symbol | str):
            if value1 != value2:
                return False
        elif isinstance(
            value1,
            scipy.sparse.csr_matrix | scipy.sparse.csc_matrix | scipy.sparse.bsr_matrix | scipy.sparse.coo_matrix,
        ):
            if not np.allclose(value1.toarray(), value2.toarray(), atol=atol, rtol=rtol):
                return False
        else:
            try:
                if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                    return False
            except ValueError:
                if value1 != value2:
                    return False
    return True


def process_symbol_in_dict(dict: dict[Any, Any]) -> dict[Any, Any]:
    new_dict = {}
    for key, value in dict.items():
        new_dict[key] = value
        if isinstance(value, sympy.Symbol):
            new_dict[key] = str(value)
        if isinstance(key, sympy.Symbol):
            new_dict[str(key)] = dict[key]
            new_dict.pop(key)
    return new_dict


def are_csc_matrix_close(matrix1: Any, matrix2: Any) -> bool:
    return bool(np.allclose(matrix1.toarray(), matrix2.toarray()))


def cmp_tuple_or_list(var1: Any, var2: Any) -> bool:
    if len(var1) != len(var2):
        return False
    for v1, v2 in zip(var1, var2, strict=False):
        if isinstance(v1, dict):
            if not are_dicts_close(v1, v2):
                return False
        elif isinstance(v1, scipy.sparse.csr_matrix | scipy.sparse.csc_matrix):
            if not are_csc_matrix_close(v1, v2):
                return False
        elif isinstance(v1, bool):
            if v1 != v2:
                return False
        else:
            try:
                if not np.allclose(v1, v2):
                    return False
            except ValueError as e:
                print(e)
                if v1 != v2:
                    return False
    return True


def process_hdf5_list(group: Any) -> list[Any]:
    return [group[key][()] for key in group]


def process_hdf5_dict(group: Any) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            out[key] = process_hdf5_sparse_matrix(obj["sparse_matrix"])
        elif isinstance(obj[()], bytes):
            out[key] = obj[()].decode("utf-8", errors="strict")
        else:
            try:
                out[float(key)] = obj[()]
            except ValueError:
                out[key] = obj[()]
    return out


def process_hdf5_sparse_matrix(group: Any) -> Any:
    data = group["data"][()]
    shape = tuple(group["shape"][()])
    if "row" in group and "col" in group:
        return scipy.sparse.coo_matrix((data, (group["row"][()], group["col"][()])), shape=shape)
    if "blocksize" in group:
        blocksize = tuple(group["blocksize"][()])
        return scipy.sparse.bsr_matrix(
            (data, group["indices"][()], group["indptr"][()]), shape=shape, blocksize=blocksize
        )
    return scipy.sparse.csr_matrix((data, group["indices"][()], group["indptr"][()]), shape=shape)


def process_hdf5_datagroup(group: Any) -> Any:
    for key in group:
        if key == "list":
            return process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return process_hdf5_sparse_matrix(group[key])
        return process_hdf5_dict(group)
    return None


def process_hdf5_value(obj: Any) -> Any:
    if isinstance(obj, h5py.Dataset):
        value = obj[()]
        return value.decode("utf-8", errors="strict") if isinstance(value, bytes) else value
    return process_hdf5_datagroup(obj)


def process_hdf5_to_tuple(step_id: str, test_num: int, h5py_file: str) -> list[Any]:
    data_lst: list[Any] = []
    with h5py.File(h5py_file, "r") as f:
        for test_id in range(test_num):
            group_path = f"{step_id}/test{test_id + 1}"
            if not isinstance(f[group_path], h5py.Group):
                raise FileNotFoundError(f"Path {group_path} not found in the file.")
            group = f[group_path]
            keys = list(group.keys())
            if len(keys) == 1:
                data_lst.append(process_hdf5_value(group[keys[0]]))
            else:
                data_lst.append(tuple(process_hdf5_value(group[key]) for key in keys))
    return data_lst
