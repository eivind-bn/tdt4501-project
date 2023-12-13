from typing import *

import torch

DataType = Literal[
    "float32",
    "float",
    "float64",
    "double",
    "float16",
    "bfloat16",
    "half",
    "uint8",
    "int8",
    "int16",
    "short",
    "int32",
    "int",
    "int64",
    "long",
    "complex32",
    "complex64",
    "cfloat",
    "complex128",
    "cdouble",
    "quint8",
    "qint8",
    "qint32",
    "bool",
    "quint4x2",
    "quint2x4",
    ]

Device = Literal["cpu", "cuda:0", "cuda:1"]

def torch_dtype(name: DataType) -> torch.dtype:
    dtype = getattr(torch, name)
    assert isinstance(dtype, torch.dtype)
    return dtype