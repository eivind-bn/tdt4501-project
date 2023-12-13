from typing import *
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from torch import FloatTensor, from_numpy

import numpy as np

from ..util import literals

if TYPE_CHECKING:
    from ..util import Device, DataType

T = TypeVar("T", bound=np.generic)

class Observation(Generic[T], ABC): 
    
    @abstractmethod
    def numpy(self) -> NDArray[T]:
        pass

    def tensor(self, 
               device: "Device", 
               dtype: "DataType",
               flatten: bool,
               use_grad: bool = False) -> FloatTensor:
        frame = self.numpy()
        h,w,c = frame.shape
        return cast(FloatTensor, (from_numpy(frame)\
            .movedim(2,0)\
            .reshape((-1,) if flatten else (1,c,h,w))\
            .type(literals.torch_dtype(dtype))\
            .to(device) / 255) \
            .requires_grad_(use_grad))
    