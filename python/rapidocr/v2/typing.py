from typing import Literal, TypeAlias

import numpy as np

HWCImage: TypeAlias = np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.uint8]]
