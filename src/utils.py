from collections import deque
from enum import Enum

def log(filename, string):
    with open(filename, "w") as h:
        h.write(string)


class DistributionType(Enum):
    CLOSED = (1, "CLOSED")
    POINT = (2, "POINT")
    POISSON = (3, "POISSON")