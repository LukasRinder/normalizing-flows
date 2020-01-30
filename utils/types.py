from enum import Enum

class FlowType(Enum):
    planar = 1
    realnvp = 2
    neuralspline = 3
    maf = 4
    

class DataType(Enum):
    toydata = 1
    uci = 2
    mnist = 3
    celeb = 4