"""LightGBM 稠密训练路径所需的 scipy.sparse 类型占位。"""


class spmatrix:
    pass


class csr_matrix(spmatrix):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("v1.8 的 SciPy 兼容层不支持稀疏矩阵")


class csc_matrix(spmatrix):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("v1.8 的 SciPy 兼容层不支持稀疏矩阵")


def hstack(*args, **kwargs):
    raise RuntimeError("v1.8 的 SciPy 兼容层不支持稀疏矩阵")
