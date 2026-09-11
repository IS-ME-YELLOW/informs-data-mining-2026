"""仅供本项目 LightGBM 稠密 DataFrame 路径使用的最小 SciPy 兼容层。

当前机器缺少 SciPy，而 LightGBM 在导入时无条件引用 scipy.sparse 的类型。
v1.8 全程不创建稀疏矩阵；真正进入稀疏路径时本模块会明确报错。
"""

from . import sparse

__version__ = "dense-only-compat"
__all__ = ["sparse"]
