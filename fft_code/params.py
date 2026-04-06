"""参数数据类。"""


class IstParams:
    """Newton 插值的离散化参数。"""
    def __init__(self, nc: int, ms: int):
        self.nc = nc   # Newton 节点数
        self.ms = ms   # 滤波中心数


class PhysParams:
    """滤波器的物理/缩放参数。"""
    def __init__(self, dE: float, Vmin: float, dt: float):
        self.dE   = dE
        self.Vmin = Vmin
        self.dt   = dt
