"""
fft_code — FFT 滤波对角化（Filter Diagonalization Method）子包
==============================================================
将 main.py 原有功能按职责拆分为以下模块：

  params.py        — 数据类 IstParams / PhysParams
  grid.py          — 实空间网格与 k 空间动能对角元
  wavefunction.py  — 波函数归一化与随机初态生成
  hamiltonian.py   — FFT 动能算符 / 哈密顿量作用 / 滤波器作用
  filter_coeff.py  — Newton 插值滤波系数构建
  rayleigh_ritz.py — SVD + Rayleigh-Ritz 对角化
  potentials.py    — 测试势能构建（谐振子、调和阱、高斯凹坑）
                     真实势能由根目录 gaussian_potential_builder 提供
  plotting.py      — 所有绘图函数（非交互，全部存文件）
"""
