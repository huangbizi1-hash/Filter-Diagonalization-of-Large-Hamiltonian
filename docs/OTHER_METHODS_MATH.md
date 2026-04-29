# 代码仓库中“非滤波对角化”方法的数学公式整理

> 说明：本文**刻意不整理 Filter Diagonalization (FDM)** 的公式，只覆盖仓库里的其它方法，方便复习“每个方法在做什么”。

## 0. 统一问题与记号

代码都在求解三维定态薛定谔本征问题

\[
\hat H\psi(\mathbf r)=E\psi(\mathbf r),\qquad
\hat H=-\frac12\nabla^2+V(\mathbf r).
\]

离散后统一写作

\[
H\mathbf x=E\mathbf x
\]

或广义本征形式

\[
A\mathbf x=E\,B\mathbf x.
\]

其中 \(\mathbf x\) 是把三维波函数展平后的向量。

---

## 1. 势能重构（GaussianPotentialBuilder）

对每个原子中心 \(\mathbf R_a\)，在截断半径 \(r_\mathrm{cut}\) 内用多高斯叠加：

\[
V_a(\mathbf r)=\sum_{m=1}^{M_a} A_{a,m}\exp\!\big(-\alpha_{a,m}\|\mathbf r-\mathbf R_a\|^2\big),
\qquad \|\mathbf r-\mathbf R_a\|\le r_\mathrm{cut}.
\]

总势能为

\[
V(\mathbf r)=\sum_a V_a(\mathbf r),
\]

超过截断半径的贡献置零。这样把 \(.cube\) 中的原子坐标 + JSON 中的拟合参数变成任意分辨率网格势能。

---

## 2. 三类离散化（Numerov / Sinc-DVR / FFT-DVR）+ Chebyshev

## 2.1 Numerov-3D：广义本征值问题

代码构造的是

\[
A\mathbf x=E\,B\mathbf x,
\]

其中 \(A\) 来自三维 Numerov 九点/二十七点模板的动能+势能耦合，\(B\) 是对应质量矩阵（不是单位阵）。

连续算子意义上等价于对

\[
\left(-\frac12\nabla^2+V\right)\psi=E\psi
\]

做高阶差分离散，但离散结果是 \((A,B)\) 广义谱问题。实际求解时调用 `eigsh(..., M=B)`。

---

## 2.2 Sinc-DVR：实空间全矩阵动能

一维 Sinc-DVR 动能矩阵（代码中 `T1d`）为

\[
(T^{(1D)})_{ij}=\frac{1}{2h^2}
\begin{cases}
\pi^2/3, & i=j,\\
2(-1)^{i-j}/(i-j)^2, & i\ne j.
\end{cases}
\]

三维动能通过三个方向张量求和：

\[
T^{(3D)}=T_x\otimes I\otimes I
+I\otimes T_y\otimes I
+I\otimes I\otimes T_z.
\]

哈密顿量作用为

\[
H\mathbf x=(T^{(3D)}+\operatorname{diag}(V))\mathbf x.
\]

这对应 DVR 的“势能对角、动能稠密”结构。若显式矩阵乘，复杂度趋向高阶（代码注释里与 FFT 对比为 \(O(N^4)\) 级别 matvec 成本）。

---

## 2.3 FFT-DVR：动能在 \(k\)-空间对角化

定义离散波矢

\[
k_\ell=2\pi\,\mathrm{fftfreq}(N,d),
\]

三维动能对角元

\[
T(\mathbf k)=\frac{k_x^2+k_y^2+k_z^2}{2},
\]

并在代码中做截断

\[
T(\mathbf k)\leftarrow \min\{T(\mathbf k),\,E_\mathrm{kin\_cut}\}.
\]

对任意 \(\psi\)：

\[
T\psi=\mathcal F^{-1}\!\big[T(\mathbf k)\,\mathcal F[\psi]\big],
\qquad
H\psi=T\psi+V\psi.
\]

因此单次 matvec 主成本是 FFT，复杂度 \(O(N^3\log N)\)。

---

## 2.4 Chebyshev 配点离散

代码构造 Chebyshev 节点

\[
x_j=L\cos\!\left(\frac{\pi j}{N-1}\right),
\]

并基于配点微分矩阵 \(D\) 构造二阶导近似 \(D^2\)，三维动能采用 Kronecker 和：

\[
-\frac12\left(D_x^{(2)}\otimes I\otimes I
+I\otimes D_y^{(2)}\otimes I
+I\otimes I\otimes D_z^{(2)}\right).
\]

再加对角势能得到离散 \(H\)。

---

## 3. 特征值求解器数学原理

## 3.1 Lanczos / `eigsh`

- **最低能级模式**：求最小代数特征值（`which="SA"`）。
- **目标能量模式**：shift-invert，给 \(\sigma=E_\text{target}\)，等价放大靠近 \(\sigma\) 的谱分量。

核心思想：在 Krylov 子空间
\[
\mathcal K_m(H,\mathbf v_0)=\operatorname{span}\{\mathbf v_0,H\mathbf v_0,\dots,H^{m-1}\mathbf v_0\}
\]
上做三对角投影近似。

---

## 3.2 LOBPCG

LOBPCG 在块子空间上最小化 Rayleigh 商

\[
\mathcal R(\mathbf x)=\frac{\mathbf x^\top H\mathbf x}{\mathbf x^\top \mathbf x}
\]

（广义问题则分母是 \(\mathbf x^\top B\mathbf x\)），并结合预条件方向、历史方向做块共轭更新。代码里可给预条件器 `M`（如 Sinc-DVR 对角预条件）。

---

## 3.3 PRIMME 家族：GD / Davidson / JD / JDQMR

代码通过 `primme.eigsh` 调用不同策略（如 `PRIMME_GD`, `PRIMME_JDQMR`）。

统一框架：维护搜索子空间 \(\mathcal V\)，对投影问题做 Rayleigh-Ritz，得到 Ritz 对 \((\theta,u)\)，残差

\[
r=Hu-\theta u.
\]

然后用校正方程更新子空间。Jacobi-Davidson 的典型校正方程形式是

\[
(I-uu^\top)(H-\theta I)(I-uu^\top)t=-r,
\]

JDQMR 则用 QMR 思想近似求解该校正步骤（更适合大规模稀疏/算子形式问题）。

在目标谱附近时，PRIMME 允许直接用 `which=E_target` 拉取邻近特征值。

---

## 3.4 CGMIN（折叠谱 + 流形非线性共轭梯度）

代码里 `cg_minimize_folded` 最小化

\[
f(x)=\|(H-E_tI)x\|^2,
\qquad \|x\|=1.
\]

即“折叠谱”目标函数。令
\[
A=H-E_tI,
\]
则
\[
f(x)=x^\top A^\top A x,
\quad
\nabla f(x)=2A^\top A x.
\]

在单位球面约束下使用切空间梯度

\[
g_\mathrm{tan}=\nabla f-(x^\top \nabla f)x.
\]

搜索方向投影到切空间后，沿测地线

\[
x(\theta)=x\cos\theta+\hat p\sin\theta
\]

做一维黄金分割线搜索选 \(\theta\)。

代码的停止准则：
1. 相邻迭代 Ritz 能量变化 \(|\Delta E|<\varepsilon_E\)（主判据）；
2. 切空间梯度范数 \(\|g_\mathrm{tan}\|<\varepsilon_g\)（辅判据）。

最后报告

\[
E_\mathrm{ritz}=x^\top Hx,
\quad
\|Hx-E_\mathrm{ritz}x\|,
\quad
\|(H-E_tI)x\|.
\]

---

## 4. 多网格初猜（coarse-to-fine）

若粗网格已得特征向量 \(\psi_c\)，代码把其 reshape 为 3D 后做三线性插值到细网格：

\[
\psi_f=\mathcal I_{c\to f}[\psi_c],
\]

再归一化

\[
\psi_f\leftarrow \frac{\psi_f}{\|\psi_f\|},
\]

作为细网格迭代初值 `v0`。这通常可减少目标求解迭代步数，尤其在 JD/JDQMR 目标模式下明显。

---

## 5. 目标模式与残差判据（JDQMR 扫描脚本）

扫描脚本按多个 \(E_\text{target}\) 调用

\[
\texttt{primme.eigsh}(H,\,\texttt{which}=E_\text{target},\,\texttt{method}=\texttt{PRIMME\_JDQMR}).
\]

并记录每个收敛本征对残差范数

\[
\|r_i\|=\|H u_i-\lambda_i u_i\|.
\]

筛选时常看

\[
\min_i |\lambda_i-E_\text{target}|,
\]

来判定“是否命中目标谱窗”。

---

## 6. 一页速记：各方法“在干什么”

- **Numerov-3D**：高阶差分，把 PDE 变成广义本征值问题 \(A x=E B x\)。
- **Sinc-DVR**：动能用全矩阵 sinc 核，势能是对角项。
- **FFT-DVR**：动能在频域乘法（对角），回到实空间再加势能。
- **Chebyshev**：在 Chebyshev 节点上做谱配点微分。
- **Lanczos**：Krylov 子空间三对角化，求极端/目标附近本征值。
- **LOBPCG**：块预条件共轭梯度，直接压低多个 Rayleigh 商。
- **PRIMME-JDQMR/JD/GD**：子空间 + 残差校正，面向大规模本征问题。
- **CGMIN 折叠谱**：最小化 \(\|(H-E_tI)x\|^2\) 在单位球面上的非线性优化。
- **多网格初猜**：粗网格本征向量插值到细网格，提高收敛效率。

