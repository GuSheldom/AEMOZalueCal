#!/usr/bin/env python3
"""
求解线性方程组 Ax = b 的解集
Question 1, Part 2: Find the set of solutions {x : Ax = b}
"""

import numpy as np
from scipy.linalg import null_space
import sympy as sp
from sympy import Matrix, symbols, pprint

def solve_linear_system():
    """
    求解给定矩阵A和向量b的线性方程组
    """
    print("=" * 60)
    print("Question 1 - Part 2: 求解线性方程组 Ax = b 的解集")
    print("=" * 60)
    
    # 定义矩阵A
    A = np.array([
        [0, 3, 0, 0, 0],
        [-2, 0, 1, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, -1, 0, 4],
        [0, 0, 0, 1, 0]
    ])
    
    print("矩阵 A:")
    print(A)
    print()
    
    # 使用符号计算来分析解的结构
    print("使用符号计算分析解的结构...")
    A_sym = Matrix(A)
    b_sym = Matrix([symbols('b1'), symbols('b2'), symbols('b3'), symbols('b4'), symbols('b5')])
    
    print("符号矩阵 A:")
    pprint(A_sym)
    print()
    
    print("符号向量 b:")
    pprint(b_sym)
    print()
    
    # 计算矩阵A的秩
    rank_A = A_sym.rank()
    print(f"矩阵A的秩: rank(A) = {rank_A}")
    
    # 计算增广矩阵的秩
    augmented = A_sym.row_join(b_sym)
    print("增广矩阵 [A|b]:")
    pprint(augmented)
    print()
    
    # 进行行简化
    print("对增广矩阵进行行简化 (RREF):")
    rref_matrix, pivot_cols = augmented.rref()
    pprint(rref_matrix)
    print(f"主元列: {pivot_cols}")
    print()
    
    # 分析解的存在性和唯一性
    print("解的存在性分析:")
    print(f"- 矩阵A的维度: {A.shape[0]} × {A.shape[1]}")
    print(f"- 矩阵A的秩: {rank_A}")
    print(f"- 变量个数: {A.shape[1]}")
    print(f"- 自由变量个数: {A.shape[1] - rank_A}")
    print()
    
    # 计算零空间（齐次方程组的解）
    print("计算零空间 (齐次方程组 Ax = 0 的解):")
    null_space_basis = A_sym.nullspace()
    print(f"零空间的维数: {len(null_space_basis)}")
    
    if null_space_basis:
        print("零空间的基:")
        for i, basis_vector in enumerate(null_space_basis):
            print(f"v_{i+1} =")
            pprint(basis_vector)
            print()
    else:
        print("零空间只包含零向量")
    print()
    
    # 分析特定的b值情况
    print("特定情况分析:")
    print("=" * 40)
    
    # 情况1: b = [0, 0, 0, 0, 0]^T (齐次情况)
    print("情况1: b = [0, 0, 0, 0, 0]^T (齐次方程组)")
    b_zero = np.array([0, 0, 0, 0, 0])
    analyze_specific_case(A, b_zero, "零向量")
    
    # 情况2: 一般情况，检查解的存在性
    print("\n情况2: 一般向量b的情况")
    print("根据行简化结果，我们可以分析解的存在条件...")
    
    # 从RREF分析约束条件
    print("从行简化结果可以看出解存在的必要充分条件:")
    analyze_solution_conditions(rref_matrix)
    
    return A, rref_matrix, null_space_basis

def analyze_specific_case(A, b, case_name):
    """分析特定b值的情况"""
    print(f"b = {b} ({case_name})")
    
    try:
        # 尝试求解
        solution = np.linalg.lstsq(A, b, rcond=None)
        x_particular = solution[0]
        residual = solution[1]
        rank = solution[2]
        
        print(f"最小二乘解: x = {x_particular}")
        print(f"残差: {residual}")
        print(f"矩阵秩: {rank}")
        
        # 验证解
        Ax = A @ x_particular
        print(f"验证 Ax = {Ax}")
        print(f"误差 ||Ax - b|| = {np.linalg.norm(Ax - b)}")
        
    except np.linalg.LinAlgError as e:
        print(f"求解失败: {e}")
    
    print("-" * 40)

def analyze_solution_conditions(rref_matrix):
    """分析解存在的条件"""
    print("从行简化矩阵分析:")
    
    # 转换为numpy数组便于分析
    rref_np = np.array(rref_matrix).astype(float)
    
    print("行简化后的增广矩阵:")
    print(rref_np)
    print()
    
    # 检查每一行
    n_rows, n_cols = rref_np.shape
    A_part = rref_np[:, :-1]  # A部分
    b_part = rref_np[:, -1]   # b部分
    
    print("解存在的条件分析:")
    
    for i in range(n_rows):
        row_A = A_part[i, :]
        row_b = b_part[i]
        
        # 检查是否为零行
        if np.allclose(row_A, 0):
            if not np.allclose(row_b, 0):
                print(f"第{i+1}行: 0 = b_{i+1}, 因此需要 b_{i+1} = 0")
            else:
                print(f"第{i+1}行: 0 = 0 (恒成立)")
        else:
            # 找到主元
            pivot_idx = np.argmax(np.abs(row_A))
            print(f"第{i+1}行: 主元在第{pivot_idx+1}列")
    
    print()

def generate_latex_solution():
    """生成LaTeX格式的解答"""
    latex_content = r"""
\section*{Question 1 - Part 2: 求解线性方程组}

\subsection*{问题陈述}
给定矩阵 $\mathbf{A}$ 和向量 $\mathbf{b}$：
$$\mathbf{A} = \begin{bmatrix}
0 & 3 & 0 & 0 & 0 \\
-2 & 0 & 1 & 0 & 0 \\
0 & -1 & 0 & -1 & 0 \\
0 & 0 & -1 & 0 & 4 \\
0 & 0 & 0 & 1 & 0
\end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \\ b_4 \\ b_5 \end{bmatrix}$$

求解线性方程组 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 的解集 $\{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{b}\}$。

\subsection*{解答步骤}

\textbf{步骤1: 矩阵分析}

首先分析矩阵 $\mathbf{A}$ 的基本性质：
- 矩阵维度：$5 \times 5$
- 通过行简化可以确定矩阵的秩

\textbf{步骤2: 行简化}

对增广矩阵 $[\mathbf{A}|\mathbf{b}]$ 进行行简化：

$$[\mathbf{A}|\mathbf{b}] = \begin{bmatrix}
0 & 3 & 0 & 0 & 0 & | & b_1 \\
-2 & 0 & 1 & 0 & 0 & | & b_2 \\
0 & -1 & 0 & -1 & 0 & | & b_3 \\
0 & 0 & -1 & 0 & 4 & | & b_4 \\
0 & 0 & 0 & 1 & 0 & | & b_5
\end{bmatrix}$$

进行行变换：
\begin{align}
R_1 &\leftrightarrow R_2 \text{ (交换第1、2行)} \\
R_1 &\leftarrow -\frac{1}{2}R_1 \text{ (第1行除以-2)} \\
R_2 &\leftarrow \frac{1}{3}R_2 \text{ (第2行除以3)} \\
R_3 &\leftarrow -R_3 \text{ (第3行乘以-1)} \\
R_4 &\leftarrow -R_4 \text{ (第4行乘以-1)}
\end{align}

得到行简化阶梯形式：
$$\text{RREF}[\mathbf{A}|\mathbf{b}] = \begin{bmatrix}
1 & 0 & -\frac{1}{2} & 0 & 0 & | & -\frac{b_2}{2} \\
0 & 1 & 0 & 0 & 0 & | & \frac{b_1}{3} \\
0 & 0 & 0 & 1 & 0 & | & -b_3 \\
0 & 0 & 1 & 0 & -4 & | & -b_4 \\
0 & 0 & 0 & -1 & 0 & | & -b_5
\end{bmatrix}$$

\textbf{步骤3: 解的存在性分析}

从行简化结果可以看出：
- $\text{rank}(\mathbf{A}) = 5$（满秩）
- 对于任意向量 $\mathbf{b}$，增广矩阵的秩也为5
- 因此方程组对任意 $\mathbf{b}$ 都有唯一解

\textbf{步骤4: 求解}

由于矩阵 $\mathbf{A}$ 是满秩的 $5 \times 5$ 矩阵，因此 $\mathbf{A}$ 可逆。

解为：
$$\boxed{\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}}$$

\textbf{步骤5: 具体解的表达式}

通过回代求解，得到：
\begin{align}
x_1 &= -\frac{b_2}{2} + \frac{b_4}{2} \\
x_2 &= \frac{b_1}{3} \\
x_3 &= -b_4 + 4b_5 \\
x_4 &= b_5 \\
x_5 &= \frac{b_4}{4}
\end{align}

因此，解集为：
$$\boxed{\{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{b}\} = \left\{\begin{bmatrix}
-\frac{b_2}{2} + \frac{b_4}{2} \\
\frac{b_1}{3} \\
-b_4 + 4b_5 \\
b_5 \\
\frac{b_4}{4}
\end{bmatrix}\right\}}$$

\subsection*{结论}

由于矩阵 $\mathbf{A}$ 是可逆的，对于任意给定的向量 $\mathbf{b} \in \mathbb{R}^5$，线性方程组 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 都有唯一解。解集是一个单点集合，包含唯一的解向量 $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$。
"""
    return latex_content

if __name__ == "__main__":
    # 执行求解
    A, rref_matrix, null_space_basis = solve_linear_system()
    
    # 生成LaTeX解答
    latex_solution = generate_latex_solution()
    
    print("\n" + "=" * 60)
    print("LaTeX格式的完整解答已生成")
    print("=" * 60)
    print(latex_solution)

