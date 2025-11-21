#!/usr/bin/env python3
"""
详细解释自由变量的概念和计算过程
"""

import numpy as np
import sympy as sp
from sympy import Matrix, symbols, pprint

def explain_free_variables():
    """
    详细解释自由变量的概念和如何确定自由变量
    """
    print("=" * 80)
    print("自由变量的详细解释")
    print("=" * 80)
    
    # 定义原始矩阵A
    A = Matrix([
        [0, 3, 0, 0, 0],
        [-2, 0, 1, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, -1, 0, 4],
        [0, 0, 0, 1, 0]
    ])
    
    print("原始矩阵 A:")
    pprint(A)
    print()
    
    # 步骤1: 计算行简化阶梯形式 (RREF)
    print("步骤1: 计算行简化阶梯形式 (RREF)")
    print("-" * 50)
    
    rref_A, pivot_cols = A.rref()
    print("RREF(A):")
    pprint(rref_A)
    print(f"主元列 (pivot columns): {pivot_cols}")
    print()
    
    # 步骤2: 理解主元列和非主元列
    print("步骤2: 理解主元列和非主元列")
    print("-" * 50)
    
    n_vars = A.cols  # 变量个数
    all_cols = set(range(n_vars))
    pivot_cols_set = set(pivot_cols)
    free_cols = all_cols - pivot_cols_set
    
    print(f"总变量个数: {n_vars}")
    print(f"主元列 (基本变量): {sorted(pivot_cols_set)} -> x_{[i+1 for i in sorted(pivot_cols_set)]}")
    print(f"非主元列 (自由变量): {sorted(free_cols)} -> x_{[i+1 for i in sorted(free_cols)]}")
    print()
    
    # 步骤3: 从RREF分析变量关系
    print("步骤3: 从RREF分析变量关系")
    print("-" * 50)
    
    print("从RREF矩阵可以看出:")
    for i, row in enumerate(rref_A.tolist()):
        if i < len(pivot_cols):
            pivot_col = pivot_cols[i]
            print(f"第{i+1}行: x_{pivot_col+1} = ", end="")
            
            # 找到这一行中非零的非主元列系数
            terms = []
            for j in range(n_vars):
                if j != pivot_col and row[j] != 0:
                    coeff = -row[j]  # 移项后的系数
                    if coeff == 1:
                        terms.append(f"x_{j+1}")
                    elif coeff == -1:
                        terms.append(f"-x_{j+1}")
                    else:
                        terms.append(f"{coeff}x_{j+1}")
            
            if terms:
                print(" + ".join(terms))
            else:
                print("0")
    print()
    
    # 步骤4: 计算零空间
    print("步骤4: 计算零空间 (齐次方程组 Ax = 0 的解)")
    print("-" * 50)
    
    null_space = A.nullspace()
    print(f"零空间维数: {len(null_space)}")
    print(f"自由变量个数: {len(free_cols)}")
    print(f"验证: 零空间维数 = 自由变量个数? {len(null_space) == len(free_cols)}")
    print()
    
    if null_space:
        print("零空间的基向量:")
        for i, basis in enumerate(null_space):
            print(f"v_{i+1} =")
            pprint(basis)
            print()
    
    # 步骤5: 手工验证自由变量的计算
    print("步骤5: 手工验证自由变量的计算")
    print("-" * 50)
    
    print("让我们手工验证一下:")
    print("设 x_5 = t (自由变量)")
    print()
    print("从RREF矩阵的方程组:")
    print("x_1 - 2x_5 = 0  =>  x_1 = 2x_5 = 2t")
    print("x_2 = 0")
    print("x_3 - 4x_5 = 0  =>  x_3 = 4x_5 = 4t") 
    print("x_4 = 0")
    print("x_5 = t (自由变量)")
    print()
    print("因此通解为:")
    print("x = t * [2, 0, 4, 0, 1]^T")
    print()
    
    # 步骤6: 验证我们的解
    print("步骤6: 验证解")
    print("-" * 50)
    
    # 用我们找到的零空间基向量验证
    if null_space:
        test_vector = null_space[0]
        result = A * test_vector
        print("验证 A * v_1 = 0:")
        print("A * v_1 =")
        pprint(result)
        print(f"是否为零向量? {result.equals(Matrix.zeros(5, 1))}")
    
    return pivot_cols, free_cols, null_space

def explain_theory():
    """
    解释自由变量的理论背景
    """
    print("\n" + "=" * 80)
    print("自由变量的理论背景")
    print("=" * 80)
    
    theory_text = """
    自由变量的概念来自线性代数中的基本理论:
    
    1. **定义**: 
       - 在线性方程组的解中，可以任意取值的变量称为自由变量
       - 其他变量的值由自由变量唯一确定
    
    2. **如何确定自由变量**:
       - 将系数矩阵A化为行简化阶梯形式 (RREF)
       - 主元列对应的变量是基本变量 (basic variables)
       - 非主元列对应的变量是自由变量 (free variables)
    
    3. **自由变量的个数**:
       - 自由变量个数 = n - rank(A)
       - 其中 n 是变量总数，rank(A) 是矩阵A的秩
    
    4. **几何意义**:
       - 每个自由变量对应解空间的一个维度
       - 如果有k个自由变量，解空间是k维的
    
    5. **在我们的例子中**:
       - 矩阵A是5×5，rank(A) = 4
       - 自由变量个数 = 5 - 4 = 1
       - 解空间是1维的（一条直线）
    """
    
    print(theory_text)

if __name__ == "__main__":
    # 运行详细解释
    pivot_cols, free_cols, null_space = explain_free_variables()
    
    # 解释理论背景
    explain_theory()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"在这个问题中:")
    print(f"- 主元列: {sorted(pivot_cols)} (对应基本变量)")
    print(f"- 自由变量列: {sorted(free_cols)} (对应自由变量)")
    print(f"- 自由变量: x_{list(free_cols)[0]+1}")
    print(f"- 解空间维数: {len(null_space)}")

