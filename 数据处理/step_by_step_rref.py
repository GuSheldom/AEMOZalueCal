#!/usr/bin/env python3
"""
逐步展示行简化过程，清楚地看到自由变量是如何产生的
"""

import sympy as sp
from sympy import Matrix, pprint

def step_by_step_rref():
    """
    逐步展示行简化过程
    """
    print("=" * 80)
    print("逐步行简化过程 - 看清楚自由变量是如何产生的")
    print("=" * 80)
    
    # 原始矩阵
    A = Matrix([
        [0, 3, 0, 0, 0],
        [-2, 0, 1, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, -1, 0, 4],
        [0, 0, 0, 1, 0]
    ])
    
    print("原始矩阵 A:")
    pprint(A)
    print("\n对应的方程组:")
    print("0*x₁ + 3*x₂ + 0*x₃ + 0*x₄ + 0*x₅ = 0")
    print("-2*x₁ + 0*x₂ + 1*x₃ + 0*x₄ + 0*x₅ = 0")
    print("0*x₁ - 1*x₂ + 0*x₃ - 1*x₄ + 0*x₅ = 0")
    print("0*x₁ + 0*x₂ - 1*x₃ + 0*x₄ + 4*x₅ = 0")
    print("0*x₁ + 0*x₂ + 0*x₃ + 1*x₄ + 0*x₅ = 0")
    print("\n" + "="*60)
    
    # 步骤1: 交换行，让第一行有非零首元素
    print("\n步骤1: 交换第1行和第2行 (R₁ ↔ R₂)")
    A1 = A.copy()
    A1[0, :], A1[1, :] = A1[1, :], A1[0, :]
    pprint(A1)
    print("现在第1行的首元素是-2")
    
    # 步骤2: 将第一行首元素化为1
    print("\n步骤2: R₁ ← -1/2 * R₁")
    A2 = A1.copy()
    A2[0, :] = A2[0, :] * sp.Rational(-1, 2)
    pprint(A2)
    print("现在第1行: x₁ - 1/2*x₃ = 0")
    
    # 步骤3: 处理第2行，将第2列首元素化为1
    print("\n步骤3: R₂ ← 1/3 * R₂")
    A3 = A2.copy()
    A3[1, :] = A3[1, :] * sp.Rational(1, 3)
    pprint(A3)
    print("现在第2行: x₂ = 0")
    
    # 步骤4: 消除第3行的x₂项
    print("\n步骤4: R₃ ← R₃ + R₂")
    A4 = A3.copy()
    A4[2, :] = A4[2, :] + A4[1, :]
    pprint(A4)
    
    # 步骤5: 处理第4行
    print("\n步骤5: R₄ ← -R₄")
    A5 = A4.copy()
    A5[3, :] = -A5[3, :]
    pprint(A5)
    print("现在第4行: x₃ - 4*x₅ = 0")
    
    # 步骤6: 处理第5行 - 已经是标准形式
    print("\n步骤6: 第5行已经是标准形式")
    print("第5行: x₄ = 0")
    
    # 步骤7: 继续简化，消除上三角
    print("\n步骤7: 消除第1行的x₃项: R₁ ← R₁ + 1/2 * R₄")
    A6 = A5.copy()
    A6[0, :] = A6[0, :] + sp.Rational(1, 2) * A6[3, :]
    pprint(A6)
    
    print("\n最终的RREF形式:")
    final_rref, pivot_cols = A.rref()
    pprint(final_rref)
    print(f"主元列: {pivot_cols}")
    
    print("\n" + "="*60)
    print("分析最终结果:")
    print("="*60)
    
    print("\n从最终的RREF矩阵可以看出:")
    print("第1行: x₁ - 2*x₅ = 0  =>  x₁ = 2*x₅")
    print("第2行: x₂ = 0")
    print("第3行: x₃ - 4*x₅ = 0  =>  x₃ = 4*x₅")
    print("第4行: x₄ = 0")
    print("第5行: 0 = 0 (恒成立)")
    
    print("\n关键观察:")
    print("- 列1,2,3,4都有主元 (leading 1)")
    print("- 列5没有主元")
    print("- 因此 x₅ 是自由变量!")
    
    print("\n为什么x₅是自由变量?")
    print("- 在RREF中，第5列没有主元")
    print("- 这意味着x₅可以取任意值")
    print("- 一旦x₅确定，其他变量的值就被唯一确定了")
    
    print("\n通解:")
    print("设 x₅ = t (t ∈ ℝ)")
    print("则:")
    print("x₁ = 2t")
    print("x₂ = 0")
    print("x₃ = 4t")
    print("x₄ = 0")
    print("x₅ = t")
    
    print("\n向量形式:")
    print("x = t * [2, 0, 4, 0, 1]ᵀ")
    
    return final_rref, pivot_cols

def visualize_pivot_structure():
    """
    可视化主元结构
    """
    print("\n" + "="*80)
    print("主元结构可视化")
    print("="*80)
    
    print("\nRREF矩阵的结构:")
    print("     x₁  x₂  x₃  x₄  x₅")
    print("R₁ [ 1   0   0   0  -2 ]  <- 主元在第1列")
    print("R₂ [ 0   1   0   0   0 ]  <- 主元在第2列")
    print("R₃ [ 0   0   1   0  -4 ]  <- 主元在第3列")
    print("R₄ [ 0   0   0   1   0 ]  <- 主元在第4列")
    print("R₅ [ 0   0   0   0   0 ]  <- 没有主元")
    print("     ↑   ↑   ↑   ↑   ↑")
    print("     主  主  主  主  自")
    print("     元  元  元  元  由")
    print("     列  列  列  列  列")
    
    print("\n规律:")
    print("✓ 有主元的列 → 基本变量 (basic variables)")
    print("✗ 没有主元的列 → 自由变量 (free variables)")
    
    print("\n在我们的例子中:")
    print("• 基本变量: x₁, x₂, x₃, x₄")
    print("• 自由变量: x₅")
    print("• 自由变量个数 = 总变量数 - 矩阵的秩 = 5 - 4 = 1")

if __name__ == "__main__":
    final_rref, pivot_cols = step_by_step_rref()
    visualize_pivot_structure()

