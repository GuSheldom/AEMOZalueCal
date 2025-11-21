import numpy as np
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def solve_question_2():
    """解答Question 2: Rock n Roll Die"""
    print("=" * 80)
    print("Question 2: Rock n Roll Die 解答")
    print("=" * 80)
    
    print("\n题目设定：")
    print("- 四面骰子，每面出现概率相等")
    print("- 两面值为2，一面值为3，一面值为4")
    print("- 每面出现概率 = 1/4")
    
    # (a) 单次投掷概率
    print("\n" + "="*60)
    print("(a) 单次投掷各面值概率 [1 mark]")
    print("="*60)
    
    print("\n计算过程：")
    print("P(面值=2) = 2/4 = 1/2 = 0.5")
    print("P(面值=3) = 1/4 = 0.25") 
    print("P(面值=4) = 1/4 = 0.25")
    
    print("\n验证：0.5 + 0.25 + 0.25 = 1.0 ✓")
    
    # (b) 两次独立投掷，和为6的概率
    print("\n" + "="*60)
    print("(b) 两次独立投掷，和为6的概率 [3 marks]")
    print("="*60)
    
    print("\n所有可能的组合：")
    outcomes = []
    probs = []
    
    # 所有可能的面值和对应概率
    values = [2, 2, 3, 4]  # 骰子的四个面
    prob_each_face = 1/4
    
    # 计算所有可能的两次投掷结果
    for i, val1 in enumerate(values):
        for j, val2 in enumerate(values):
            outcomes.append((val1, val2))
            probs.append(prob_each_face * prob_each_face)
    
    # 找出和为6的情况
    sum_6_cases = [(val1, val2) for val1, val2 in outcomes if val1 + val2 == 6]
    
    print("和为6的情况：")
    prob_sum_6 = 0
    for case in sum_6_cases:
        val1, val2 = case
        # 计算这种情况的概率
        prob1 = 0.5 if val1 == 2 else 0.25
        prob2 = 0.5 if val2 == 2 else 0.25
        case_prob = prob1 * prob2
        prob_sum_6 += case_prob
        print(f"  ({val1}, {val2}): P = {prob1} × {prob2} = {case_prob}")
    
    print(f"\nP(和=6) = {prob_sum_6}")
    
    # (c) 两次独立投掷乘积的期望和方差
    print("\n" + "="*60)
    print("(c) 乘积X的期望E[X]和方差Var[X] [5 marks]")
    print("="*60)
    
    # 首先计算单次投掷的期望和方差
    print("\n步骤1：计算单次投掷的期望和方差")
    values_unique = [2, 3, 4]
    probs_unique = [0.5, 0.25, 0.25]
    
    # E[Y] where Y is single roll
    E_Y = sum(val * prob for val, prob in zip(values_unique, probs_unique))
    print(f"E[Y] = 2×0.5 + 3×0.25 + 4×0.25 = {E_Y}")
    
    # E[Y²]
    E_Y2 = sum(val**2 * prob for val, prob in zip(values_unique, probs_unique))
    print(f"E[Y²] = 4×0.5 + 9×0.25 + 16×0.25 = {E_Y2}")
    
    # Var[Y]
    Var_Y = E_Y2 - E_Y**2
    print(f"Var[Y] = E[Y²] - (E[Y])² = {E_Y2} - {E_Y}² = {Var_Y}")
    
    print("\n步骤2：计算乘积X = Y₁ × Y₂的期望和方差")
    print("由于Y₁和Y₂独立：")
    
    # E[X] = E[Y₁ × Y₂] = E[Y₁] × E[Y₂]
    E_X = E_Y * E_Y
    print(f"E[X] = E[Y₁] × E[Y₂] = {E_Y} × {E_Y} = {E_X}")
    
    # Var[X] = Var[Y₁ × Y₂] = E[Y₁]²Var[Y₂] + E[Y₂]²Var[Y₁] + Var[Y₁]Var[Y₂]
    Var_X = E_Y**2 * Var_Y + E_Y**2 * Var_Y + Var_Y * Var_Y
    print(f"Var[X] = E[Y₁]²Var[Y₂] + E[Y₂]²Var[Y₁] + Var[Y₁]Var[Y₂]")
    print(f"       = {E_Y}²×{Var_Y} + {E_Y}²×{Var_Y} + {Var_Y}×{Var_Y}")
    print(f"       = {E_Y**2 * Var_Y} + {E_Y**2 * Var_Y} + {Var_Y * Var_Y}")
    print(f"       = {Var_X}")
    
    # (d) 三次独立投掷，三个面值都不同的概率
    print("\n" + "="*60)
    print("(d) 三次独立投掷，三个面值都不同的概率 [3 marks]")
    print("="*60)
    
    print("\n三个面值都不同意味着：一个2，一个3，一个4")
    print("可能的排列：(2,3,4), (2,4,3), (3,2,4), (3,4,2), (4,2,3), (4,3,2)")
    print("共6种排列")
    
    # 每种排列的概率
    prob_each_arrangement = 0.5 * 0.25 * 0.25  # P(2) × P(3) × P(4)
    prob_all_different = 6 * prob_each_arrangement
    
    print(f"\n每种排列的概率 = 0.5 × 0.25 × 0.25 = {prob_each_arrangement}")
    print(f"总概率 = 6 × {prob_each_arrangement} = {prob_all_different}")
    
    # (e) 游戏期望奖励
    print("\n" + "="*60)
    print("(e) 游戏期望奖励E[R] [4 marks]")
    print("="*60)
    
    print("\n游戏规则：")
    print("- 持续投掷直到连续两次相同面值")
    print("- 奖励 = 停止前所有面值的和")
    
    print("\n使用hint中的方法：")
    print("设Rⱼ为从现在开始的期望奖励，假设上一次投掷结果为j")
    
    print("\n状态转移分析：")
    print("- 如果当前投掷结果与上次相同 → 游戏结束")
    print("- 如果当前投掷结果与上次不同 → 继续游戏")
    
    # 设置方程组
    print("\n设R₂, R₃, R₄分别为上次投掷2,3,4时的期望奖励")
    
    print("\nR₂的计算：")
    print("P(投出2) = 0.5 → 游戏结束，奖励 = 2")
    print("P(投出3) = 0.25 → 继续，当前奖励3 + 期望奖励R₃")
    print("P(投出4) = 0.25 → 继续，当前奖励4 + 期望奖励R₄")
    print("R₂ = 0.5×2 + 0.25×(3+R₃) + 0.25×(4+R₄)")
    
    print("\n类似地：")
    print("R₃ = 0.5×(2+R₂) + 0.25×3 + 0.25×(4+R₄)")
    print("R₄ = 0.5×(2+R₂) + 0.25×(3+R₃) + 0.25×4")
    
    # 求解方程组
    print("\n求解方程组：")
    # R₂ = 1 + 0.25R₃ + 0.25R₄ + 0.75
    # R₃ = 1 + 0.5R₂ + 0.75 + 0.25R₄
    # R₄ = 1 + 0.5R₂ + 0.25R₃ + 1
    
    # 简化：
    # R₂ = 1.75 + 0.25R₃ + 0.25R₄
    # R₃ = 1.75 + 0.5R₂ + 0.25R₄  
    # R₄ = 2 + 0.5R₂ + 0.25R₃
    
    # 使用矩阵求解
    A = np.array([
        [1, -0.25, -0.25],
        [-0.5, 1, -0.25],
        [-0.5, -0.25, 1]
    ])
    b = np.array([1.75, 1.75, 2])
    
    R_values = np.linalg.solve(A, b)
    R2, R3, R4 = R_values
    
    print(f"R₂ = {R2:.4f}")
    print(f"R₃ = {R3:.4f}")  
    print(f"R₄ = {R4:.4f}")
    
    # 游戏开始时的期望奖励
    E_R = 0.5 * (2 + R2) + 0.25 * (3 + R3) + 0.25 * (4 + R4)
    print(f"\n游戏开始时的期望奖励：")
    print(f"E[R] = 0.5×(2+R₂) + 0.25×(3+R₃) + 0.25×(4+R₄)")
    print(f"     = 0.5×(2+{R2:.4f}) + 0.25×(3+{R3:.4f}) + 0.25×(4+{R4:.4f})")
    print(f"     = {E_R:.4f}")
    
    return {
        'prob_2': 0.5,
        'prob_3': 0.25,
        'prob_4': 0.25,
        'prob_sum_6': prob_sum_6,
        'E_X': E_X,
        'Var_X': Var_X,
        'prob_all_different': prob_all_different,
        'E_R': E_R
    }

def create_summary():
    """创建答案总结"""
    print("\n" + "="*80)
    print("答案总结")
    print("="*80)
    
    results = solve_question_2()
    
    print(f"\n(a) P(面值=2) = {results['prob_2']}, P(面值=3) = {results['prob_3']}, P(面值=4) = {results['prob_4']}")
    print(f"\n(b) P(两次投掷和=6) = {results['prob_sum_6']}")
    print(f"\n(c) E[X] = {results['E_X']}, Var[X] = {results['Var_X']:.4f}")
    print(f"\n(d) P(三次投掷都不同) = {results['prob_all_different']}")
    print(f"\n(e) E[R] = {results['E_R']:.4f}")

if __name__ == "__main__":
    solve_question_2()
    create_summary()
