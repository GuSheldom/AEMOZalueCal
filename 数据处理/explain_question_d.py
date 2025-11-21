import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def f_prime(x):
    """f'(x) = 12(x-1)²(x-3)"""
    return 12*(x-1)**2*(x-3)

def gradient_descent_step_by_step():
    """逐步展示梯度下降过程"""
    print("=" * 80)
    print("第四题详细解释：为什么答案不只是 {1}")
    print("=" * 80)
    
    print("\n问题：找到所有会困在鞍点 x=1 的起始点")
    print("梯度下降公式：x_{n+1} = x_n - λf'(x_n)")
    print("其中 f'(x) = 12(x-1)²(x-3)")
    
    print("\n第一步：分析 x=1 附近的行为")
    print("-" * 50)
    
    # 测试非常接近1的点
    test_points = [0.999999, 1.0, 1.000001]
    learning_rate = 0.01
    
    for start in test_points:
        print(f"\n起始点: x₀ = {start}")
        x = start
        
        for i in range(5):
            grad = f_prime(x)
            x_new = x - learning_rate * grad
            
            print(f"  步骤 {i+1}: f'({x:.6f}) = {grad:.8f}")
            print(f"           x_{i+1} = {x:.6f} - {learning_rate} × {grad:.8f} = {x_new:.6f}")
            
            if abs(x_new - x) < 1e-10:
                print(f"           → 收敛！")
                break
            x = x_new
        
        print(f"  结论: 从 {start} 开始 → 收敛到 {x:.6f}")

def analyze_the_hint():
    """分析hint的含义"""
    print("\n" + "="*80)
    print("理解hint：'给出一个生成过程'")
    print("="*80)
    
    print("\nhint说：'One approach of \"characterizing\" all such starting points")
    print("is to give a process that generates them.'")
    
    print("\n这意味着什么？")
    print("1. 答案不是简单列举几个点")
    print("2. 需要一个'过程'来生成所有这样的点")
    print("3. 暗示可能有无穷多个这样的点")
    
    print("\n关键洞察：")
    print("如果我们从某个点 x₀ 开始梯度下降会到达 x=1，")
    print("那么从 x=1 开始'反向'梯度下降应该能到达 x₀")

def backward_gradient_flow():
    """展示反向梯度流的概念"""
    print("\n" + "="*80)
    print("反向梯度流：寻找所有能到达鞍点的起始点")
    print("="*80)
    
    print("\n正向梯度下降：x_{n+1} = x_n - λf'(x_n)")
    print("反向梯度下降：x_{n+1} = x_n + λf'(x_n)")
    
    print("\n从 x=1 开始反向梯度下降：")
    
    # 从1开始，但需要微小扰动来启动
    perturbations = [-1e-10, 1e-10]
    learning_rate = 0.001
    
    for pert in perturbations:
        print(f"\n从 x = 1 + {pert} 开始反向积分：")
        x = 1.0 + pert
        trajectory = [x]
        
        for i in range(20):
            grad = f_prime(x)
            # 反向：加上梯度而不是减去
            x_new = x + learning_rate * grad
            trajectory.append(x_new)
            
            if i < 5:  # 只显示前几步
                print(f"  步骤 {i+1}: x = {x:.8f}, f'(x) = {grad:.10f}")
                print(f"           x_{i+1} = {x:.8f} + {learning_rate} × {grad:.10f} = {x_new:.8f}")
            
            if abs(x_new) > 10:  # 防止发散
                break
            x = x_new
        
        print(f"  轨迹: {[round(p, 6) for p in trajectory[:10]]}...")
        print(f"  这条轨迹上的每个点都会收敛到鞍点！")

def mathematical_explanation():
    """数学解释"""
    print("\n" + "="*80)
    print("数学原理：稳定流形（Stable Manifold）")
    print("="*80)
    
    print("\n对于微分方程 dx/dt = -f'(x)：")
    print("- 鞍点的'稳定流形'是所有会收敛到鞍点的点的集合")
    print("- 这个集合通过反向积分梯度流来构造")
    
    print("\n具体步骤：")
    print("1. 从鞍点 x=1 开始")
    print("2. 求解反向方程：dx/dt = +f'(x) = +12(x-1)²(x-3)")
    print("3. 从 t=0 积分到 t=∞")
    print("4. 轨迹上的所有点构成稳定流形")
    
    print("\n为什么这样做？")
    print("- 如果点 A 经过梯度下降到达点 B")
    print("- 那么从点 B 反向梯度下降应该能回到点 A")
    print("- 所以从鞍点反向积分，能找到所有会到达鞍点的起始点")

def why_not_just_one_point():
    """解释为什么不只是{1}"""
    print("\n" + "="*80)
    print("为什么答案不只是 {1}？")
    print("="*80)
    
    print("\n我们之前的错误思路：")
    print("'只有从 x=1 开始才会停在鞍点，所以答案是 {1}'")
    
    print("\n但这忽略了一个重要问题：")
    print("hint要求'生成过程'，暗示有更多的点")
    
    print("\n正确理解：")
    print("1. 确实，在数值计算中几乎只有 x=1 会困在鞍点")
    print("2. 但从数学理论角度，存在一个'稳定流形'")
    print("3. 这个流形包含无穷多个点（虽然测度为零）")
    print("4. 题目要求的是理论上的完整刻画，不是数值结果")
    
    print("\n类比：")
    print("就像问'哪些实数的平方等于0'")
    print("- 数值上：几乎只有0")
    print("- 数学上：确实只有0")
    print("但对于鞍点的稳定流形：")
    print("- 数值上：几乎只有x=1")
    print("- 数学上：整个稳定流形（无穷多点，但测度为零）")

def final_answer():
    """最终答案"""
    print("\n" + "="*80)
    print("第四题的最终答案")
    print("="*80)
    
    print("\n生成过程（Process）：")
    print("1. 从鞍点 x = 1 开始")
    print("2. 求解微分方程：dx/dt = 12(x-1)²(x-3)")
    print("3. 初始条件：x(0) = 1")
    print("4. 积分从 t = 0 到 t = ∞")
    print("5. 收集轨迹 x(t) 上的所有点")
    
    print("\n数学表达：")
    print("S = {x(t) : t ≥ 0, dx/dt = 12(x-1)²(x-3), x(0) = 1}")
    
    print("\n这个集合的特点：")
    print("- 包含鞍点本身：1 ∈ S")
    print("- 是测度为零的集合")
    print("- 理论上有无穷多个点")
    print("- 实际计算中几乎遇不到")
    
    print("\n为什么这是正确答案：")
    print("1. 满足hint要求的'生成过程'")
    print("2. 数学上完整刻画了稳定流形")
    print("3. 解释了为什么数值上几乎只有{1}")

def main():
    gradient_descent_step_by_step()
    analyze_the_hint()
    backward_gradient_flow()
    mathematical_explanation()
    why_not_just_one_point()
    final_answer()

if __name__ == "__main__":
    main()
