#!/usr/bin/env python3
"""
网格搜索演示脚本
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False

def demo_function(x):
    """演示函数：一个有多个峰值的函数"""
    return 100 * np.exp(-0.1 * (x - 5)**2) + 50 * np.exp(-0.2 * (x - 15)**2) + 30 * np.sin(x) + 20

def grid_search_demo():
    """演示网格搜索的工作原理"""
    print("🔍 网格搜索演示")
    print("=" * 50)
    
    # 1. 定义搜索空间
    x_min, x_max = 0, 20
    step_size = 0.5
    x_grid = np.arange(x_min, x_max + step_size, step_size)
    
    print(f"📊 搜索空间: [{x_min}, {x_max}]")
    print(f"🔢 步长: {step_size}")
    print(f"📋 总共测试点数: {len(x_grid)}")
    
    # 2. 评估每个点
    print(f"\n🧮 开始网格搜索...")
    results = []
    
    for i, x in enumerate(x_grid):
        y = demo_function(x)
        results.append((x, y))
        
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(x_grid)} (x={x:.1f}, f(x)={y:.2f})")
    
    # 3. 找到最优解
    best_x, best_y = max(results, key=lambda item: item[1])
    
    print(f"\n🎯 网格搜索结果:")
    print(f"   最优参数: x = {best_x:.1f}")
    print(f"   最优值: f(x) = {best_y:.2f}")
    
    # 4. 显示前5个最佳结果
    sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
    print(f"\n📊 前5个最佳结果:")
    for i, (x, y) in enumerate(sorted_results[:5]):
        print(f"   {i+1}. x={x:4.1f}, f(x)={y:6.2f}")
    
    # 5. 绘制结果图
    plt.figure(figsize=(12, 8))
    
    # 绘制连续函数曲线
    x_continuous = np.linspace(x_min, x_max, 1000)
    y_continuous = demo_function(x_continuous)
    plt.plot(x_continuous, y_continuous, 'b-', linewidth=2, label='真实函数 f(x)', alpha=0.7)
    
    # 绘制网格搜索点
    x_vals, y_vals = zip(*results)
    plt.scatter(x_vals, y_vals, c='red', s=30, alpha=0.6, label='网格搜索点')
    
    # 标记最优点
    plt.scatter([best_x], [best_y], c='gold', s=100, marker='*', 
                label=f'最优点 (x={best_x:.1f}, f(x)={best_y:.2f})', zorder=5)
    
    plt.xlabel('参数 x')
    plt.ylabel('函数值 f(x)')
    plt.title('网格搜索演示：寻找函数最大值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('grid_search_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 结果图已保存为: grid_search_demo.png")
    
    return best_x, best_y

def compare_with_our_z_search():
    """对比我们的Z值搜索"""
    print(f"\n🔄 对比我们的Z值搜索:")
    print("=" * 50)
    
    print("📋 我们的Z值搜索就是网格搜索的应用:")
    print("   1. 搜索空间: Z ∈ [0, 50]")
    print("   2. 步长: 0.5 (或 2.0)")
    print("   3. 目标函数: f(Z) = 在Z值约束下的最大收益")
    print("   4. 评估方法: 对每个Z值调用PuLP求解")
    print("   5. 结果: 找到收益最大的Z值")
    
    print(f"\n💡 网格搜索的特点:")
    print("   ✅ 优点:")
    print("      - 简单易懂，容易实现")
    print("      - 保证找到全局最优解（在搜索范围内）")
    print("      - 不需要函数可导或连续")
    print("      - 适合并行计算")
    
    print("   ❌ 缺点:")
    print("      - 计算量大（指数级增长）")
    print("      - 对高维问题效率低")
    print("      - 步长选择影响精度和速度")
    
    print(f"\n🎯 为什么适合我们的Z值问题:")
    print("   - Z值是一维参数（搜索空间小）")
    print("   - 收益函数可能不连续（PuLP求解结果可能跳跃）")
    print("   - 我们需要全局最优解")
    print("   - 每个Z值的PuLP求解很快")

def grid_search_vs_other_methods():
    """网格搜索 vs 其他优化方法"""
    print(f"\n🔀 网格搜索 vs 其他优化方法:")
    print("=" * 50)
    
    methods = [
        ("网格搜索", "暴力测试所有可能值", "简单可靠，保证全局最优", "计算量大"),
        ("随机搜索", "随机选择参数测试", "适合高维，计算量可控", "可能错过最优解"),
        ("梯度下降", "沿梯度方向迭代", "收敛快，适合连续函数", "可能陷入局部最优"),
        ("贝叶斯优化", "用概率模型指导搜索", "样本效率高，适合昂贵函数", "实现复杂"),
        ("遗传算法", "模拟进化过程", "适合复杂非凸问题", "参数多，调优困难")
    ]
    
    print(f"{'方法':<12} {'原理':<20} {'优点':<25} {'缺点':<15}")
    print("-" * 80)
    for method, principle, pros, cons in methods:
        print(f"{method:<12} {principle:<20} {pros:<25} {cons:<15}")

if __name__ == "__main__":
    # 运行演示
    best_x, best_y = grid_search_demo()
    compare_with_our_z_search()
    grid_search_vs_other_methods()
    
    print(f"\n🎉 演示完成！")
    print(f"💡 网格搜索找到的最优解: x={best_x:.1f}, f(x)={best_y:.2f}") 