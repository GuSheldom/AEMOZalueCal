def explain_sgd_concepts():
    """解释SGD中的关键概念"""
    print("=" * 80)
    print("SGD算法关键概念解释")
    print("=" * 80)
    
    print("\n🔍 什么是Epoch？")
    print("-" * 40)
    print("Epoch = 完整遍历一次整个训练数据集")
    print()
    print("举例说明：")
    print("- 假设有1000个训练样本")
    print("- Batch size = 32 (题目给定)")
    print("- 那么一个epoch需要：1000 ÷ 32 = 32个batches (向上取整)")
    print("- 每个epoch后，所有1000个样本都被用来训练过一次")
    
    print("\n📊 Epoch vs Iteration vs Batch：")
    print("-" * 40)
    print("• Batch: 一次前向+反向传播处理的样本数 (32个)")
    print("• Iteration: 处理一个batch = 1次迭代")
    print("• Epoch: 处理完所有训练数据 = 多次迭代")
    print()
    print("关系：1 Epoch = ⌈总样本数/Batch大小⌉ 次 Iterations")
    
    print("\n🎯 为什么需要多个Epochs？")
    print("-" * 40)
    print("1. 一次遍历通常不足以学习复杂模式")
    print("2. 需要多次看到相同数据才能收敛")
    print("3. 每个epoch后可以评估模型性能")
    print("4. 可以调整学习率或其他超参数")
    
    print("\n⚙️ Max Epochs的作用：")
    print("-" * 40)
    print("• 防止无限循环训练")
    print("• 控制训练时间")
    print("• 通常结合早停(Early Stopping)使用")
    print("• 典型值：10-1000，取决于数据集大小和复杂度")

def detailed_sgd_with_32_batch():
    """详细的SGD算法，使用32个样本的batch"""
    print("\n" + "="*80)
    print("修正的SGD算法 (Batch Size = 32)")
    print("="*80)
    
    print("\n算法步骤：")
    print("-" * 50)
    
    print("1. 初始化：")
    print("   • 随机初始化 θ⁽⁰⁾ = (θ₁, θ₂, θ₃, θ₄)")
    print("   • 随机初始化 w⁽⁰⁾ = (w₁, w₂, w₃, w₄)")
    print("   • 设置学习率 α (如 0.01)")
    print("   • 设置batch大小 B = 32 (题目给定)")
    print("   • 设置最大epoch数 T (如 100)")
    
    print("\n2. 训练循环：")
    print("   For epoch = 1, 2, ..., T:")
    
    print("\n   a) 数据准备：")
    print("      • 随机打乱所有N个训练样本")
    print("      • 计算需要的batch数：num_batches = ⌈N/32⌉")
    print("      • 将数据分成num_batches个批次")
    print("      • 最后一个batch可能少于32个样本")
    
    print("\n   b) 批次处理：")
    print("      For batch_idx = 1, 2, ..., num_batches:")
    print("         获取当前batch: {(x₁, y₁), (x₂, y₂), ..., (x₃₂, y₃₂)}")
    print("         (最后一批可能少于32个)")
    
    print("\n         i) 前向传播：")
    print("            For i = 1 to batch_size:")
    print("               计算 u₁ᵢ, u₂ᵢ, u₃ᵢ, u₄ᵢ")
    print("               计算 zᵢ = w₁u₁ᵢ + w₂u₂ᵢ + w₃u₃ᵢ + w₄u₄ᵢ")
    print("               计算 hᵢ = 1/(1 + e^(-zᵢ))")
    
    print("\n         ii) 计算batch损失：")
    print("             L = (1/batch_size) × Σᵢ[-yᵢlog(hᵢ) - (1-yᵢ)log(1-hᵢ)]")
    
    print("\n         iii) 计算梯度：")
    print("              For j = 1 to 4:")
    print("                 ∂L/∂θⱼ = (1/batch_size) × Σᵢ(hᵢ-yᵢ)×hᵢ(1-hᵢ)×(∂zᵢ/∂θⱼ)")
    print("                 ∂L/∂wⱼ = (1/batch_size) × Σᵢ(hᵢ-yᵢ)×hᵢ(1-hᵢ)×uⱼᵢ")
    
    print("\n         iv) 更新参数：")
    print("             For j = 1 to 4:")
    print("                θⱼ ← θⱼ - α × (∂L/∂θⱼ)")
    print("                wⱼ ← wⱼ - α × (∂L/∂wⱼ)")
    
    print("\n   c) Epoch结束：")
    print("      • 可选：在验证集上评估性能")
    print("      • 可选：调整学习率")
    print("      • 检查是否收敛或达到最大epoch数")
    
    print("\n3. 训练结束条件：")
    print("   • 达到最大epoch数 T")
    print("   • 或者损失函数收敛 (变化很小)")
    print("   • 或者验证集性能不再提升 (早停)")

def concrete_example():
    """具体例子"""
    print("\n" + "="*80)
    print("具体例子：假设有1000个训练样本")
    print("="*80)
    
    print("\n设置：")
    print("• 总样本数 N = 1000")
    print("• Batch大小 B = 32")
    print("• 最大epochs T = 50")
    
    print("\n计算：")
    print("• 每个epoch的batch数 = ⌈1000/32⌉ = 32个batches")
    print("• 前31个batches：每个32个样本")
    print("• 第32个batch：8个样本 (1000 - 31×32 = 8)")
    
    print("\n训练过程：")
    print("Epoch 1:")
    print("  Batch 1: 样本 1-32   → 前向传播 → 反向传播 → 更新参数")
    print("  Batch 2: 样本 33-64  → 前向传播 → 反向传播 → 更新参数")
    print("  ...")
    print("  Batch 32: 样本 993-1000 → 前向传播 → 反向传播 → 更新参数")
    print("  → Epoch 1 完成，所有1000个样本都用过一次")
    
    print("\nEpoch 2:")
    print("  重新打乱数据，重复上述过程...")
    
    print("\n...")
    print("Epoch 50: 训练结束")
    
    print("\n总计算量：")
    print("• 总iterations = 50 epochs × 32 batches/epoch = 1600次参数更新")
    print("• 每个样本被使用了50次")

def why_batch_size_matters():
    """为什么batch大小很重要"""
    print("\n" + "="*80)
    print("为什么Batch Size = 32很重要？")
    print("="*80)
    
    print("\n🔄 不同batch大小的影响：")
    print("-" * 40)
    
    print("Batch Size = 1 (随机梯度下降)：")
    print("  ✅ 优点：更随机，容易逃离局部最优")
    print("  ❌ 缺点：梯度噪声大，收敛不稳定")
    print("  ❌ 缺点：无法利用向量化计算")
    
    print("\nBatch Size = 32 (Mini-batch)：")
    print("  ✅ 优点：平衡了随机性和稳定性")
    print("  ✅ 优点：可以向量化计算，效率高")
    print("  ✅ 优点：梯度估计相对准确")
    print("  ✅ 优点：内存使用合理")
    
    print("\nBatch Size = 全部数据 (批量梯度下降)：")
    print("  ✅ 优点：梯度计算最准确")
    print("  ❌ 缺点：容易陷入局部最优")
    print("  ❌ 缺点：内存需求大")
    print("  ❌ 缺点：收敛慢")
    
    print("\n🎯 为什么选择32？")
    print("-" * 40)
    print("• 2的幂次，对GPU友好")
    print("• 足够大：梯度估计相对稳定")
    print("• 足够小：保持一定随机性")
    print("• 内存友好：不会占用太多显存")
    print("• 经验证明：在很多任务上效果好")

def main():
    explain_sgd_concepts()
    detailed_sgd_with_32_batch()
    concrete_example()
    why_batch_size_matters()
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("• Epoch = 完整遍历一次所有训练数据")
    print("• Max Epochs = 最大训练轮数，防止过拟合")
    print("• Batch Size = 32 是题目给定的，每次处理32个样本")
    print("• 一个epoch包含多个batches")
    print("• 训练过程：多个epochs，每个epoch多个batches")

if __name__ == "__main__":
    main()
