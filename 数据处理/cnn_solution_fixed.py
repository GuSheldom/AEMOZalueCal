import numpy as np

def solve_cnn_question():
    """解答CNN相关问题"""
    print("=" * 80)
    print("Question 3: Baby Convolutional Neural Network 解答")
    print("=" * 80)
    
    print("\n题目设定：")
    print("- 3×3图像，特征为 x = (x₁, x₂, ..., x₉)")
    print("- CNN应用滤波器到每个2×2子网格")
    print("- 四个2×2子网格，每个产生一个值 u₁, u₂, u₃, u₄")
    print("- 逻辑函数：h = 1/(1 + e^(-(w₁u₁ + w₂u₂ + w₃u₃ + w₄u₄)))")
    
    # 显示滤波器参数和权重
    print("\n参数：")
    print("- 滤波器参数：θ = (θ₁, θ₂, θ₃, θ₄)")
    print("- 权重参数：w = (w₁, w₂, w₃, w₄)")
    
    # 显示u值的计算
    print("\nu值计算：")
    print("u₁ = θ₁x₁ + θ₂x₂ + θ₃x₄ + θ₄x₅")
    print("u₂ = θ₁x₂ + θ₂x₃ + θ₃x₅ + θ₄x₆")
    print("u₃ = θ₁x₄ + θ₂x₅ + θ₃x₇ + θ₄x₈")
    print("u₄ = θ₁x₅ + θ₂x₆ + θ₃x₈ + θ₄x₉")

def part_a_gradients():
    """(a) 计算梯度 ∂h/∂θⱼ 和 ∂h/∂wⱼ"""
    print("\n" + "="*60)
    print("(a) 计算梯度 ∂h/∂θⱼ 和 ∂h/∂wⱼ [6 marks]")
    print("="*60)
    
    print("\n设 z = w₁u₁ + w₂u₂ + w₃u₃ + w₄u₄")
    print("则 h = 1/(1 + e^(-z)) = σ(z) (sigmoid函数)")
    
    print("\n步骤1: 使用链式法则")
    print("∂h/∂θⱼ = (∂h/∂z) × (∂z/∂θⱼ)")
    print("∂h/∂wⱼ = (∂h/∂z) × (∂z/∂wⱼ)")
    
    print("\n步骤2: 计算 ∂h/∂z")
    print("∂h/∂z = ∂/∂z [1/(1 + e^(-z))]")
    print("      = e^(-z)/(1 + e^(-z))²")
    print("      = [1/(1 + e^(-z))] × [e^(-z)/(1 + e^(-z))]")
    print("      = h × (1 - h)")
    
    print("\n步骤3: 计算 ∂z/∂θⱼ")
    print("由于 z = w₁u₁ + w₂u₂ + w₃u₃ + w₄u₄")
    print("∂z/∂θⱼ = w₁(∂u₁/∂θⱼ) + w₂(∂u₂/∂θⱼ) + w₃(∂u₃/∂θⱼ) + w₄(∂u₄/∂θⱼ)")
    
    print("\n计算各个 ∂uᵢ/∂θⱼ：")
    print("∂u₁/∂θ₁ = x₁,  ∂u₁/∂θ₂ = x₂,  ∂u₁/∂θ₃ = x₄,  ∂u₁/∂θ₄ = x₅")
    print("∂u₂/∂θ₁ = x₂,  ∂u₂/∂θ₂ = x₃,  ∂u₂/∂θ₃ = x₅,  ∂u₂/∂θ₄ = x₆")
    print("∂u₃/∂θ₁ = x₄,  ∂u₃/∂θ₂ = x₅,  ∂u₃/∂θ₃ = x₇,  ∂u₃/∂θ₄ = x₈")
    print("∂u₄/∂θ₁ = x₅,  ∂u₄/∂θ₂ = x₆,  ∂u₄/∂θ₃ = x₈,  ∂u₄/∂θ₄ = x₉")
    
    print("\n因此：")
    print("∂z/∂θ₁ = w₁x₁ + w₂x₂ + w₃x₄ + w₄x₅")
    print("∂z/∂θ₂ = w₁x₂ + w₂x₃ + w₃x₅ + w₄x₆")
    print("∂z/∂θ₃ = w₁x₄ + w₂x₅ + w₃x₇ + w₄x₈")
    print("∂z/∂θ₄ = w₁x₅ + w₂x₆ + w₃x₈ + w₄x₉")
    
    print("\n步骤4: 计算 ∂z/∂wⱼ")
    print("∂z/∂w₁ = u₁")
    print("∂z/∂w₂ = u₂")
    print("∂z/∂w₃ = u₃")
    print("∂z/∂w₄ = u₄")
    
    print("\n最终结果：")
    print("∂h/∂θ₁ = h(1-h) × (w₁x₁ + w₂x₂ + w₃x₄ + w₄x₅)")
    print("∂h/∂θ₂ = h(1-h) × (w₁x₂ + w₂x₃ + w₃x₅ + w₄x₆)")
    print("∂h/∂θ₃ = h(1-h) × (w₁x₄ + w₂x₅ + w₃x₇ + w₄x₈)")
    print("∂h/∂θ₄ = h(1-h) × (w₁x₅ + w₂x₆ + w₃x₈ + w₄x₉)")
    print()
    print("∂h/∂w₁ = h(1-h) × u₁")
    print("∂h/∂w₂ = h(1-h) × u₂")
    print("∂h/∂w₃ = h(1-h) × u₃")
    print("∂h/∂w₄ = h(1-h) × u₄")

def part_b_likelihood():
    """(b) 推导负对数似然函数"""
    print("\n" + "="*60)
    print("(b) 推导负对数似然函数 [3 marks]")
    print("="*60)
    
    print("\n给定：")
    print("- 训练数据集：{(xᵢ, yᵢ)}ᵢ₌₁ᴺ")
    print("- xᵢ ∈ ℝ⁹，yᵢ ∈ {0,1}")
    print("- xᵢⱼ 表示第i个数据点的第j个特征")
    
    print("\n步骤1: 单个数据点的似然")
    print("对于二元分类，每个数据点的似然为：")
    print("P(yᵢ|xᵢ) = hᵢʸⁱ × (1-hᵢ)¹⁻ʸⁱ")
    print("其中 hᵢ = h(xᵢ, θ, w)")
    
    print("\n这个公式的含义：")
    print("- 当 yᵢ = 1 时：P(yᵢ|xᵢ) = hᵢ")
    print("- 当 yᵢ = 0 时：P(yᵢ|xᵢ) = 1-hᵢ")
    
    print("\n步骤2: 总似然函数")
    print("假设数据点独立，总似然为：")
    print("L(θ, w) = ∏ᵢ₌₁ᴺ P(yᵢ|xᵢ)")
    print("        = ∏ᵢ₌₁ᴺ hᵢʸⁱ × (1-hᵢ)¹⁻ʸⁱ")
    
    print("\n步骤3: 对数似然函数")
    print("取对数简化乘积：")
    print("ℓ(θ, w) = log L(θ, w)")
    print("         = ∑ᵢ₌₁ᴺ [yᵢ log hᵢ + (1-yᵢ) log(1-hᵢ)]")
    
    print("\n步骤4: 负对数似然函数")
    print("机器学习中通常最小化损失，所以取负号：")
    print()
    print("NLL(θ, w) = -ℓ(θ, w)")
    print("          = -∑ᵢ₌₁ᴺ [yᵢ log hᵢ + (1-yᵢ) log(1-hᵢ)]")
    print("          = ∑ᵢ₌₁ᴺ [-yᵢ log hᵢ - (1-yᵢ) log(1-hᵢ)]")
    
    print("\n这就是著名的二元交叉熵损失函数！")

def part_c_sgd_algorithm():
    """(c) 描述SGD算法"""
    print("\n" + "="*60)
    print("(c) SGD算法与mini-batch方法 [8 marks]")
    print("="*60)
    
    print("\n随机梯度下降(SGD)算法：")
    print("-" * 40)
    
    print("\n1. 初始化：")
    print("   - 随机初始化参数 θ⁽⁰⁾ 和 w⁽⁰⁾ (通常用小随机数)")
    print("   - 设置学习率 α > 0 (如 0.01, 0.001)")
    print("   - 设置mini-batch大小 B (如 32, 64, 128)")
    print("   - 设置最大epoch数 T")
    
    print("\n2. 对于每个epoch t = 1, 2, ..., T：")
    
    print("\n   a) 数据预处理：")
    print("      - 随机打乱训练数据 (shuffle)")
    print("      - 将N个数据点分成 ⌈N/B⌉ 个mini-batches")
    
    print("\n   b) 对于每个mini-batch ℬ = {(xᵢ, yᵢ)}ᵢ∈ℬ：")
    
    print("\n      i) 前向传播 (Forward Pass)：")
    print("         对batch中每个样本(xᵢ, yᵢ)：")
    print("         • 计算 u₁ᵢ, u₂ᵢ, u₃ᵢ, u₄ᵢ (应用滤波器)")
    print("         • 计算 zᵢ = w₁u₁ᵢ + w₂u₂ᵢ + w₃u₃ᵢ + w₄u₄ᵢ")
    print("         • 计算 hᵢ = 1/(1 + e^(-zᵢ))")
    
    print("\n      ii) 计算mini-batch损失：")
    print("          L_batch = (1/B) × ∑ᵢ∈ℬ [-yᵢ log hᵢ - (1-yᵢ) log(1-hᵢ)]")
    
    print("\n      iii) 反向传播 (Backward Pass) - 计算梯度：")
    print("           对于每个参数θⱼ：")
    print("           ∂L_batch/∂θⱼ = (1/B) × ∑ᵢ∈ℬ ∂[-yᵢ log hᵢ - (1-yᵢ) log(1-hᵢ)]/∂θⱼ")
    print()
    print("           使用链式法则：")
    print("           ∂L_batch/∂θⱼ = (1/B) × ∑ᵢ∈ℬ (hᵢ - yᵢ) × (∂hᵢ/∂θⱼ)")
    print("           其中：∂hᵢ/∂θⱼ = hᵢ(1-hᵢ) × (∂zᵢ/∂θⱼ)")
    print()
    print("           类似地：")
    print("           ∂L_batch/∂wⱼ = (1/B) × ∑ᵢ∈ℬ (hᵢ - yᵢ) × hᵢ(1-hᵢ) × uⱼᵢ")
    
    print("\n      iv) 参数更新 (Parameter Update)：")
    print("          θⱼ ← θⱼ - α × (∂L_batch/∂θⱼ)")
    print("          wⱼ ← wⱼ - α × (∂L_batch/∂wⱼ)")
    
    print("\n3. 可选：每个epoch后在验证集上评估性能")
    print("4. 重复直到收敛或达到最大epoch数")
    
    print("\n关键要点：")
    print("• Mini-batch梯度是真实梯度的无偏估计")
    print("• Batch size B 影响：")
    print("  - 小B：更随机，可能更好地逃离局部最优")
    print("  - 大B：更稳定，但可能陷入局部最优")
    print("• 学习率 α 需要调优：")
    print("  - 太大：可能不收敛或震荡")
    print("  - 太小：收敛太慢")
    print("• 通常使用学习率衰减策略")

def create_latex_solution():
    """创建LaTeX解答"""
    print("\n" + "="*60)
    print("创建LaTeX格式解答...")
    print("="*60)
    
    latex_content = r"""
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage{algorithm}
\usepackage{algorithmic}

\geometry{a4paper, margin=1in}

\title{Question 3: Baby Convolutional Neural Network Solution}
\author{}
\date{}

\begin{document}

\maketitle

\section*{(a) Compute gradients $\frac{\partial h}{\partial \theta_j}$ and $\frac{\partial h}{\partial w_j}$ [6 marks]}

Given:
\begin{align}
u_1 &= \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_4 + \theta_4 x_5 \\
u_2 &= \theta_1 x_2 + \theta_2 x_3 + \theta_3 x_5 + \theta_4 x_6 \\
u_3 &= \theta_1 x_4 + \theta_2 x_5 + \theta_3 x_7 + \theta_4 x_8 \\
u_4 &= \theta_1 x_5 + \theta_2 x_6 + \theta_3 x_8 + \theta_4 x_9 \\
h &= \frac{1}{1 + e^{-(w_1 u_1 + w_2 u_2 + w_3 u_3 + w_4 u_4)}}
\end{align}

Let $z = w_1 u_1 + w_2 u_2 + w_3 u_3 + w_4 u_4$, so $h = \frac{1}{1 + e^{-z}} = \sigma(z)$.

Using the chain rule:
\begin{align}
\frac{\partial h}{\partial \theta_j} &= \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial \theta_j} \\
\frac{\partial h}{\partial w_j} &= \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial w_j}
\end{align}

First, compute $\frac{\partial h}{\partial z}$:
\begin{align}
\frac{\partial h}{\partial z} = \frac{\partial}{\partial z}\left(\frac{1}{1 + e^{-z}}\right) = \frac{e^{-z}}{(1 + e^{-z})^2} = h(1-h)
\end{align}

Next, compute $\frac{\partial z}{\partial \theta_j}$:
\begin{align}
\frac{\partial z}{\partial \theta_1} &= w_1 x_1 + w_2 x_2 + w_3 x_4 + w_4 x_5 \\
\frac{\partial z}{\partial \theta_2} &= w_1 x_2 + w_2 x_3 + w_3 x_5 + w_4 x_6 \\
\frac{\partial z}{\partial \theta_3} &= w_1 x_4 + w_2 x_5 + w_3 x_7 + w_4 x_8 \\
\frac{\partial z}{\partial \theta_4} &= w_1 x_5 + w_2 x_6 + w_3 x_8 + w_4 x_9
\end{align}

And $\frac{\partial z}{\partial w_j}$:
\begin{align}
\frac{\partial z}{\partial w_1} = u_1, \quad \frac{\partial z}{\partial w_2} = u_2, \quad \frac{\partial z}{\partial w_3} = u_3, \quad \frac{\partial z}{\partial w_4} = u_4
\end{align}

Therefore:
\begin{align}
\frac{\partial h}{\partial \theta_1} &= h(1-h)(w_1 x_1 + w_2 x_2 + w_3 x_4 + w_4 x_5) \\
\frac{\partial h}{\partial \theta_2} &= h(1-h)(w_1 x_2 + w_2 x_3 + w_3 x_5 + w_4 x_6) \\
\frac{\partial h}{\partial \theta_3} &= h(1-h)(w_1 x_4 + w_2 x_5 + w_3 x_7 + w_4 x_8) \\
\frac{\partial h}{\partial \theta_4} &= h(1-h)(w_1 x_5 + w_2 x_6 + w_3 x_8 + w_4 x_9)
\end{align}

\begin{align}
\frac{\partial h}{\partial w_1} = h(1-h) u_1, \quad \frac{\partial h}{\partial w_2} = h(1-h) u_2, \quad \frac{\partial h}{\partial w_3} = h(1-h) u_3, \quad \frac{\partial h}{\partial w_4} = h(1-h) u_4
\end{align}

\section*{(b) Derive the negative log-likelihood function [3 marks]}

For binary classification with training dataset $\{(x_i, y_i)\}_{i=1}^N$ where $x_i \in \mathbb{R}^9$ and $y_i \in \{0,1\}$:

The likelihood for each data point is:
$$P(y_i | x_i) = h_i^{y_i} (1-h_i)^{1-y_i}$$
where $h_i = h(x_i, \theta, w)$.

The total likelihood is:
$$L(\theta, w) = \prod_{i=1}^N P(y_i | x_i) = \prod_{i=1}^N h_i^{y_i} (1-h_i)^{1-y_i}$$

The log-likelihood is:
$$\ell(\theta, w) = \sum_{i=1}^N [y_i \log h_i + (1-y_i) \log(1-h_i)]$$

Therefore, the negative log-likelihood function to minimize is:
$$\boxed{\text{NLL}(\theta, w) = -\sum_{i=1}^N [y_i \log h_i + (1-y_i) \log(1-h_i)]}$$

This is the binary cross-entropy loss function.

\section*{(c) SGD Algorithm with Mini-batch [8 marks]}

\begin{algorithm}
\caption{Stochastic Gradient Descent with Mini-batch}
\begin{algorithmic}[1]
\STATE \textbf{Initialize:} $\theta^{(0)}, w^{(0)}$ randomly, learning rate $\alpha > 0$, batch size $B$, max epochs $T$
\FOR{$t = 1$ to $T$}
    \STATE Randomly shuffle training data
    \STATE Divide data into mini-batches of size $B$
    \FOR{each mini-batch $\mathcal{B}$}
        \STATE \textbf{Forward Pass:}
        \FOR{each $(x_i, y_i) \in \mathcal{B}$}
            \STATE Compute $u_{1i}, u_{2i}, u_{3i}, u_{4i}$ using filter parameters $\theta$
            \STATE Compute $z_i = w_1 u_{1i} + w_2 u_{2i} + w_3 u_{3i} + w_4 u_{4i}$
            \STATE Compute $h_i = \frac{1}{1 + e^{-z_i}}$
        \ENDFOR
        \STATE \textbf{Compute Loss:}
        \STATE $L_{\text{batch}} = \frac{1}{B} \sum_{i \in \mathcal{B}} [-y_i \log h_i - (1-y_i) \log(1-h_i)]$
        \STATE \textbf{Backward Pass:}
        \FOR{$j = 1$ to $4$}
            \STATE $\frac{\partial L_{\text{batch}}}{\partial \theta_j} = \frac{1}{B} \sum_{i \in \mathcal{B}} (h_i - y_i) \cdot h_i(1-h_i) \cdot \frac{\partial z_i}{\partial \theta_j}$
            \STATE $\frac{\partial L_{\text{batch}}}{\partial w_j} = \frac{1}{B} \sum_{i \in \mathcal{B}} (h_i - y_i) \cdot h_i(1-h_i) \cdot u_{ji}$
        \ENDFOR
        \STATE \textbf{Parameter Update:}
        \FOR{$j = 1$ to $4$}
            \STATE $\theta_j \leftarrow \theta_j - \alpha \cdot \frac{\partial L_{\text{batch}}}{\partial \theta_j}$
            \STATE $w_j \leftarrow w_j - \alpha \cdot \frac{\partial L_{\text{batch}}}{\partial w_j}$
        \ENDFOR
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}

\textbf{Key Points:}
\begin{itemize}
\item Mini-batch gradient is an unbiased estimator of the true gradient
\item Batch size $B$ affects convergence: smaller $B$ adds more noise but may escape local minima
\item Learning rate $\alpha$ requires tuning: too large causes instability, too small causes slow convergence
\item Common practice: use learning rate scheduling (decay over time)
\item Validation set performance should be monitored to prevent overfitting
\end{itemize}

\end{document}
"""
    
    # 保存LaTeX文件
    with open('/Users/guxiuchen/Desktop/数据处理/question3_solution.tex', 'w') as f:
        f.write(latex_content)
    
    print("✅ LaTeX解答已保存为 question3_solution.tex")

def main():
    solve_cnn_question()
    part_a_gradients()
    part_b_likelihood()
    part_c_sgd_algorithm()
    create_latex_solution()
    
    print("\n" + "="*80)
    print("Question 3 解答完成！")
    print("="*80)
    
    print("\n📋 答案总结：")
    print("-" * 50)
    print("(a) 梯度计算：")
    print("    ∂h/∂θⱼ = h(1-h) × (对应的线性组合)")
    print("    ∂h/∂wⱼ = h(1-h) × uⱼ")
    print()
    print("(b) 负对数似然函数：")
    print("    NLL(θ,w) = -∑[yᵢ log hᵢ + (1-yᵢ) log(1-hᵢ)]")
    print()
    print("(c) SGD算法：")
    print("    1. 初始化参数")
    print("    2. 对每个mini-batch：前向传播→计算损失→反向传播→更新参数")
    print("    3. 重复直到收敛")

if __name__ == "__main__":
    main()
