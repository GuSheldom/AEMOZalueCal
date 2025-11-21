import numpy as np
import sympy as sp
from sympy import symbols, exp, log, diff, simplify

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
    
    # 定义符号变量
    theta1, theta2, theta3, theta4 = symbols('theta1 theta2 theta3 theta4')
    w1, w2, w3, w4 = symbols('w1 w2 w3 w4')
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9')
    
    # 定义u值
    u1 = theta1*x1 + theta2*x2 + theta3*x4 + theta4*x5
    u2 = theta1*x2 + theta2*x3 + theta3*x5 + theta4*x6
    u3 = theta1*x4 + theta2*x5 + theta3*x7 + theta4*x8
    u4 = theta1*x5 + theta2*x6 + theta3*x8 + theta4*x9
    
    # 定义逻辑函数
    z = w1*u1 + w2*u2 + w3*u3 + w4*u4
    h = 1 / (1 + exp(-z))
    
    print("\n步骤1: 使用链式法则")
    print("∂h/∂θⱼ = (∂h/∂z) × (∂z/∂θⱼ)")
    print("∂h/∂wⱼ = (∂h/∂z) × (∂z/∂wⱼ)")
    
    print("\n步骤2: 计算 ∂h/∂z")
    dh_dz = diff(h, z)
    print(f"∂h/∂z = {dh_dz}")
    print("     = h(1-h)")  # 简化形式
    
    print("\n步骤3: 计算 ∂z/∂θⱼ")
    # ∂z/∂θ₁
    dz_dtheta1 = diff(z, theta1)
    print(f"∂z/∂θ₁ = {dz_dtheta1}")
    print("      = w₁x₁ + w₂x₂ + w₃x₄ + w₄x₅")
    
    # ∂z/∂θ₂
    dz_dtheta2 = diff(z, theta2)
    print(f"∂z/∂θ₂ = {dz_dtheta2}")
    print("      = w₁x₂ + w₂x₃ + w₃x₅ + w₄x₆")
    
    # ∂z/∂θ₃
    dz_dtheta3 = diff(z, theta3)
    print(f"∂z/∂θ₃ = {dz_dtheta3}")
    print("      = w₁x₄ + w₂x₅ + w₃x₇ + w₄x₈")
    
    # ∂z/∂θ₄
    dz_dtheta4 = diff(z, theta4)
    print(f"∂z/∂θ₄ = {dz_dtheta4}")
    print("      = w₁x₅ + w₂x₆ + w₃x₈ + w₄x₉")
    
    print("\n步骤4: 计算 ∂z/∂wⱼ")
    # ∂z/∂w₁
    dz_dw1 = diff(z, w1)
    print(f"∂z/∂w₁ = {dz_dw1}")
    
    # ∂z/∂w₂
    dz_dw2 = diff(z, w2)
    print(f"∂z/∂w₂ = {dz_dw2}")
    
    # ∂z/∂w₃
    dz_dw3 = diff(z, w3)
    print(f"∂z/∂w₃ = {dz_dw3}")
    
    # ∂z/∂w₄
    dz_dw4 = diff(z, w4)
    print(f"∂z/∂w₄ = {dz_dw4}")
    
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
    
    print("\n步骤1: 似然函数")
    print("对于二元分类，每个数据点的似然为：")
    print("P(yᵢ|xᵢ) = hᵢʸⁱ × (1-hᵢ)¹⁻ʸⁱ")
    print("其中 hᵢ = h(xᵢ, θ, w)")
    
    print("\n步骤2: 总似然函数")
    print("L(θ, w) = ∏ᵢ₌₁ᴺ P(yᵢ|xᵢ)")
    print("        = ∏ᵢ₌₁ᴺ hᵢʸⁱ × (1-hᵢ)¹⁻ʸⁱ")
    
    print("\n步骤3: 对数似然函数")
    print("ℓ(θ, w) = log L(θ, w)")
    print("         = ∑ᵢ₌₁ᴺ [yᵢ log hᵢ + (1-yᵢ) log(1-hᵢ)]")
    
    print("\n步骤4: 负对数似然函数")
    print("我们要最小化的负对数似然函数为：")
    print()
    print("NLL(θ, w) = -ℓ(θ, w)")
    print("          = -∑ᵢ₌₁ᴺ [yᵢ log hᵢ + (1-yᵢ) log(1-hᵢ)]")
    print("          = ∑ᵢ₌₁ᴺ [-yᵢ log hᵢ - (1-yᵢ) log(1-hᵢ)]")
    
    print("\n这就是二元交叉熵损失函数！")

def part_c_sgd_algorithm():
    """(c) 描述SGD算法"""
    print("\n" + "="*60)
    print("(c) SGD算法与mini-batch方法 [8 marks]")
    print("="*60)
    
    print("\n随机梯度下降(SGD)算法：")
    print("-" * 40)
    
    print("\n1. 初始化：")
    print("   - 随机初始化参数 θ⁽⁰⁾ 和 w⁽⁰⁾")
    print("   - 设置学习率 α > 0")
    print("   - 设置mini-batch大小 B")
    print("   - 设置最大迭代次数 T")
    
    print("\n2. 对于每个epoch t = 1, 2, ..., T：")
    print("   a) 随机打乱训练数据")
    print("   b) 将数据分成大小为B的mini-batches")
    print("   c) 对于每个mini-batch:")
    
    print("\n      i) 前向传播：")
    print("         - 对batch中每个样本(xᵢ, yᵢ)计算：")
    print("           * u₁ᵢ, u₂ᵢ, u₃ᵢ, u₄ᵢ (应用滤波器)")
    print("           * hᵢ = 1/(1 + e^(-(w₁u₁ᵢ + w₂u₂ᵢ + w₃u₃ᵢ + w₄u₄ᵢ)))")
    
    print("\n      ii) 计算mini-batch损失：")
    print("          L_batch = (1/B) × ∑ᵢ∈batch [-yᵢ log hᵢ - (1-yᵢ) log(1-hᵢ)]")
    
    print("\n      iii) 反向传播 - 计算梯度：")
    print("           对于每个参数θⱼ：")
    print("           ∂L_batch/∂θⱼ = (1/B) × ∑ᵢ∈batch ∂(-yᵢ log hᵢ - (1-yᵢ) log(1-hᵢ))/∂θⱼ")
    print("                        = (1/B) × ∑ᵢ∈batch (hᵢ - yᵢ) × (∂hᵢ/∂θⱼ)")
    
    print("\n           其中：∂hᵢ/∂θⱼ = hᵢ(1-hᵢ) × (∂zᵢ/∂θⱼ)")
    print("           类似地计算 ∂L_batch/∂wⱼ")
    
    print("\n      iv) 参数更新：")
    print("          θⱼ ← θⱼ - α × (∂L_batch/∂θⱼ)")
    print("          wⱼ ← wⱼ - α × (∂L_batch/∂wⱼ)")
    
    print("\n3. 重复直到收敛或达到最大迭代次数")
    
    print("\n关键要点：")
    print("- Mini-batch梯度是真实梯度的无偏估计")
    print("- Batch size B 是超参数，需要调优")
    print("- 学习率 α 控制收敛速度和稳定性")
    print("- 每个epoch后可以在验证集上评估性能")

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

Let $z = w_1 u_1 + w_2 u_2 + w_3 u_3 + w_4 u_4$, so $h = \frac{1}{1 + e^{-z}}$.

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
\frac{\partial z}{\partial w_1} &= u_1, \quad \frac{\partial z}{\partial w_2} = u_2, \quad \frac{\partial z}{\partial w_3} = u_3, \quad \frac{\partial z}{\partial w_4} = u_4
\end{align}

Therefore:
\begin{align}
\frac{\partial h}{\partial \theta_1} &= h(1-h)(w_1 x_1 + w_2 x_2 + w_3 x_4 + w_4 x_5) \\
\frac{\partial h}{\partial \theta_2} &= h(1-h)(w_1 x_2 + w_2 x_3 + w_3 x_5 + w_4 x_6) \\
\frac{\partial h}{\partial \theta_3} &= h(1-h)(w_1 x_4 + w_2 x_5 + w_3 x_7 + w_4 x_8) \\
\frac{\partial h}{\partial \theta_4} &= h(1-h)(w_1 x_5 + w_2 x_6 + w_3 x_8 + w_4 x_9)
\end{align}

\begin{align}
\frac{\partial h}{\partial w_1} &= h(1-h) u_1, \quad \frac{\partial h}{\partial w_2} = h(1-h) u_2 \\
\frac{\partial h}{\partial w_3} &= h(1-h) u_3, \quad \frac{\partial h}{\partial w_4} = h(1-h) u_4
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

\section*{(c) SGD Algorithm with Mini-batch [8 marks]}

\textbf{Stochastic Gradient Descent with Mini-batch Algorithm:}

\begin{enumerate}
\item \textbf{Initialize:}
   \begin{itemize}
   \item Randomly initialize parameters $\theta^{(0)}$ and $w^{(0)}$
   \item Set learning rate $\alpha > 0$
   \item Set mini-batch size $B$
   \item Set maximum epochs $T$
   \end{itemize}

\item \textbf{For each epoch $t = 1, 2, \ldots, T$:}
   \begin{enumerate}
   \item Randomly shuffle the training data
   \item Divide data into mini-batches of size $B$
   \item For each mini-batch $\mathcal{B}$:
   
   \textbf{Forward Pass:}
   \begin{itemize}
   \item For each sample $(x_i, y_i) \in \mathcal{B}$, compute:
   \item $u_{1i}, u_{2i}, u_{3i}, u_{4i}$ using the filter
   \item $h_i = \frac{1}{1 + e^{-(w_1 u_{1i} + w_2 u_{2i} + w_3 u_{3i} + w_4 u_{4i})}}$
   \end{itemize}
   
   \textbf{Compute Mini-batch Loss:}
   $$L_{\text{batch}} = \frac{1}{B} \sum_{i \in \mathcal{B}} [-y_i \log h_i - (1-y_i) \log(1-h_i)]$$
   
   \textbf{Backward Pass - Compute Gradients:}
   $$\frac{\partial L_{\text{batch}}}{\partial \theta_j} = \frac{1}{B} \sum_{i \in \mathcal{B}} (h_i - y_i) \cdot \frac{\partial h_i}{\partial \theta_j}$$
   $$\frac{\partial L_{\text{batch}}}{\partial w_j} = \frac{1}{B} \sum_{i \in \mathcal{B}} (h_i - y_i) \cdot \frac{\partial h_i}{\partial w_j}$$
   
   \textbf{Parameter Update:}
   $$\theta_j \leftarrow \theta_j - \alpha \cdot \frac{\partial L_{\text{batch}}}{\partial \theta_j}$$
   $$w_j \leftarrow w_j - \alpha \cdot \frac{\partial L_{\text{batch}}}{\partial w_j}$$
   \end{enumerate}

\item \textbf{Repeat until convergence or maximum epochs reached}
\end{enumerate}

\textbf{Key Points:}
\begin{itemize}
\item Mini-batch gradient is an unbiased estimator of the true gradient
\item Batch size $B$ is a hyperparameter that needs tuning
\item Learning rate $\alpha$ controls convergence speed and stability
\item Performance can be evaluated on validation set after each epoch
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

if __name__ == "__main__":
    main()
