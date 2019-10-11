## policy gradient
通过奖励值来判断这个动作的相对好坏。如果一个动作得到的reward多，那么就使其出现的概率增加，如果一个动作得到的reward少，就使其出现的概率减小。
定义损失函数：$loss=-log(policy(s_t,a_t)) * v_t$
策略梯度算法采用回合更新，式中的$v_t$代表的是当前状态$s$下采取动作$a$所能得到的奖励，它是当前奖励和未来奖励的贴现求和。即未来奖励会乘以一个衰减因子。

### Monte-Carlo Policy Gradient(REINFORCE)

- function REINFORCE
    - Initialise $\theta$ arbitrarily 
    - for each episode $\{s_1,a_1,r_2,...,s_{T−1},a_{T−1},r_T\}∼\pi_\theta$ do
        - for $t = 1$ to $T − 1$ do
            - $\theta ← \theta + α∇_θ log π_θ (s_t , a_t )v_t$
        - end for 
    - end for
    - return $\theta$
- end function

### 优缺点
#### 优点
- Better convergence properties （更好的收敛属性）
- Effective in high-dimensional or continuous action spaces（在高维度和连续动作空间更加有效）
- Can learn stochastic policies（可以Stochastic 的策略）

#### 缺点
- Typically converge to a local rather than global optimum（通常得到的都是局部最优解）
- Evaluating a policy is typically inefficient and high variance （评价策略通常不是非常高效，并且有很高的偏差）
