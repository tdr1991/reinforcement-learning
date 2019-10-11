

## 伪代码
PPO，Actor-Critic Style

- for $ iteration=1,2,... $ do
    - for $ actor=1,2,...,N $ do
        - Run policy $\pi_{\theta_{old}}$ in environment for $ T$ timesteps
        - Compute advantage estimates $A'_1,...,A'_T$
    - end for
    Optimize surrogate $ L $ wrt $ \theta$ , with $K$ epochs and minibatch size $ M\leq NT$
    - $\theta_{old} \leftarrow \theta$
- end for

## PPO目地
存在问题
- 标准的策略梯度方法每个数据样本更新一次梯度。
- Q-learning（具有函数逼近）许多简单的问题不能解决并且很难理解， vanilla policy gradient方法的数据有效性差以及稳健性差，信任区域策略优化（TRPO）相对复杂，并且与包含噪声（例如丢失）或参数共享的体系结构不兼容（在策略和值函数之间，或与辅助任务之间）。
- 在开发可扩展（对于大型模型和并行实现），数据有效和稳健（即，在没有超参数调整的情况下的各种问题上成功）的方法方面存在改进的空间。


解决方法
- 小批量更新。
- 具有TRPO的优点，更容易实现，更通用，更好的样本复杂性。
- 在基准测试中，取得了复杂性、简洁性和实际时间之间的一个平衡。
- 一阶优化，有限概率比的新目标。