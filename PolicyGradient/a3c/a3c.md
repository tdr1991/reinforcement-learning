## 异步训练（Asynchronous Methods）
将异步训练（Asynchronous Methods）的方法应用到强化学习的各个算法中（Sarsa,one-step Q-learning n-step Q-learning和 advantage actor-critic），异步训练的方式应用在A2C中的效果最好，于是就有了A3C(Asynchronous advantage actor-critic)。

**Asynchronous Methods初衷是为了解决：在线学习获得的训练数据不稳定，而且数据与数据之间的相关性比较大。**

相对于replay memory, Asynchronous Methods优点在于： 
（1）可以将算法应用在on-policy。
（2）减少大量的显存，可以在多核CPU上进行训练，大大少训练成本。

## algorithm

- //assume global shared parameter vectors $\theta$ and $\theta_v$ and global shared counter $T=0$
- //assume thread-specific parameter vector $\theta'$ and $\theta_v'$
- initialize thread step counter $t\leftarrow 1$
- repeat
    - reset gradients:$d\theta \leftarrow 0$ and $d\theta_v \leftarrow 0$
    - synchronize thread-specific parameters $\theta' = \theta$ and $\theta_v' = \theta_v$
    - $t_{start} = t$
    - get state $s_t$
    - repeat
        - perform $a_t$ according to policy $\pi(a_t|s_t;\theta')$
        - receive reward $r_t$ and new state $s_{t+1}$
        - $t\leftarrow t+1$
        - $T \leftarrow T+1$
    - until terminal $s_t$ or $t - s_{start} == t_{max}$
    - $$R=\begin{cases}
    0 & for \space terminal \space s_t\\
    V(s_t,\theta_v') & for \space non-terminal \space s_t\\
    \end{cases}
    $$
    - for $i \in \{t-1,...,t_{start}\}$ do
        - $R \leftarrow r_i + \gamma R$ 
        - accumulate gradients wrt $\theta':d\theta \leftarrow d\theta + \nabla_{\theta'}log\pi(a_i|s_i;\theta')(R-V(s_i;\theta_v'))$
        - accumulate gradients wrt $\theta_v':d\theta_v + \alpha(R-V(s_i;\theta_v'))^2/\alpha\theta_v'$
    - end for 
    - perform asynchronous update of $\theta$ using $d\theta \space and \space of \space \theta_v \space using \space d\theta_v$
- until $T>T_{max}$