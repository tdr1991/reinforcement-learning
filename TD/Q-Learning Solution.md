# 阅读笔记
## 自定义环境CliffWalkingEnv
环境共有4\*12=48个状态，其中的37～47（从0开始）为终态。用4\*12的矩阵表示，每个状态的动作可以上下左右移动一格。例如状态15索引为[1, 3](从0
开始索引的)，动作有$\uparrow[-1, 0],\rightarrow[0,1],\downarrow[1,0],\leftarrow[0,-1]$，此时$\uparrow[-1, 0]$,那么新状态的索引为[0,3],对应的值为3。如果新状态为37～46奖励为-100，否则-1。如果新状态为37～47表示回合结束。

## Q-Learning (Off Policy TD Learning)
离线学习是根据过往数据进行参数更新，不会使用新状态的数据学习。
$ Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma max_{\acute a} Q(\acute{s},\acute{a}) - Q(s,a)] $
```bash
初始化Q表，shape=(s,a),s为状态，a为动作
for 每一个回合
    重置环境获取当前状态
    for 直到回合结束

        把当前状态传给策略函数生成动作的概率矩阵分布
        根据上一步的概率矩阵分布随机选择一个动作
        环境根据选择的动作执行一步step返回next_state, reward, done等信息

        根据新状态next_state获得Q值最大的动作
        td_target = reward + discount_factor * Q[next_state][best_next_action]
        td_delta = td_target - Q[state][action]
        Q[state][action] += alpha * td_delta

        把新状态赋值为当前状态
    done
done
```

## 回合更新（MC）与单步更新（TD）
回合更新需要等到回合结束才会去更新行为准则，但所有的节点数据都会保存，训练的数据就是本次回合的数据。单步更新就是每一步都会去更新行为准则，更新的时候可以一条数据（q-learning，sarsa）或batch_size条（dqn）。两种更新的时候并不只有一条数据，都可以采用minibatch的思想。