


## SARSA (On Policy TD Learning)
在线学习根据当前状态采取的动作之后的新状态进行学习。
$ Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma Q(\acute{s},\acute{a}) - Q(s,a)] $
```bash
初始化Q表，shape=(s,a),s为状态，a为动作
for 每一个回合 do
    重置环境获取当前状态
    把当前状态传给策略函数生成动作的概率矩阵分布
    根据上一步的概率矩阵分布随机选择一个动作
    for 直到回合结束 do

        环境根据选择的动作执行一步step返回next_state, reward, done等信息

        把新状态next_state传给策略函数生成动作的概率矩阵分布
        根据上一步的概率矩阵分布随机选择一个动作next_action
        
        td_target = reward + discount_factor * Q[next_state][next_action]
        td_delta = td_target - Q[state][action]
        Q[state][action] += alpha * td_delta

        把新动作赋值给当前动作
        把新状态赋值为当前状态
    done
done
```
