
# coding: utf-8

# In[1]:


import gym
import numpy as np
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from lib.envs.cliff_walking import CliffWalkingEnv


# In[2]:


env = CliffWalkingEnv()

print(env.reset())
env.render()

print(env.step(0))
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

print(env.step(2))
env.render()

