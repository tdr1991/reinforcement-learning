
# coding: utf-8

# In[1]:


import gym
import numpy as np
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from lib.envs.windy_gridworld import WindyGridworldEnv


# In[2]:


env = WindyGridworldEnv()

print(env.reset())
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

print(env.step(2))
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

