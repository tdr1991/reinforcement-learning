
# coding: utf-8

# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')

import gym
import numpy as np
from matplotlib import pyplot as plt


# In[34]:


env = gym.envs.make("Breakout-v0")


# In[35]:


print("Action space size: {}".format(env.action_space.n))
print(env.get_action_meanings()) # env.unwrapped.get_action_meanings() for gym 0.8.0 or later

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

plt.figure()
plt.imshow(env.render(mode='rgb_array'))

[env.step(2) for x in range(1)]
plt.figure()
plt.imshow(env.render(mode='rgb_array'))

env.render(close=True)


# In[73]:


# Check out what a cropped image looks like
plt.imshow(observation[34:-16,:,:])

