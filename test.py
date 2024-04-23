#!/usr/bin/env python

# coding: utf-8



# In[1]:





get_ipython().system('pip install -f https://storage.googleapis.com/jax-releases/libtpu_releases.html jax[tpu]==0.4.8')





# In[2]:





from functools import partial

import numpy as np

import jax

import jax.numpy as jnp

from jax.sharding import PartitionSpec as PS

from jax.experimental.pjit import pjit, with_sharding_constraint

from jax.sharding import Mesh





# In[3]:





jax.__version__





# In[7]:





@partial(pjit, in_shardings=PS(), out_shardings=PS())

def function(x):

    jax.debug.inspect_array_sharding(x, callback=print)

    return jnp.sum(x)





# In[8]:





mesh = Mesh(np.array(jax.devices()).reshape(-1), ['dp'])





# In[9]:





with mesh:

    print(function(np.ones(4)))
