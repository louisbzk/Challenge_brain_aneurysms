#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data_augmentation import DataEnhancer
from utils import load_data


# In[6]:


data = load_data('challenge_dataset/')


# In[7]:


data_enhancer = DataEnhancer(data)


# In[8]:


# try different params
data_enhancer.rand_rotate(max_abs_rot=45.)
data_enhancer.crop(final_size=32)
# data_enhancer.clean_labels(dist_from_center_threshold=10.)
data_enhancer.contrast_raws(factor=1.9)
# data_enhancer.sharpen(factor=1.)
# data_enhancer.cluster_raws(n_colors=8)


# In[9]:


labels = data_enhancer.labels()
labels.max()


# In[14]:


data_enhancer.save('augmented_data', index=2)


# In[ ]:
