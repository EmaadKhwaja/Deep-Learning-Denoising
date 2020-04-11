#!/usr/bin/env python3

# coding: utf-8

# In[37]:


import os
print(os.getcwd())


# In[38]:


##!jupyter nbconvert --to python 'Wynton Send Job.ipynb' --stdout --template=hashbang.tpl > wynton_send_job.sh
##!qsub -cwd -q gpu.q -j no wynton_send_job.sh


# In[45]:


##%%javascript
##IPython.notebook.kernel.execute(`notebookName = '${IPython.notebook.notebook_name}'`);


# In[46]:


##notebookName


# In[ ]:




