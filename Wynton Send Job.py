#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def send_to_wynton(code):
    '#!/bin/python3'
    python3


# In[1]:


import io
import os
from notebook.utils import to_api_path

_script_exporter = None
_html_exporter = None

def script_post_save(model, os_path, contents_manager, **kwargs):
    """convert notebooks to Python script after save with nbconvert
    replaces `ipython notebook --script`
    """
    from nbconvert.exporters.script import ScriptExporter
    from nbconvert.exporters.html import HTMLExporter

    if model['type'] != 'notebook':
        return

    global _script_exporter
    if _script_exporter is None:
        _script_exporter = ScriptExporter(parent=contents_manager)
    log = contents_manager.log

    # save .py file
    base, ext = os.path.splitext(os_path)
    script, resources = _script_exporter.from_filename(os_path)
    script_fname = base + resources.get('output_extension', '.txt')
    log.info("Saving script /%s", to_api_path(script_fname, contents_manager.root_dir))
    with io.open(script_fname, 'w', encoding='utf-8') as f:
        f.write(script)

c.FileContentsManager.post_save_hook = script_post_save


# In[7]:


script_post_save('notebook','a','abs')


# In[14]:


ScriptExporter(parent=contents_manager)


# In[15]:


resources.get('output_extension', '.txt')


# In[17]:


script_post_save


# In[19]:


import sys,json

f = open(sys.argv[1], 'r') #input.ipynb
j = json.load(f)
of = open(sys.argv[2], 'w') #output.py
if j["nbformat"] >=4:
        for i,cell in enumerate(j["cells"]):
                of.write("#cell "+str(i)+"\n")
                for line in cell["source"]:
                        of.write(line)
                of.write('\n\n')
else:
        for i,cell in enumerate(j["worksheets"][0]["cells"]):
                of.write("#cell "+str(i)+"\n")
                for line in cell["input"]:
                        of.write(line)
                of.write('\n\n')

of.close()


# In[ ]:


import sys,json

f = open(sys.argv[1], 'r') #input.ipynb
j = json.load(f)
of = open(sys.argv[2], 'w') #output.py
if j["nbformat"] >=4:
        for i,cell in enumerate(j["cells"]):
                of.write("#cell "+str(i)+"\n")
                for line in cell["source"]:
                        of.write(line)
                of.write('\n\n')
else:
        for i,cell in enumerate(j["worksheets"][0]["cells"]):
                of.write("#cell "+str(i)+"\n")
                for line in cell["input"]:
                        of.write(line)
                of.write('\n\n')

of.close()

