#!/usr/bin/env python
# coding: utf-8

# In[6]:


from keras.callbacks import Callback
import matplotlib.pyplot as plt


# In[7]:


import json


# In[14]:


class HistoryGraph(Callback):
    def __init__(self, model_path_name):
        self.model_path_name = model_path_name
        
    def on_train_begin(self, logs=None):
        self.eppoch = []
        self.history = {}
        
    def on_epoch_end(seelf , epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.save_training_history(self.model_path_name, self.history)
    
    def save_training_history(self, path, history):
        for metric in history:
            if "val" not in metric:
                plt.clf()
                history[metric] = list(map(float, history[metric]))
                plt.plot(history[metric])
                plt.plot(history["val_" + metric])
                plt.title('model' + metric)
                plt.ylabel(metric)
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.gcf().savefig(path + '/' + metrci + '_history' + '.jpg')
        with open(path + '/log' + '.json', 'w') as fp:
            json.dump(history, fp, indent=True)


# In[ ]:




