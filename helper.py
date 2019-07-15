

'''
Setup utils.py which contains seed_everything, get_preds_ordered,
print_test_metrics,get_data,get_true_labels
'''
import os
import torch
import random
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from pathlib import Path
from sklearn.model_selection import train_test_split

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
def get_preds_ordered(ds_type:DatasetType,learner:Learner):
    """
    The get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in learner.data.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]
  
def print_test_metrics(true_labels,learner):
  preds = get_preds_ordered(DatasetType.Test,learner)
  pred_values = np.argmax(preds, axis = 1)
  acc, f1s = accuracy_score(true_labels,pred_values),f1_score(true_labels,pred_values)
  print(f'The accuracy is {round(acc*100,2)}%, the f1_score is {round(f1s*100,2)}%')
  
def get_data(path,train_file:str,test_file:str):
  df_train = pd.read_csv(Path(path)/train_file)
  if test_file != None:
    df_test = pd.read_csv(Path(path)/test_file)
  else: 
    df_test = None
  return df_train,df_test   


def get_true_labels(df,config_label:list)->np.array:
  item = config_label[0]
  true_labels = np.array(df[item].tolist())
  return true_labels
