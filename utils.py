
'''
Setup utils.py which contains Config, get_learner
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
from fastai.metrics import *
from pathlib import Path
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

config = Config(
    testing=False,
    bert_model_name='bert-base-chinese', 
    max_lr=2e-5,#The recommended lr is 3e-5 in bert paper, but 2e-5 is better in this project
    epochs=1,
    use_fp16=False, #learner.to_fp16() Mixup precision can speedup training
    bs=8,
    text_cols  = [],
    label_cols = [],
    max_seq_len=256, # Max value in bert is 512
    num_labels = 2,  # 0:negative, 1:positive for default
    train_file = '',
    test_file = '' #None if no test data

)



class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]
      

      
def get_toknvocab(config:Config):
  bert_tok = BertTokenizer.from_pretrained(
      config.bert_model_name,
  )

  fastai_tokenizer = Tokenizer(
      tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), 
      pre_rules=[], 
      post_rules=[]
  )
  fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
  return fastai_tokenizer, fastai_bert_vocab

def get_learner(df_train:pd.DataFrame,df_test:pd.DataFrame=None,config:Config=None):
  
  fastai_tokenizer, fastai_bert_vocab = get_toknvocab(config)
  
  train_df, valid_df = train_test_split(df_train,random_state=42)
  if config.testing:
    train = train_df.head(1024)
    valid = valid_df.head(1024)
  else:
    train = train_df
    valid = valid_df

  databunch = TextClasDataBunch.from_df(".", train, valid, df_test,
                    tokenizer=fastai_tokenizer,
                    vocab=fastai_bert_vocab,
                    include_bos=False,
                    include_eos=False,
                    text_cols= config.text_cols,
                    label_cols= config.label_cols,
                    bs=config.bs,
                    collate_fn=partial(pad_collate, pad_first=False, pad_idx=0)
               )

  loss_func = nn.CrossEntropyLoss()

  bert_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 
                              'bertForSequenceClassification', 
                               config.bert_model_name, 
                               num_labels=config.num_labels)

  learner = Learner(
      databunch, 
      bert_model,
      loss_func=loss_func,
      metrics=[accuracy]
  )

  return learner