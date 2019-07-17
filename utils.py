
'''
Setup utils.py which contains Config and get_learner
'''
import os
import torch
import numpy as np
from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from fastai.metrics import *
from pathlib import Path
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer,BertForSequenceClassification,BertModel


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

config_l = Config(
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

class easy_classification(BertForSequenceClassification):
    def __init__(self, config):
        super(easy_classification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs[0]  # (loss), logits, (hidden_states), (attentions)
      

  



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
      

      
def get_toknvocab(config_l:Config):
  bert_tok = BertTokenizer.from_pretrained(
      config_l.bert_model_name,
  )

  fastai_tokenizer = Tokenizer(
      tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config_l.max_seq_len), 
      pre_rules=[], 
      post_rules=[]
  )
  fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
  return fastai_tokenizer, fastai_bert_vocab

def get_learner(df_train:pd.DataFrame,df_test:pd.DataFrame=None,config_l:Config=None):
  
  fastai_tokenizer, fastai_bert_vocab = get_toknvocab(config_l)
  
  train_df, valid_df = train_test_split(df_train,random_state=42)
  if config_l.testing:
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
                    text_cols= config_l.text_cols,
                    label_cols= config_l.label_cols,
                    bs=config_l.bs,
                    collate_fn=partial(pad_collate, pad_first=False, pad_idx=0)
               )

  loss_func = nn.CrossEntropyLoss()

  
  bert_model = easy_classification.from_pretrained(config_l.bert_model_name,
                                                             num_labels=config_l.num_labels)

    
    

  learner = Learner(
      databunch, 
      bert_model,
      loss_func=loss_func,
      metrics=[accuracy]
  )

  return learner