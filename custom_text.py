

#sys libs
import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")

#data manupulation libs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Initialization
pandarallel.initialize()


#string manupulation libs
import re
import string
from string import digits
import spacy

#load data
data = pd.read_csv('D:\\pytorch\\Hindi_English_Truncated_Corpus.csv')
data = data.reset_index(drop=True)
data.drop('source',axis=1,inplace=True)

#preprocess
data = data.dropna().drop_duplicates()

#lower and remove quotes
data['english_sentence'] = data.english_sentence.apply(lambda x: re.sub("'", '',x).lower())
data['hindi_sentence'] = data.hindi_sentence.apply(lambda x: re.sub("'", '', x).lower())
    
#remove special chars
exclude = set(string.punctuation)#set of all special chars
#remove all the special chars
data['english_sentence'] = data.english_sentence.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
data['hindi_sentence'] = data.hindi_sentence.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    
remove_digits = str.maketrans('','',digits)
data['english_sentence'] = data.english_sentence.apply(lambda x: x.translate(remove_digits))
data['hindi_sentence'] = data.hindi_sentence.apply(lambda x: x.translate(remove_digits))

data['hindi_sentence'] = data.hindi_sentence.apply(lambda x: re.sub("[२३०८१५७९४६]","",x))

val_frac = 0.1 #precentage data in val
val_split_idx = int(len(data)*val_frac) #index on which to split
data_idx = list(range(len(data))) #create a list of ints till len of data
np.random.shuffle(data_idx)

#get indexes for validation and train
val_idx, train_idx = data_idx[:val_split_idx], data_idx[val_split_idx:]
print('len of train: ', len(train_idx))
print('len of val: ', len(val_idx))

#create the sets
train = data.iloc[train_idx].reset_index().drop('index',axis=1)
val = data.iloc[val_idx].reset_index().drop('index',axis=1)

  #initiate the index to token dict
        ## <PAD> -> padding, used for padding the shorter sentences in a batch to match the length of longest sentence in the batch
        ## <SOS> -> start token, added in front of each sentence to signify the start of sentence
        ## <EOS> -> End of sentence token, added to the end of each sentence to signify the end of sentence
        ## <UNK> -> words which are not found in the vocab are replace by this token

  #max_size : max source vocab size. Eg. if set to 10,000, we pick the top 10,000 most frequent words and discard others

class Vocabulary:
    def __init__(self,freq_threshold,max_size):
        self.itos = {0:'<PAD>',1:'<SOS>',2:'<EOS>',3:'<UNK>'}
        self.stoi = {k:j for j,k in self.itos.items()}

        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer(text):
        return [tok.lower().strip() for tok in text.split(' ')]
    
    '''
    build the vocab: create a dictionary mapping of index to string (itos) and string to index (stoi)
    output ex. for stoi -> {'the':5, 'a':6, 'an':7}
    '''
    def build_vocabulary(self,sentence_list):
        #index from which we want our dict to start. We already used 4 indexes for pad, start, end, unk
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
            
            frequencies = {k:v for k,v in frequencies.items() if v>self.freq_threshold}

            #limit vocab to the max_size specified
        frequencies = dict(sorted(frequencies.items(), key = lambda x: -x[1])[:self.max_size-idx]) # idx =4 for pad, start, end , unk

        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx+=1

    def numericalize(self,text):
        tokenized_text = self.tokenizer(text)
        numericalized_text = []

        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else:
                numericalized_text.append(self.stoi['<UNK>'])
        return numericalized_text
    
voc  = Vocabulary(0,100)
sentence_list = ['this is a cat','that is a dog']
voc.build_vocabulary(sentence_list)
print('index to string: ',voc.itos)
print('string to index: ',voc.stoi)

print('numericalize ->cat and a dog: ',voc.numericalize('cat and a dog'))

from torch.utils.data import Dataset
import torch

class Train_Dataset(Dataset):
    def __init__(self,df,source_column,target_column,transform=None,freq_threshold=5,source_vocab_max_size=10000,
                 target_vocab_max_size=10000):
        
        self.df = df
        self.transform = transform
        
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]

        self.source_vocab = Vocabulary(freq_threshold,source_vocab_max_size)
        self.source_vocab.build_vocabulary(self.source_texts.tolist())

        self.target_vocab = Vocabulary(freq_threshold,target_vocab_max_size)
        self.source_vocab.build_vocabulary(self.target_texts.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]

        if self.transform is not None:
            source_text = self.transform(source_text)

        numerialized_source = [self.source_vocab.stoi['<SOS>']]
        numerialized_source += self.source_vocab.numericalize(source_text)
        numerialized_source.append(self.source_vocab.stoi['<EOS>'])

        numerialized_target = [self.target_vocab.stoi['<SOS>']]
        numerialized_target += self.target_vocab.numericalize(target_text)
        numerialized_target.append(self.target_vocab.stoi['<EOS>'])

        return torch.tensor(numerialized_source), torch.tensor(numerialized_target)

train_dataset = Train_Dataset(train,'english_sentence','hindi_sentence')
print(train.iloc[1])
print(train_dataset[1])

class Validation_Dataset:
    def __init__(self,train_dataset,df,source_column,target_column,transform=None):
        pass

class MyCollate:
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self,batch):
        source = [item[0] for item in batch]
        source = pad_sequence(source,batch_first=False,padding_value=self.pad_idx)

        target = [item[1] for item in batch]
        target = pad_sequence(target,batch_first=False,padding_value=self.pad_idx)
        return source,target
    
def get_train_loader(dataset,batch_size,num_workers=0,shuffle=True,pin_memory=True):
    pad_idx = dataset.source_vocab.stoi['<PAD>']
    loader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,
                        pin_memory=pin_memory,collate_fn=MyCollate(pad_idx=pad_idx))
    return loader    

def get_valid_loader(dataset,train_dataset,batch_size,num_workers=0,shuffle=True,pin_memory=True):
    pad_idx = train_dataset.source_vocab.stoi['<PAD>']
    loader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False,
                        pin_memory=pin_memory,collate_fn=MyCollate(pad_idx=pad_idx))
    return loader

train_loader = get_train_loader(train_dataset,32)
source = next(iter(train_loader))[0]
target = next(iter(train_loader))[1]

print('source: \n', source)

print('source shape: ',source.shape)
print('target shape: ',target.shape)