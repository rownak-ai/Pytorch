import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,dataset,src_tokenizer,tgt_tokenizer,src_lang,tgt_lang,seq_len):
        super().__init__()

        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([src_tokenizer.token_to_id(['[SOS]'])],dtype=torch.int64)
        self.end_token = torch.Tensor([tgt_tokenizer.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token = torch.Tensor([tgt_tokenizer.token_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index)
        src_target_pair = self.dataset[index]
        src_txt = src_target_pair['translation'][self.src_lang]
        tgt_txt = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.src_tokenizer.encode(src_txt).ids
        dec_input_tokens = self.tgt_tokenizer.encode(tgt_txt).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                self.end_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64)
            ]
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                self.end_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            ]
        )

        return {
            'encoder_input': encoder_input,
            'decoder_input':decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_length)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), #(1,seq_length) & (1,seq_length,seq_length)
            'label': label, #(seq_length)
            'src_txt': src_txt,
            'tgt_txt':tgt_txt
        }
    
def casual_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask == 0


