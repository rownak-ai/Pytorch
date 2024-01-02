import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model,vocab_dim):
        super().__init__()
        self.d_model = d_model
        self.vocab_dim = vocab_dim
        self.embeddings = nn.Embedding(vocab_dim,d_model)

    def forward(self,x):
        return self.embeddings(x)*math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,seq_len,drop_out:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.drop_out = nn.Dropout(drop_out)

        #Create a matrix of sequence len and vector dimensions
        pe = torch.zeros(seq_len,d_model)

        #Create a vecotor of shape (seq_len,1)
        pos = torch.arange(0,seq_len,dtype=float).unsqueeze(1)
        den = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))

        #Now apply sig to even and cos to odd
        pe[:,0::2] = torch.sin(pos*den)
        pe[:,1::2] = torch.cos(pos*den)

        #We want batch dimensions to be added at 0th position
        pe = pe.unsqueeze(0) #(1,Seq_len,d_model)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x+(self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.drop_out(x)
    
class LayerNormalization(nn.Module):
    def __init__(self,eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)

        return self.aplha * (x-mean) / (std+self.eps)*self.bias

class feedforward(nn.Module):
    def __init__(self,d_model,d_ff,drop_out):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.drop_out = nn.Dropout(drop_out)
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.linear2(self.drop_out(nn.ReLU(self.linear1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,h,drop_out):
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        self.wq = nn.Linear(d_model,d_model)
        self.wk = nn.Linear(d_model,d_model)
        self.wv = nn.Linear(d_model,d_model)

        self.wo = nn.Linear(d_model,d_model)
        self.drop_out = nn.Dropout(drop_out)

    @staticmethod
    def attention(query,key,value,mask,dropout=nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query@key.transpose(-2,-1))/math.sqrt(d_k) #Batch,h,seq_len,dk
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = attention_scores.softmax(dim=-1) #Batch,h,seq_len,seq_len
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores@value),attention_scores
    
    def forward(self,q,k,v,mask):
        #(Batch,seq_len,d_modal)
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)

        #(Batch,h,seq_length,dk)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttention.attention(query,key,value,mask,self.drop_out)

        #(Batch,h,seq_len,dk)-->(Batch,seq_len,h,dk)-->(Batch,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.wo(x)
    
class ResidualConnection(nn.Module):
    def __init__(self,drop_out):
        super().__init__()
        self.drop_out = nn.Dropout(drop_out)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer):
        return x + self.drop_out(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block: MultiHeadAttention, feed_forward_block: feedforward, drop_out: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(drop_out) for _ in range(2)])

    def forward(self,x,src_mask):
        x = self.residual_connection[0](x,lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self,layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: feedforward, drop_out):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(drop_out) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residual_connection[0](x,lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connection[1](x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2](x,self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self,layer: nn.ModuleList):
        super().__init__()
        self.layer = layer
        self.norm = LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layer:
            x = layer(x,encoder_output,src_mask,tgt_mask)

        return self.norm(x)

class Projection(nn.Module):
    def __init__(self,d_model,voc_size):
        super().__init__()
        self.proj = nn.Linear(d_model,voc_size)

    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
    
class Transformer(nn.Module):
    def __init__(self,encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, src_position: PositionalEncoding, tgt_position: PositionalEncoding, projection_layer: Projection):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer

    def encoder(self,src,src_mask):
        src = self.src_embedding
        src = self.src_position(src)
        return self.encoder(src,src_mask)

    def decoder(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_position(tgt)
        return self.decoder(tgt,src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size,tgt_vocab_size,src_seq_len,tgt_seq_length,d_model=512,N=6,h=8,drop_out=0.1,d_ff=2048)->Transformer:
    #Create embedding layers for source and target
    src_embedding = InputEmbeddings(d_model,src_vocab_size)
    tgt_embedding = InputEmbeddings(d_model,tgt_vocab_size)


    #Create positional embeddings
    src_pos = PositionalEncoding(d_model,src_seq_len,drop_out)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_length,drop_out)

    encoder_block = []

    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model,h,drop_out)
        feed_forward_block = feedforward(d_model,d_ff,drop_out)
        encoder_block = EncoderBlock(encoder_self_attention,feed_forward_block,drop_out)
        encoder_block.append(encoder_block)

    #Create the decoder block
    decoder_block = []

    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model,h,drop_out)
        decoder_cross_attention_block = MultiHeadAttention(d_model,h,drop_out)
        feed_forward_block = feedforward(d_model,d_ff,drop_out)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,drop_out)
        decoder_block.append(decoder_block)

    #Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_block))
    decoder = Decoder(nn.ModuleList(decoder_block))

    #Projection
    projection = Projection(d_model,tgt_vocab_size)

    #Create transformer
    transformer = Transformer(encoder,decoder,src_embedding,tgt_embedding,src_pos,tgt_pos,projection)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
