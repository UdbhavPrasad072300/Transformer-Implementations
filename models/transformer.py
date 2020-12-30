import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math


# Multiheaded Attention for when Input Tensor: (B, S, E)
# B: Batch Size
# S: Sequence Length
# E: Embedding Size
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.2, batch_dim=0):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_dim = batch_dim
        
        self.dropout_layer = nn.Dropout(dropout)

        self.head_size = self.embed_size // self.num_heads

        assert self.head_size * self.num_heads == self.embed_size, "Heads cannot split Embedding size equally"

        self.Q = nn.Linear(self.embed_size, self.embed_size)
        self.K = nn.Linear(self.embed_size, self.embed_size)
        self.V = nn.Linear(self.embed_size, self.embed_size)

        self.linear = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, q, k, v, mask=None):
        if self.batch_dim == 0:
            out = self.batch_0(q, k, v, mask)
        elif self.batch_dim == 1:
            out = self.batch_1(q, k, v, mask)
    
        return out
    
    def batch_0(self, q, k, v, mask=None):
        q_batch_size, q_seq_len, q_embed_size = q.size()
        k_batch_size, k_seq_len, k_embed_size = k.size()
        v_batch_size, v_seq_len, v_embed_size = v.size()
        
        q = self.Q(q).reshape(q_batch_size, q_seq_len, self.num_heads, self.head_size)
        k = self.K(k).reshape(k_batch_size, k_seq_len, self.num_heads, self.head_size)
        v = self.V(v).reshape(v_batch_size, v_seq_len, self.num_heads, self.head_size)
        
        scores = self.attention(q, k, v, self.num_heads, mask)
        concatenated = scores.reshape(v_batch_size, -1, self.embed_size)
        
        out = self.linear(concatenated)

        return out
    
    def batch_1(self, q, k, v, mask=None):
        q_seq_len, q_batch_size, q_embed_size = q.size()
        k_seq_len, k_batch_size, k_embed_size = k.size()
        v_seq_len, v_batch_size, v_embed_size = v.size()

        q = self.Q(q).reshape(q_batch_size, q_seq_len, self.num_heads, self.head_size)
        k = self.K(k).reshape(k_batch_size, k_seq_len, self.num_heads, self.head_size)
        v = self.V(v).reshape(v_batch_size, v_seq_len, self.num_heads, self.head_size)
        
        scores = self.attention(q, k, v, self.num_heads, mask)
        concatenated = scores.reshape(-1, v_batch_size, self.embed_size)
        
        out = self.linear(concatenated)

        return out
    
    def attention(self, q, k, v, d_k, mask=None):
        scores = torch.einsum("bqhe,bkhe->bhqk", [q, k]) / math.sqrt(d_k)

        #if mask is not None:
        #    scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)
        out = torch.einsum("bhql,blhd->bqhd", [scores, v])

        return out
    
    
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=5000, d_model=300, dropout=0.1, device="cpu"):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0), :], requires_grad=False)
        return self.dropout(x)


class Transformer_Encoder(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.2, device="cpu"):
        super(Transformer_Encoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_hidden_size = ff_hidden_size
        self.dropout = dropout
        self.device = device

        self.Norm1 = nn.LayerNorm(self.embed_size)
        self.Norm2 = nn.LayerNorm(self.embed_size)

        self.multi_attention = nn.MultiheadAttention(self.embed_size, self.num_heads, self.dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, self.ff_hidden_size),
            nn.ReLU(),
            nn.Linear(self.ff_hidden_size, self.embed_size)
        )

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, mask=None):
        attention, _ = self.multi_attention(x, x, x, mask)
        x = self.dropout_layer(self.Norm1(x + attention))
        x = self.dropout_layer(self.Norm2(x + self.feed_forward(x)))
        return x


class Transformer_Decoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_ff, dropout=0.1, device="cpu"):
        super(Transformer_Decoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_ff = num_ff
        self.dropout = dropout
        self.device = device

        self.masked_multiheadattention = nn.MultiheadAttention(self.embed_size, self.num_heads, self.dropout)
        self.multiheadattention = nn.MultiheadAttention(self.embed_size, self.num_heads, self.dropout)

        self.Norm1 = nn.LayerNorm(self.embed_size)
        self.Norm2 = nn.LayerNorm(self.embed_size)
        self.Norm3 = nn.LayerNorm(self.embed_size)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, self.num_ff),
            nn.ReLU(),
            nn.Linear(self.num_ff, self.embed_size)
        )

    def forward(self, x, y, mask=None):
        y_mask = torch.tril(torch.ones((y.size(0), y.size(0)))).to(self.device)

        attention1, _ = self.masked_multiheadattention(y, y, y)
        
        y = self.dropout_layer(self.Norm1(y + attention1))

        attention2, _ = self.multiheadattention(y, x, x)
        
        x = self.dropout_layer(self.Norm2(y + attention2))

        x = self.dropout_layer(self.Norm3(x + self.feed_forward(x)))

        return x


class Transformer(nn.Module):
    def __init__(self, s_vocab_size, t_vocab_size, embed_size, num_heads, num_ff, encode_layers, decode_layers,
                 hidden_size, dropout=0.2, device="cpu"):
        super(Transformer, self).__init__()

        self.s_vocab_size = s_vocab_size
        self.t_vocab_size = t_vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_ff = num_ff
        self.encoder_num_layers = encode_layers
        self.decoder_num_layers = decode_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device
        
        self.dropout_layer = nn.Dropout(self.dropout)

        self.encoder_embed = nn.Embedding(self.s_vocab_size, embed_size)
        self.decoder_embed = nn.Embedding(self.t_vocab_size, embed_size)
        self.encoder_positional_encoding = PositionalEncoding(self.s_vocab_size, self.embed_size, device=device)
        self.decoder_positional_encoding = PositionalEncoding(self.t_vocab_size, self.embed_size, device=device)
        
        self.encoders = nn.ModuleList([])
        for layer in range(self.encoder_num_layers):
            self.encoders.append(Transformer_Encoder(self.embed_size, self.num_heads, self.hidden_size, dropout))

        self.decoders = nn.ModuleList([])
        for layer in range(self.decoder_num_layers):
            self.decoders.append(Transformer_Decoder(self.embed_size, self.num_heads, self.hidden_size, dropout,
                                                     self.device))

        self.final = nn.Linear(self.embed_size, self.t_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        x = self.encoder_embed(x) * math.sqrt(self.embed_size)
        y = self.decoder_embed(y) * math.sqrt(self.embed_size)
        
        x = self.encoder_positional_encoding(x)
        y = self.decoder_positional_encoding(y)

        for encoder in self.encoders:
            x = encoder(x)

        for decoder in self.decoders:
            y = decoder(x, y)

        y = self.final(y)

        return self.softmax(y)


class Transformer_with_nn(nn.Module):
    def __init__(self, s_vocab_size, t_vocab_size, embed_size, num_head, num_ff, encode_layers, decode_layers,
                 dropout=0.2, device="cpu"):
        super(Transformer_with_nn, self).__init__()

        self.s_vocab_size = s_vocab_size
        self.t_vocab_size = t_vocab_size
        self.embed_size = embed_size
        self.num_head = num_head
        self.num_ff = num_ff
        self.encoder_num_layers = encode_layers
        self.decoder_num_layers = decode_layers
        self.dropout = dropout
        self.device = device

        self.encoder_embed = nn.Embedding(self.s_vocab_size, embed_size)
        self.decoder_embed = nn.Embedding(self.t_vocab_size, embed_size)
        self.encoder_positional_encoding = PositionalEncoding(self.s_vocab_size, self.embed_size, device=device)
        self.decoder_positional_encoding = PositionalEncoding(self.t_vocab_size, self.embed_size, device=device)

        self.encoder_layer = nn.TransformerEncoderLayer(self.embed_size, self.num_head, self.num_ff,
                                                        dropout=self.dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, self.encoder_num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(self.embed_size, self.num_head, self.num_ff,
                                                        dropout=self.dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, self.decoder_num_layers)

        self.transformer = nn.Transformer(self.embed_size, self.num_head, self.encoder_num_layers, self.decoder_num_layers, self.num_ff, self.dropout)
        
        self.final = nn.Linear(self.embed_size, self.t_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.encoder_embed(x) / math.sqrt(self.embed_size)
        y = self.decoder_embed(y) / math.sqrt(self.embed_size)

        x = self.encoder_positional_encoding(x)
        y = self.decoder_positional_encoding(y)

        x = self.softmax(self.transformer(x, y))
        
        #memory = self.encoder(x)
        
        #out = self.decoder(y, memory)

        #x = self.final(out)
        #x = self.softmax(x)

        return x


class VisionEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, dropout=0.1):
        super(VisionEncoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.attention = MultiHeadAttention(self.embed_size, self.num_heads, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.embed_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attention(x, x, x)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size,
                 dropout=0.1):
        super(ViT, self).__init__()

        self.p = patch_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = channel_size * patch_size ** 2
        self.num_heads = num_heads
        self.classes = classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.embeddings = nn.Linear(self.patch_size, self.embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches+1, self.embed_size))

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(VisionEncoder(self.embed_size, self.num_heads, self.hidden_size, self.dropout))

        self.norm = nn.LayerNorm(self.embed_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.classes)
        )

    def forward(self, x, mask=None):
        b, c, h, w = x.size()

        x = x.reshape(b, int((h / self.p) * (w / self.p)), c * self.p * self.p)
        x = self.embeddings(x)

        b, n, e = x.size()

        class_tokens = self.class_token.expand(b, 1, e)
        x = torch.cat((x, class_tokens), dim=1)
        x = self.dropout_layer(x + self.positional_encoding[:, :(n+1)])
        
        for encoder in self.encoders:
            x = encoder(x)

        x = x[:, 0, :]

        x = F.log_softmax(self.classifier(self.norm(x)), dim=-1)

        return x
