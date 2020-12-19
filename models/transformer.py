import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def attention(q, k, v, d_k, mask=None, dropout=0.2):
    dropout_layer = nn.Dropout(dropout)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    scores = dropout_layer(scores)
    out = torch.matmul(scores, v)

    return out


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=2000, embedding_size=300, dropout=0.2, device="cpu"):
        super(PositionalEncoding, self).__init__()
        import math

        self.dropout = nn.Dropout(p=dropout)

        self.pe_matrix = torch.zeros(max_seq_len, embedding_size).to(device)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        self.pe_matrix[:, 0::2] = torch.sin(
            position * div_term)  # Source: https://pytorch.org/tutorials/beginner/transformer_tutorial
        self.pe_matrix[:, 1::2] = torch.cos(position * div_term)

        self.pe_matrix = self.pe_matrix.unsqueeze(0).transpose(0, 1)

        self.register_buffer("Positional Encoding", self.pe_matrix)

    def forward(self, x):
        x = x + self.pe_matrix[:x.size(0), :]
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.2):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.head_size = self.embed_size // self.num_heads

        assert self.head_size * self.num_heads == self.embed_size, "Heads cannot split Embedding size equally"

        self.Q = nn.Linear(self.embed_size, self.embed_size)
        self.K = nn.Linear(self.embed_size, self.embed_size)
        self.V = nn.Linear(self.embed_size, self.embed_size)

        self.linear = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.Q(q).reshape(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = self.K(k).reshape(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.V(v).reshape(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        scores = attention(q, k, v, self.num_heads, mask, self.dropout)
        concatenated = scores.transpose(1, 2).reshape(batch_size, -1, self.embed_size)
        out = self.linear(concatenated)

        return out


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

        self.multi_attention = MultiHeadAttention(self.embed_size, self.num_heads, self.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, self.ff_hidden_size),
            nn.ReLU(),
            nn.Linear(self.ff_hidden_size, self.embed_size),
            nn.Dropout(self.dropout),
        )

    def forward(self, x, mask=None):
        x += self.Norm1(self.multi_attention(x, x, x, mask, self.dropout))
        x += self.Norm2(self.feed_forward(x))
        return x


class Transformer_Decoder(nn.Module):
    def __init__(self):
        super(Transformer_Decoder, self).__init__()

        self.masked_multiheadattention = None
        self.multiheadattention = None

        self.Norm1 = None
        self.Norm2 = None
        self.Norm3 = None

        self.feed_forward = None

    def forward(self, x, y):
        return


class Transformer_Implemented(nn.Module):
    def __init__(self):
        super(Transformer_Implemented, self).__init__()

    def forward(self):
        return


class Transformer(nn.Module):
    def __init__(self, s_vocab_size, t_vocab_size, embed_size, num_head, num_ff, encode_layers, decode_layers,
                 dropout=0.2, device="cpu"):
        super(Transformer, self).__init__()

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

        self.final = nn.Linear(self.embed_size, self.t_vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, x, y):
        x = self.encoder_embed(x) * math.sqrt(self.embed_size)
        y = self.decoder_embed(y) * math.sqrt(self.embed_size)

        x = self.encoder_positional_encoding(x)
        y = self.decoder_positional_encoding(y)

        memory = self.encoder(x)

        out = self.decoder(y, memory)

        x = self.final(out)
        x = self.softmax(x)
        
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

        self.ff = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.embed_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attention(x, x, x)
        x = x + self.ff(self.norm2(x))
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
        self.dropout = nn.Dropout(dropout)

        self.embeddings = nn.Linear(self.patch_size, self.embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(VisionEncoder(self.embed_size, self.num_heads, self.hidden_size, dropout))

        self.norm = nn.LayerNorm(self.embed_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.classes)
        )

    def forward(self, x):
        b, c, h, w = x.size()

        x = x.reshape(b, int((h / self.p) * (w / self.p)), c * self.p * self.p)
        x = self.embeddings(x)

        b, n, e = x.size()
        class_tokens = self.class_token.expand(b, 1, e)

        x = torch.cat((x, class_tokens), dim=1)

        for encoder in self.encoders:
            x = encoder(x)

        x = x[:, 0, :]

        x = F.log_softmax(self.classifier(self.norm(x)), dim=-1)

        return x