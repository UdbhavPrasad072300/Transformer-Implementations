import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_package.models.transformer import Transformer, ViT, BERT, GPT

torch.manual_seed(0)


def test1(DEVICE):
    x = torch.rand(2, 50).type(torch.LongTensor).to(DEVICE)

    print("Input Dimensions: {}".format(x.size()))

    out = model(x)

    print("Output Dimensions: {}".format(out.size()))
    print("-" * 100)

    del x


def test2(DEVICE):
    x = torch.rand(2, 50).type(torch.LongTensor).to(DEVICE)
    y = torch.rand(2, 50).type(torch.LongTensor).to(DEVICE)

    print("Input Dimensions: {} & {}".format(x.size(), y.size()))

    out = model(x, y)

    print("Output Dimensions: {}".format(out.size()))
    print("-" * 100)

    del x, y


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used: {}".format(DEVICE))

    # Vanilla Transformers

    source_vocab_size = 1000
    target_vocab_size = 1200
    embed_size = 512
    num_head = 8
    num_ff = 1024
    encoder_layers = 2
    decoder_layers = 2
    hidden_size = 256
    dropout = 0.2

    model = Transformer(source_vocab_size, target_vocab_size, embed_size, num_head, num_ff, encoder_layers,
                        decoder_layers, hidden_size=hidden_size, dropout=dropout, device=DEVICE).to(DEVICE)

    print("-" * 100)
    print(model)
    print("-" * 100)
    del model

    # BERT

    classes = 10

    model = BERT(vocab_size=source_vocab_size, classes=10, embed_size=embed_size, num_layers=encoder_layers,
                 num_heads=num_head, hidden_size=hidden_size, dropout=dropout, device=DEVICE).to(DEVICE)

    print("-" * 100)
    print(model)
    test1(DEVICE)
    print("-" * 100)
    del model

    # GPT

    model = GPT(vocab_size=source_vocab_size, embed_size=embed_size, num_layers=decoder_layers, num_heads=num_head,
                hidden_size=hidden_size, dropout=dropout, device=DEVICE).to(DEVICE)

    print("-" * 100)
    print(model)
    test2(DEVICE)
    print("-" * 100)
    del model

    print("Program has Ended")
