import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer import Transformer, ViT

torch.manual_seed(0)


def test(size):
    x, y = torch.rand(50, 64).type(torch.LongTensor).to(device), torch.rand(size, 64).type(torch.LongTensor).to(
        device)

    print("Input Dimensions: {} & {}".format(x.size(), y.size()))

    out = model(x, y)

    print("Output Dimensions: {}".format(out.size()))
    print("-" * 100)


def main():
    return


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used: {}".format(device))

    # Model Parameters
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
                        decoder_layers, hidden_size=hidden_size, dropout=dropout, device=device).to(device)

    print("-" * 100)
    print(model)
    print("-" * 100)

    test(40)
    test(50)
    test(60)

    print("Program has Ended")
