import torch

from transformer_package.models.transformer import DeiT

torch.manual_seed(0)


class Test_DeiT:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_out_size(self, img_size=28, patch_size=7, channel_size=3, n_classes=10, batch_size=2):
        x = torch.rand(batch_size, channel_size, img_size, img_size).to(self.device)

        embed_size = 512
        num_heads = 8
        num_layers = 2
        hidden_size = 256
        dropout = 0.2

        model = DeiT(img_size,
                     channel_size,
                     patch_size,
                     embed_size,
                     num_heads,
                     n_classes,
                     num_layers,
                     hidden_size,
                     dropout=dropout
                     ).to(self.device)

        y = model(x)

        assert list(y.size()) == [batch_size, n_classes], "Output Size Incorrect"

        del model
        del x
        del y
        torch.cuda.empty_cache()


if __name__ == '__main__':
    tester = Test_DeiT()
    tester.test_out_size(90)
    tester.test_out_size(100)
    tester.test_out_size(110)
