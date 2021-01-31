import torch

from transformer_package.models.transformer import Transformer

torch.manual_seed(0)


class Test_Transformer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_out_size(self):

        model = Transformer().to(self.device)

        y = model(x)

        assert list(y.size()) == [batch_size, n_classes], "Output Size Incorrect"

        del model
        del x
        del y
        torch.cuda.empty_cache()


if __name__ == '__main__':
    tester = Transformer()
    tester.test_out_size()
    tester.test_out_size()
    tester.test_out_size()
