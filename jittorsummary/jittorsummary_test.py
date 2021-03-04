import unittest
import jittor as jt
from jittor import init
from jittor import nn
from jittorsummary import summary, summary_string
from tests.test_models.test_model import SingleInputNet, MultipleOutputNet, MultipleInputNet, MultipleInputNetDifferentDtypes

class torchsummaryTests(unittest.TestCase):
    def test_single_input(self):
        model = SingleInputNet()
        input = (1, 28, 28)
        total_params, trainable_params = summary(model, input)
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)

    def test_multiple_input(self):
        model = MultipleInputNet()
        input1 = (1, 300)
        input2 = (1, 300)
        total_params, trainable_params = summary(
            model, [input1, input2])
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

    def test_single_layer_network(self):
        model = nn.Linear(2, 5)
        input = (1, 2)
        total_params, trainable_params = summary(model, input)
        self.assertEqual(total_params, 15)
        self.assertEqual(trainable_params, 15)

    def test_single_layer_network_on_gpu(self):
        model = nn.Linear(2, 5)
        input = (1, 2)
        total_params, trainable_params = summary(model, input, device='cuda')
        self.assertEqual(total_params, 15)
        self.assertEqual(trainable_params, 15)

    def test_multiple_input_types(self):
        model = MultipleInputNetDifferentDtypes()
        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = [jt.float, jt.int64]
        total_params, trainable_params = summary(
            model, [input1, input2], dtypes=dtypes)
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

    def test_multiple_output_types(self):
        model = MultipleOutputNet()
        input = (1, 28, 28)
        total_params, trainable_params = summary(model, input)
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)


class torchsummarystringTests(unittest.TestCase):
    def test_single_input(self):
        model = SingleInputNet()
        input = (1, 28, 28)
        result, (total_params, trainable_params) = summary_string(
            model, input)
        self.assertEqual(type(result), str)
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)

    def test_single_input_on_gpu(self):
        model = SingleInputNet()
        input = (1, 28, 28)
        result, (total_params, trainable_params) = summary_string(
            model, input, device='cuda')
        self.assertEqual(type(result), str)
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)


if __name__ == '__main__':
    unittest.main(buffer=True)