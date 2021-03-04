# Keras style `model.summary()` in Jittor



### 1. Usage

- `git clone https://github.com/liuruiyang98/Jittor-summary.git`

```python
from jittorsummary import summary
summary(your_model, input_size=(channels, H, W))
```

* Note that the `input_size` is required to make a forward pass through the network.
* Note that jittorsummary with **cuda** is not support.
  * `jt.flags.use_cuda = 0`



### 2. Example

#### 2.1 CNN for MNIST

```python
import jittor as jt
from jittor import init
from jittor import nn
from jittorsummary import summary

class SingleInputNet(nn.Module):

    def __init__(self):
        super(SingleInputNet, self).__init__()
        self.conv1 = nn.Conv(1, 10, 5)
        self.conv2 = nn.Conv(10, 20, 5)
        self.conv2_drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def execute(self, x):
        x = nn.relu(nn.max_pool2d(self.conv1(x), 2))
        x = nn.relu(nn.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(((-1), 320))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)
      
model = SingleInputNet()
summary(model, (1, 28, 28))
```

```txt
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
              Conv-1           [-1, 10, 24, 24]             260
              Conv-2             [-1, 20, 8, 8]           5,020
           Dropout-3             [-1, 20, 8, 8]               0
            Linear-4                   [-1, 50]          16,050
            Linear-5                   [-1, 10]             510
    SingleInputNet-6                   [-1, 10]               0
================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 0.08
Estimated Total Size (MB): 0.15
----------------------------------------------------------------
```



#### 2.2 Multiple Inputs

```python
import jittor as jt
from jittor import init
from jittor import nn
from jittorsummary import summary

class MultipleInputNet(nn.Module):

    def __init__(self):
        super(MultipleInputNet, self).__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)
        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def execute(self, x1, x2):
        x1 = nn.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = nn.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = jt.contrib.concat((x1, x2), dim=0)
        return nn.log_softmax(x, dim=1)
      
model = MultipleInputNet()
summary(model, [(1, 300), (1, 300)])
```

```txt
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                [-1, 1, 50]          15,050
            Linear-2                [-1, 1, 10]             510
            Linear-3                [-1, 1, 50]          15,050
            Linear-4                [-1, 1, 10]             510
  MultipleInputNet-5                [-1, 1, 10]               0
================================================================
Total params: 31,120
Trainable params: 31,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.34
Forward/backward pass size (MB): 0.00
Params size (MB): 0.12
Estimated Total Size (MB): 0.46
----------------------------------------------------------------
```



#### 2.3 Try more models

We provide the implementation of **UNet, UNet++** and **Dense-UNet**, which are based on Pytorch and Jittor, respectively. You can compare the results of `torchsummary` and `jittorsummary`.

```txt
|- jittorsummary
	|- tests
		|- test_models
			|- DenseUNet_jittor.py
			|- DenseUNet_pytorch.py
			|- NestedUNet_jittor.py
			|- NestedUNet_pytorch.py
			|- UNet_jittor.py
			|- UNet_pytorch.py
```



### 3. Pytorch-to-Jittor

* Please refer to [pytorch-to-jittor](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-2-16-43-pytorchconvert/) to find the documentation related to the conversion of the pytorch model to jittor.
* Online converter click this link [pt-converter](https://cg.cs.tsinghua.edu.cn/jittor/pt_converter/).



### References

* The idea for this package sparked from [pytorch-summary](https://github.com/sksq96/pytorch-summary).

