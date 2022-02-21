import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from paddle_msssim import *
import paddle

s = SSIM(data_range=1.)

a = paddle.randint(0, 255, shape=(20, 3, 256, 256)).cast(paddle.float32) / 255.
b = a * 0.5
a.stop_gradient = False
b.stop_gradient = False

for _ in range(500):
    loss = s(a, b)
    loss.backward()
