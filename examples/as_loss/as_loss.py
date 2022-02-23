import os
import sys
import paddle
import numpy as np

from PIL import Image
from paddle.optimizer import Adam
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '..'))
from paddle_msssim import SSIM, MS_SSIM


loss_type = 'ssim'
assert loss_type in ['ssim', 'msssim']

if loss_type == 'ssim':
    loss_func = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
else:
    loss_func = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)


np_img1 = np.array(Image.open("einstein.png"))

img1 = paddle.to_tensor(np_img1).unsqueeze(0).unsqueeze(0) / 255.0
img2 = paddle.rand(img1.shape)

img1 = paddle.to_tensor(img1, stop_gradient=True)
img2 = paddle.to_tensor(img2, stop_gradient=False)

with paddle.no_grad():
    ssim_value = loss_func(img1, img2).item()
    print("Initial %s: %f:" % (loss_type, ssim_value))

optimizer = Adam(parameters=[img2], learning_rate=0.01)

step = 0
while ssim_value < 0.9999:
    step += 1
    optimizer.clear_grad()
    loss = loss_func(img1, img2)
    (1-loss).backward()
    optimizer.step()

    ssim_value = loss.item()
    if step % 10 == 0:
        print('Step: %d %s_loss: %f' % (step, loss_type, ssim_value))

img2 = (img2 * 255.0).squeeze()
np_img2 = img2.detach().numpy().astype(np.uint8)
print('Save the result image to result.png')
Image.fromarray(np_img2).save('result.png')
