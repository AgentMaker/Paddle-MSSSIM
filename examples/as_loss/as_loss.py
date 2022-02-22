import os
import sys
import paddle
import numpy as np

from PIL import Image
from paddle.optimizer import Adam
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '..'))
from paddle_msssim import ssim, ms_ssim, SSIM, MS_SSIM


loss_type = 'msssim'
assert loss_type in ['ssim', 'msssim']

if loss_type == 'ssim':
    loss = ssim
    loss_obj = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
else:
    loss = ms_ssim
    loss_obj = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)


npImg1 = np.array(Image.open("einstein.png"))

img1 = paddle.to_tensor(npImg1).unsqueeze(0).unsqueeze(0) / 255.0
img2 = paddle.rand(img1.shape)

img1 = paddle.to_tensor(img1, stop_gradient=True)
img2 = paddle.to_tensor(img2, stop_gradient=False)

ssim_value = ms_ssim(img1, img2).item()
print("Initial ssim:", ssim_value)

optimizer = Adam(parameters=[img2], learning_rate=0.01)

step = 0
while ssim_value < 0.9999:
    step += 1
    optimizer.clear_grad()
    _ssim_loss = 1-loss_obj(img1, img2)
    _ssim_loss.backward()
    optimizer.step()

    ssim_value = loss(img1, img2).item()
    print('step: %d %s_loss: %f' % (step, loss_type, ssim_value))

img2_ = (img2 * 255.0).squeeze()
np_img2 = img2_.detach().numpy().astype(np.uint8)
Image.fromarray(np_img2).save('results.png')
