import numpy as np
import urllib
import time
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from paddle_msssim import ssim
import paddle


if __name__ == "__main__":
    # comparing 2d and 3d implementation. A thin 3d slice should result in a similar measure as a 2d image.
    img = Image.open("kodim10.png")
    img = np.array(img).astype(np.float32)

    img_batch = []
    img_noise_batch = []
    single_image_ssim = []
    img_batch_3d = []
    img_noise_batch_3d = []
    single_image_ssim_3d = []
    N_repeat = 1
    print("====> Single Image")
    print("Repeat %d times" % (N_repeat))
    # params = paddle.nn.Parameter( paddle.ones(img.shape[2], img.shape[0], img.shape[1]), requires_grad=True ) # C, H, W
    for sigma in range(0, 101, 10):
        noise = sigma * np.random.rand(*img.shape)
        img_noise = (img + noise).astype(np.float32).clip(0, 255)

        img_paddle = paddle.to_tensor(img).unsqueeze(0).transpose((0, 3, 1, 2))  # 1, C, H, W
        img_noise_paddle = paddle.to_tensor(img_noise).unsqueeze(0).transpose((0, 3, 1, 2))

        img_batch.append(img_paddle)
        img_noise_batch.append(img_noise_paddle)

        begin = time.perf_counter()
        for _ in range(N_repeat):
            ssim_paddle = ssim(img_noise_paddle, img_paddle, win_size=11, data_range=255)

        time_paddle = (time.perf_counter() - begin) / N_repeat

        ssim_paddle = ssim_paddle.numpy()
        single_image_ssim.append(ssim_paddle)


        img_paddle_3d = img_paddle.unsqueeze(2).expand((-1, -1, 11, -1, -1))  # 1, C, H, W -> 1, C, T, H, W
        img_noise_paddle_3d = img_noise_paddle.unsqueeze(2).expand((-1, -1, 11, -1, -1))

        img_batch_3d.append(img_paddle_3d)
        img_noise_batch_3d.append(img_noise_paddle_3d)

        begin = time.perf_counter()
        for _ in range(N_repeat):
            ssim_paddle_3d = ssim(img_noise_paddle_3d, img_paddle_3d, win_size=11, data_range=255)

        time_paddle_3d = (time.perf_counter() - begin) / N_repeat

        ssim_paddle_3d = ssim_paddle_3d.numpy()
        single_image_ssim_3d.append(ssim_paddle_3d)


        print(
            "sigma=%f ssim_paddle=%f (%f ms) ssim_paddle_3d=%f (%f ms)"
            % (sigma, ssim_paddle, time_paddle * 1000, ssim_paddle_3d, time_paddle_3d * 1000)
        )

        # Image.fromarray( img_noise.astype('uint8') ).save('simga_%d_ssim_%.4f.png'%(sigma, ssim_paddle.item()))
        assert np.allclose(ssim_paddle, ssim_paddle_3d, atol=5e-4)

    print("Pass")
