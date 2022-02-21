import numpy as np
import time
from PIL import Image
from skimage.metrics import structural_similarity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES']='' # disable CUDA for tf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from paddle_msssim import ssim, ms_ssim
import paddle
import tensorflow as tf

if __name__ == '__main__':
    img = Image.open('kodim10.png')
    img = np.array(img).astype(np.float32)
    N_repeat = 100

    ##########
    #   SSIM
    ##########
    img_batch = []
    img_noise_batch = []
    single_image_ssim = []
    print('===================================')
    print("             Test SSIM")
    print('===================================')
    print("====> Single Image")
    print("Repeat %d times"%(N_repeat))
    # params = paddle.nn.Parameter( paddle.ones(img.shape[2], img.shape[0], img.shape[1]), requires_grad=True ) # C, H, W
    for sigma in range(0, 101, 10):
        noise = sigma * np.random.rand(*img.shape)
        img_noise = (img + noise).astype(np.float32).clip(0,255)
        
        img_tf = tf.expand_dims( tf.convert_to_tensor(img),0 )
        img_noise_tf = tf.expand_dims( tf.convert_to_tensor(img_noise),0 )
        begin = time.time()

        for _ in range(N_repeat):
            ssim_tf = tf.image.ssim(img_tf, img_noise_tf, 255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        time_tf = (time.time()-begin) / N_repeat

        begin = time.time()
        for _ in range(N_repeat):
            ssim_skimage = structural_similarity(img, img_noise, win_size=11, multichannel=True,
                                    sigma=1.5, data_range=255, use_sample_covariance=False, gaussian_weights=True)
        time_skimage = (time.time()-begin) / N_repeat

        img_paddle = paddle.to_tensor(img).unsqueeze(0).transpose((0, 3, 1, 2))  # 1, C, H, W
        img_noise_paddle = paddle.to_tensor(img_noise).unsqueeze(0).transpose((0, 3, 1, 2))

        img_batch.append(img_paddle)
        img_noise_batch.append(img_noise_paddle)

        begin = time.time()
        for _ in range(N_repeat):
            ssim_paddle = ssim(img_noise_paddle, img_paddle, win_size=11, data_range=255)
        time_paddle = (time.time()-begin) / N_repeat

        ssim_paddle = ssim_paddle.numpy()
        single_image_ssim.append(ssim_paddle)
        ssim_tf = ssim_tf.numpy()

        print("sigma=%.1f ssim_skimage=%.6f (%.4f ms), ssim_tf=%.6f (%.4f ms), ssim_paddle=%.6f (%.4f ms)" % (
            sigma, ssim_skimage, time_skimage*1000, ssim_tf, time_tf*1000, ssim_paddle, time_paddle*1000))
        assert (np.allclose(ssim_paddle, ssim_skimage, atol=5e-4))
        assert (np.allclose(ssim_paddle, ssim_tf, atol=5e-4))

    print("Pass!")

    print("====> Batch")
    img_batch = paddle.concat(img_batch, axis=0)
    img_noise_batch = paddle.concat(img_noise_batch, axis=0)
    ssim_batch = ssim(img_noise_batch, img_batch, win_size=11,size_average=False, data_range=255)
    assert np.allclose(ssim_batch.numpy(), np.concatenate(single_image_ssim, 0), atol=5e-4)
    print("Pass!")

    ##########
    #   MS-SSIM
    ##########
    print('\n')
    
    img_batch = []
    img_noise_batch = []
    single_image_ssim = []
    print('===================================')
    print("             Test MS-SSIM")
    print('===================================')
    print("====> Single Image")
    print("Repeat %d times"%(N_repeat))
    # params = paddle.nn.Parameter( paddle.ones(img.shape[2], img.shape[0], img.shape[1]), requires_grad=True ) # C, H, W
    for sigma in range(0, 101, 10):
        noise = sigma * np.random.rand(*img.shape)
        img_noise = (img + noise).astype(np.float32).clip(0,255)
        
        img_batch.append(img)
        img_noise_batch.append(img_noise)

        img_tf = tf.expand_dims( tf.convert_to_tensor(img),0 )
        img_noise_tf = tf.expand_dims( tf.convert_to_tensor(img_noise),0 )
        begin = time.time()

        for _ in range(N_repeat):
            msssim_tf = tf.image.ssim_multiscale(img_tf, img_noise_tf, 255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        time_tf = (time.time()-begin) / N_repeat

        img_paddle = paddle.to_tensor(img).unsqueeze(0).transpose((0, 3, 1, 2))  # 1, C, H, W
        img_noise_paddle = paddle.to_tensor(img_noise).unsqueeze(0).transpose((0, 3, 1, 2))

        begin = time.time()
        for _ in range(N_repeat):
            msssim_paddle = ms_ssim(img_noise_paddle, img_paddle, win_size=11, data_range=255)
        time_paddle = (time.time()-begin) / N_repeat

        msssim_paddle = msssim_paddle.numpy()
        msssim_tf = msssim_tf.numpy()

        print("sigma=%.1f msssim_tf=%.6f (%.4f ms), msssim_paddle=%.6f (%.4f ms)" % (
            sigma, msssim_tf, time_tf*1000, msssim_paddle, time_paddle*1000))
        assert (np.allclose(msssim_tf, msssim_paddle, atol=5e-4))

    print("Pass")
    print("====> Batch")
    _img_batch= np.stack( img_batch, axis=0 )
    _img_noise_batch = np.stack( img_noise_batch, axis=0 )

    img_batch = paddle.to_tensor(_img_batch).transpose((0, 3, 1, 2))
    img_noise_batch = paddle.to_tensor(_img_noise_batch).transpose((0, 3, 1, 2))
    msssim_paddle_batch = ms_ssim(img_noise_batch, img_batch, win_size=11, size_average=False, data_range=255)

    img_batch = tf.convert_to_tensor( _img_batch)
    img_noise_batch = tf.convert_to_tensor(_img_noise_batch)
    msssim_tf_batch = tf.image.ssim_multiscale( img_noise_batch, img_batch, 255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03  )
    assert np.allclose(msssim_paddle_batch.numpy().reshape(-1), msssim_tf_batch.numpy().reshape(-1), atol=5e-4)
    print("Pass")




