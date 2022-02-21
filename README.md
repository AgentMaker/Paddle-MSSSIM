# Paddle-MSSSIM

## Introduction
* Fast and differentiable MS-SSIM and SSIM for Paddle.

    <img src="./images/lcs.png" width="25%">

* Structural Similarity (SSIM): 

    <img src="./images/ssim.png" width="50%">

* Multi-Scale Structural Similarity (MS-SSIM):

    <img src="./images/ms-ssim.png" width="55%">

## Installation
* via pip

    ```bash
    $ pip install paddle-msssim
    ```

* via sources

    ```bash
    $ git clone https://github.com/AgentMaker/Paddle-MSSSIM

    $ cd Paddle-MSSSIM

    $ python setup.py install
    ```

## Usage

* Basic Usage 

    ```python
    from paddle_msssim import ssim, ms_ssim, SSIM, MS_SSIM
    # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
    # Y: (N,3,H,W)  

    # calculate ssim & ms-ssim for each image
    ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
    ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

    # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
    ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
    ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

    # reuse the gaussian kernel with SSIM & MS_SSIM. 
    ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

    ssim_loss = 1 - ssim_module(X, Y)
    ms_ssim_loss = 1 - ms_ssim_module(X, Y)
    ```

* Normalized input

    ```python
    '''
    If you need to calculate MS-SSIM/SSIM on normalized images
    Please denormalize them to the range of [0, 1] or [0, 255] first
    '''
    # X: (N,3,H,W) a batch of normalized images (-1 ~ 1)
    # Y: (N,3,H,W)  
    X = (X + 1) / 2  # [-1, 1] => [0, 1]
    Y = (Y + 1) / 2  
    ms_ssim_val = ms_ssim( X, Y, data_range=1, size_average=False ) #(N,)
    ```

## References
 
* [SSIM Research](https://ece.uwaterloo.ca/~z70wang/research/ssim/)  
* [MS-SSIM Paper](https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf)  
* [Matlab Code](https://ece.uwaterloo.ca/~z70wang/research/iwssim/)   
* [Pytorch Code](https://github.com/VainF/pytorch-msssim) 
* [TensorFlow Code](https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/image_ops_impl.py#L3314-L3438) 