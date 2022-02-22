# Train an autoencoder with SSIM & MS-SSIM

## Prepare dataset
* Download CLIC datase from http://clic.compression.cc/2021/tasks/index.html.

* Unzip them into datasets.

* The structure of the directory:

    ```yaml
    - datasets
      - CLIC
        - train
          - *.png
          - ...
        - valid
          - *.png
          - ...
    ```

## Train
* SSIM loss:

    ```bash
    $ python train.py --loss_type ssim
    ```

* MS-SSIM loss:

    ```bash
    $ python train.py --loss_type ms_ssim
    ```
    