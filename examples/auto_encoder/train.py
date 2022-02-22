import os
import sys
import paddle
import argparse

from PIL import Image
from models import AutoEncoder
from datas import ImageDataset
from paddle.vision import transforms
from paddle.optimizer import Adam
from paddle.io import DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '..'))
from paddle_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100*(1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*(1 - super(SSIM_Loss, self).forward(img1, img2))


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")
    parser.add_argument("--loss_type", type=str,
                        default='ssim', choices=['ssim', 'ms_ssim'])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--total_epochs", type=int, default=50)
    return parser


def main():
    if not os.path.exists('results'):
        os.mkdir('results')

    opts = get_argparser().parse_args()

    # dataset
    train_trainsform = transforms.Compose([
        transforms.RandomCrop(size=512, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.CenterCrop(size=512),
        transforms.ToTensor()
    ])

    train_loader = DataLoader(
        ImageDataset(root='datasets/CLIC/train', transform=train_trainsform),
        batch_size=opts.batch_size, shuffle=True, num_workers=0, drop_last=True)

    val_loader = DataLoader(
        ImageDataset(root='datasets/CLIC/valid', transform=val_transform),
        batch_size=1, shuffle=False, num_workers=0)

    print("Train set: %d, Val set: %d" %
          (len(train_loader.dataset), len(val_loader.dataset)))
    model = AutoEncoder(C=128, M=128, in_chan=3, out_chan=3)

    # optimizer
    optimizer = Adam(parameters=model.parameters(),
                     learning_rate=1e-4,
                     weight_decay=1e-5)

    # checkpoint
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        model.set_dict(paddle.load(opts.ckpt))
    else:
        print("[!] Retrain")

    if opts.loss_type == 'ssim':
        criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=3)
    else:
        criterion = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)

    #==========   Train Loop   ==========#
    for cur_epoch in range(opts.total_epochs):
        # =====  Train  =====
        model.train()
        for cur_step, (images, ) in enumerate(train_loader):
            optimizer.clear_grad()
            outputs = model(images)

            loss = criterion(outputs, images)
            loss.backward()

            optimizer.step()

            if (cur_step) % opts.log_interval == 0:
                print("Epoch %d, Batch %d/%d, loss=%.6f" %
                      (cur_epoch, cur_step, len(train_loader), loss.item()))

        # =====  Save Latest Model  =====
        paddle.save(model.state_dict(), 'latest_model.pdparams')

        # =====  Validation  =====
        print("Val...")
        best_score = 0.0
        cur_score = test(opts, model, val_loader, cur_epoch)
        print("%s = %.6f" % (opts.loss_type, cur_score))
        # =====  Save Best Model  =====
        if cur_score > best_score:  # save best model
            best_score = cur_score
            paddle.save(model.state_dict(), 'best_model.pdparams')
            print("Best model saved as best_model.pt")


def test(opts, model, val_loader, epoch):
    save_dir = os.path.join('results', 'epoch_%d' % epoch)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model.eval()
    cur_score = 0.0

    metric = ssim if opts.loss_type == 'ssim' else ms_ssim

    with paddle.no_grad():
        for i, (images, ) in enumerate(val_loader):
            outputs = model(images)
            # save the first reconstructed image
            cur_score += metric(outputs, images, data_range=1.0)
            Image.fromarray((outputs*255).squeeze(0).detach().numpy().astype('uint8').transpose(1, 2, 0)).save(os.path.join(save_dir, 'recons_%s_%d.png' % (opts.loss_type, i)))
        cur_score /= len(val_loader.dataset)

    return cur_score


if __name__ == '__main__':
    main()
