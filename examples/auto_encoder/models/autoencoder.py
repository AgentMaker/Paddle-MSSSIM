import paddle.nn as nn
import paddle.nn.functional as F

from .gdn import GDN


# https://arxiv.org/pdf/1611.01704.pdf
# A simplfied version without quantization
class AutoEncoder(nn.Layer):
    def __init__(self, C=128, M=128, in_chan=3, out_chan=3):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(C=C, M=M, in_chan=in_chan)
        self.decoder = Decoder(C=C, M=M, out_chan=out_chan)

    def forward(self, x, **kargs):
        code = self.encoder(x)
        out = self.decoder(code)
        return out


class Encoder(nn.Layer):
    """ Encoder
    """

    def __init__(self, C=32, M=128, in_chan=3):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2D(in_channels=in_chan, out_channels=M,
                      kernel_size=5, stride=2, padding=2, bias_attr=False),
            GDN(M),

            nn.Conv2D(in_channels=M, out_channels=M, kernel_size=5,
                      stride=2, padding=2, bias_attr=False),
            GDN(M),

            nn.Conv2D(in_channels=M, out_channels=M, kernel_size=5,
                      stride=2, padding=2, bias_attr=False),
            GDN(M),

            nn.Conv2D(in_channels=M, out_channels=C, kernel_size=5,
                      stride=2, padding=2, bias_attr=False)
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Layer):
    """ Decoder
    """

    def __init__(self, C=32, M=128, out_chan=3):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Conv2DTranspose(in_channels=C, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias_attr=False),
            GDN(M, inverse=True),

            nn.Conv2DTranspose(in_channels=M, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias_attr=False),
            GDN(M, inverse=True),

            nn.Conv2DTranspose(in_channels=M, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias_attr=False),
            GDN(M, inverse=True),

            nn.Conv2DTranspose(in_channels=M, out_channels=out_chan, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias_attr=False),
        )

    def forward(self, q):
        return F.sigmoid(self.dec(q))
