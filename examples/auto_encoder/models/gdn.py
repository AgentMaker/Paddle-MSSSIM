import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GDN(nn.Layer):
    def __init__(self,
                 num_features,
                 inverse=False,
                 gamma_init=.1,
                 beta_bound=1e-6,
                 gamma_bound=0.0,
                 reparam_offset=2**-18,
                 ):
        super(GDN, self).__init__()
        self._inverse = inverse
        self.num_features = num_features
        self.reparam_offset = reparam_offset
        self.pedestal = self.reparam_offset**2

        beta_init = paddle.sqrt(paddle.ones((num_features, ), dtype=paddle.float32) + self.pedestal)
        gama_init = paddle.sqrt(paddle.full((num_features, num_features), fill_value=gamma_init, dtype=paddle.float32)
                                * paddle.eye(num_features, dtype=paddle.float32) + self.pedestal)

        self.beta = self.create_parameter(
            shape=beta_init.shape, default_initializer=nn.initializer.Assign(beta_init))
        self.gamma = self.create_parameter(
            shape=gama_init.shape, default_initializer=nn.initializer.Assign(gama_init))

        self.beta_bound = (beta_bound + self.pedestal) ** 0.5
        self.gamma_bound = (gamma_bound + self.pedestal) ** 0.5

    def _reparam(self, var, bound):
        var = paddle.clip(var, min=bound)
        return (var**2) - self.pedestal

    def forward(self, x):
        gamma = self._reparam(self.gamma, self.gamma_bound).reshape((self.num_features, self.num_features, 1, 1))  # expand to (C, C, 1, 1)
        beta = self._reparam(self.beta, self.beta_bound)
        norm_pool = F.conv2d(x ** 2, gamma, bias=beta, stride=1, padding=0)
        norm_pool = paddle.sqrt(norm_pool)

        if self._inverse:
            norm_pool = x * norm_pool
        else:
            norm_pool = x / norm_pool
        return norm_pool
