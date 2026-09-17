"""
USRP 预对接版接收前端：

1) 同步：基于导频的公共相位校正
2) 信道估计：LS 估计 + 可选神经残差修正
3) 均衡：单抽头 MMSE/零迫均衡 + 可选神经残差修正

适用场景：
- 先在离线仿真中训练/验证
- 后续替换为 USRP 实测 IQ 数据

说明：
- 这是单载波/单抽头风格的最小可落地版本
- 如果后续换成 OFDM，可把信道估计/均衡扩展为逐子载波版本
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Layer, Model
from tensorflow.keras.layers import Dense, LayerNormalization


def _complex_to_features(z: tf.Tensor) -> tf.Tensor:
    """复数张量 -> 实值特征。"""
    return tf.stack(
        [
            tf.math.real(z),
            tf.math.imag(z),
            tf.abs(z),
            tf.math.angle(z),
        ],
        axis=-1,
    )


class PilotSynchronizer(Layer):
    """基于已知导频的公共相位同步。"""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def call(self, y_pilot: tf.Tensor, x_pilot: tf.Tensor, y_data: tf.Tensor):
        """
        y_pilot: [B, P]  接收导频
        x_pilot: [B, P]  已知发射导频
        y_data:   [B, S]  待同步数据
        """
        phase_err = tf.math.angle(
            tf.reduce_sum(y_pilot * tf.math.conj(x_pilot), axis=1, keepdims=True)
            + tf.cast(self.eps, y_pilot.dtype)
        )
        rot = tf.exp(tf.complex(tf.zeros_like(phase_err), -phase_err))
        y_pilot_corr = y_pilot * rot
        y_data_corr = y_data * rot
        return y_pilot_corr, y_data_corr, phase_err


class LSChannelEstimator(Layer):
    """导频 LS 信道估计：h = sum(y x*) / sum(|x|^2)。"""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def call(self, y_pilot: tf.Tensor, x_pilot: tf.Tensor):
        num = tf.reduce_sum(y_pilot * tf.math.conj(x_pilot), axis=1, keepdims=True)
        den = tf.reduce_sum(tf.abs(x_pilot) ** 2, axis=1, keepdims=True)
        den_c = tf.cast(den + tf.cast(self.eps, den.dtype), num.dtype)
        h_hat = num / den_c
        return h_hat


class LMMSEDFTChannelEstimator(Layer):
    """LMMSE + DFT 去噪（导频域）信道估计。"""

    def __init__(self, dft_taps: int = 4, eps: float = 1e-8):
        super().__init__()
        self.dft_taps = int(max(1, dft_taps))
        self.eps = eps

    def call(self, y_pilot: tf.Tensor, x_pilot: tf.Tensor, no: tf.Tensor | None = None):
        # 逐导频子载波 LS
        h_tone = y_pilot / (x_pilot + tf.cast(self.eps, x_pilot.dtype))  # [B,P]

        # DFT 去噪：限制时域 taps 数
        h_td = tf.signal.ifft(tf.cast(h_tone, tf.complex64))
        p = tf.shape(h_td)[1]
        taps = tf.minimum(tf.cast(self.dft_taps, tf.int32), p)
        mask = tf.concat([
            tf.ones([tf.shape(h_td)[0], taps], dtype=h_td.dtype),
            tf.zeros([tf.shape(h_td)[0], p - taps], dtype=h_td.dtype),
        ], axis=1)
        h_fd_denoised = tf.signal.fft(h_td * mask)

        # 平坦衰落等效：对导频域均值作为单抽头信道
        h_ls = tf.reduce_mean(h_fd_denoised, axis=1, keepdims=True)  # [B,1]

        # LMMSE shrinkage（简化近似）
        if no is None:
            return h_ls

        no = tf.cast(no, tf.float32)
        if no.shape.rank == 0:
            no = tf.fill([tf.shape(h_ls)[0], 1], no)
        elif no.shape.rank == 1:
            no = tf.reshape(no, [-1, 1])

        ex = tf.reduce_mean(tf.abs(x_pilot) ** 2, axis=1, keepdims=True)
        sigma_h2 = tf.maximum(tf.abs(h_ls) ** 2 - (no / (ex + self.eps)), self.eps)
        alpha = sigma_h2 / (sigma_h2 + (no / (ex + self.eps)))
        h_lmmse = tf.cast(alpha, h_ls.dtype) * h_ls
        return h_lmmse


class ResidualChannelEstimator(Layer):
    """LS 信道估计 + 神经残差修正。"""

    def __init__(self, hidden_dim: int = 64, base_estimator: Layer | None = None):
        super().__init__()
        self.base_estimator = base_estimator if base_estimator is not None else LSChannelEstimator()
        self.norm = LayerNormalization(axis=-1)
        self.d1 = Dense(hidden_dim, activation="relu")
        self.d2 = Dense(hidden_dim, activation="relu")
        self.out = Dense(2, activation=None)

    def call(self, y_pilot: tf.Tensor, x_pilot: tf.Tensor, no: tf.Tensor | None = None):
        if isinstance(self.base_estimator, LMMSEDFTChannelEstimator):
            h_ls = self.base_estimator(y_pilot, x_pilot, no=no)
        else:
            h_ls = self.base_estimator(y_pilot, x_pilot)
        feat = _complex_to_features(h_ls)  # [B,1,4]
        feat = tf.squeeze(feat, axis=1)
        feat = self.norm(feat)
        x = self.d1(feat)
        x = self.d2(x)
        delta = self.out(x)
        delta_c = tf.complex(delta[:, :1], delta[:, 1:2])
        h_refined = h_ls + delta_c
        return h_ls, h_refined


class MMSEEqualizer(Layer):
    """单抽头 MMSE/零迫均衡。"""

    def __init__(self, noise_floor: float = 1e-8):
        super().__init__()
        self.noise_floor = noise_floor

    def call(self, y: tf.Tensor, h_hat: tf.Tensor, no: tf.Tensor | None = None):
        """
        y: [B,S]
        h_hat: [B,1]
        no: [B,1] 或标量，可选
        """
        if no is None:
            # 零迫
            x_hat = y / (h_hat + tf.cast(self.noise_floor, h_hat.dtype))
            return x_hat

        no = tf.cast(no, tf.float32)
        if no.shape.rank == 0:
            no = tf.fill([tf.shape(y)[0], 1], no)
        elif no.shape.rank == 1:
            no = tf.reshape(no, [-1, 1])

        h_abs2 = tf.abs(h_hat) ** 2
        den = tf.cast(h_abs2 + no, h_hat.dtype)
        w = tf.math.conj(h_hat) / den
        return y * w


class IterativeMMSEEqualizer(Layer):
    """轻量迭代均衡（decision-directed），用于替代单次 MMSE。"""

    def __init__(self, num_iters: int = 2, dd_alpha: float = 0.7):
        super().__init__()
        self.num_iters = int(max(1, num_iters))
        self.dd_alpha = float(dd_alpha)
        self.mmse = MMSEEqualizer()

    @staticmethod
    def _qpsk_hard(x: tf.Tensor) -> tf.Tensor:
        r = tf.where(tf.math.real(x) >= 0.0, 1.0, -1.0)
        i = tf.where(tf.math.imag(x) >= 0.0, 1.0, -1.0)
        return tf.complex(r, i) / tf.cast(tf.sqrt(tf.constant(2.0, tf.float32)), tf.complex64)

    def call(self, y: tf.Tensor, h_hat: tf.Tensor, no: tf.Tensor | None = None):
        h_cur = h_hat
        x = self.mmse(y, h_cur, no=no)
        for _ in range(self.num_iters - 1):
            x_dec = self._qpsk_hard(x)
            num = tf.reduce_sum(y * tf.math.conj(x_dec), axis=1, keepdims=True)
            den = tf.reduce_sum(tf.abs(x_dec) ** 2, axis=1, keepdims=True)
            h_dd = num / tf.cast(den + 1e-8, num.dtype)
            h_cur = self.dd_alpha * h_cur + (1.0 - self.dd_alpha) * h_dd
            x = self.mmse(y, h_cur, no=no)
        return x, h_cur


class ResidualEqualizer(Layer):
    """MMSE 均衡 + 神经残差修正。"""

    def __init__(self, hidden_dim: int = 64, base_equalizer: Layer | None = None):
        super().__init__()
        self.base_equalizer = base_equalizer if base_equalizer is not None else MMSEEqualizer()
        self.norm = LayerNormalization(axis=-1)
        self.d1 = Dense(hidden_dim, activation="relu")
        self.d2 = Dense(hidden_dim, activation="relu")
        self.out = Dense(2, activation=None)

    def call(self, y: tf.Tensor, h_hat: tf.Tensor, no: tf.Tensor | None = None):
        if isinstance(self.base_equalizer, IterativeMMSEEqualizer):
            x_mmse, _ = self.base_equalizer(y, h_hat, no=no)
        else:
            x_mmse = self.base_equalizer(y, h_hat, no=no)
        feat = _complex_to_features(x_mmse)
        feat = self.norm(feat)
        delta = self.out(self.d2(self.d1(feat)))
        delta_c = tf.complex(delta[..., :1], delta[..., 1:2])
        x_refined = x_mmse + tf.squeeze(delta_c, axis=-1)
        return x_mmse, x_refined


class USRPReceiverFrontEnd(Model):
    """可直接替换到 USRP 后端的接收前端。"""

    def __init__(
        self,
        use_residual_ce: bool = True,
        use_residual_eq: bool = True,
        estimator_type: str = "ls",
        equalizer_type: str = "mmse",
        equalizer_iters: int = 1,
        dft_taps: int = 4,
    ):
        super().__init__()
        self.sync = PilotSynchronizer()
        if estimator_type == "lmmse_dft":
            base_ce = LMMSEDFTChannelEstimator(dft_taps=dft_taps)
        else:
            base_ce = LSChannelEstimator()

        if equalizer_type == "iterative":
            base_eq = IterativeMMSEEqualizer(num_iters=equalizer_iters)
        else:
            base_eq = MMSEEqualizer()

        self.ce = ResidualChannelEstimator(base_estimator=base_ce) if use_residual_ce else base_ce
        self.eq = ResidualEqualizer(base_equalizer=base_eq) if use_residual_eq else base_eq

    def call(self, y_pilot: tf.Tensor, x_pilot: tf.Tensor, y_data: tf.Tensor, no: tf.Tensor | None = None):
        y_pilot_corr, y_data_corr, phase_err = self.sync(y_pilot, x_pilot, y_data)

        if isinstance(self.ce, ResidualChannelEstimator):
            h_ls, h_hat = self.ce(y_pilot_corr, x_pilot, no=no)
        elif isinstance(self.ce, LMMSEDFTChannelEstimator):
            h_hat = self.ce(y_pilot_corr, x_pilot, no=no)
            h_ls = h_hat
        else:
            h_hat = self.ce(y_pilot_corr, x_pilot)
            h_ls = h_hat

        if isinstance(self.eq, ResidualEqualizer):
            x_mmse, x_hat = self.eq(y_data_corr, h_hat, no=no)
        elif isinstance(self.eq, IterativeMMSEEqualizer):
            x_mmse, _ = self.eq(y_data_corr, h_hat, no=no)
            x_hat = x_mmse
        else:
            x_hat = self.eq(y_data_corr, h_hat, no=no)
            x_mmse = x_hat

        return {
            "phase_err": phase_err,
            "h_ls": h_ls,
            "h_hat": h_hat,
            "x_mmse": x_mmse,
            "x_hat": x_hat,
            "y_pilot_corr": y_pilot_corr,
            "y_data_corr": y_data_corr,
        }


def synthetic_burst(batch_size: int = 8, num_pilot: int = 16, num_data: int = 128, snr_db: float = 10.0):
    """快速离线仿真，便于在没有 USRP 时预训练/联调。"""
    x_pilot = tf.complex(
        tf.ones([batch_size, num_pilot], tf.float32),
        tf.zeros([batch_size, num_pilot], tf.float32),
    )
    x_data = tf.complex(
        tf.random.normal([batch_size, num_data]),
        tf.random.normal([batch_size, num_data]),
    )
    h = tf.complex(
        tf.random.normal([batch_size, 1], stddev=0.5),
        tf.random.normal([batch_size, 1], stddev=0.5),
    )
    phase = tf.random.uniform([batch_size, 1], minval=-0.5, maxval=0.5)
    rot = tf.exp(tf.complex(tf.zeros_like(phase), phase))
    x_all = tf.concat([x_pilot, x_data], axis=1)
    y = h * x_all * rot
    no = tf.ones([batch_size, 1], tf.float32) * tf.pow(10.0, -snr_db / 10.0)
    noise = tf.complex(
        tf.random.normal(tf.shape(y), stddev=tf.sqrt(no[:, :1] / 2.0)),
        tf.random.normal(tf.shape(y), stddev=tf.sqrt(no[:, :1] / 2.0)),
    )
    y = y + noise
    return x_pilot, x_data, y[:, :num_pilot], y[:, num_pilot:], no


if __name__ == "__main__":
    x_pilot, x_data, y_pilot, y_data, no = synthetic_burst()
    rx = USRPReceiverFrontEnd(use_residual_ce=True, use_residual_eq=True)
    out = rx(y_pilot, x_pilot, y_data, no=no)
    print("phase_err:", out["phase_err"].shape)
    print("h_hat:", out["h_hat"].shape)
    print("x_hat:", out["x_hat"].shape)
