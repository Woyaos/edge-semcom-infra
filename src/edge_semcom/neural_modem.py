"""
神经网络调制/解调器，用于逐步替换Sionna的物理层

阶段1：替换 QAM 调制/解调，保留 LDPC 编码/解码
阶段2：替换 LDPC，保留量化
"""

import tensorflow as tf
try:
    import torch  # 可选依赖
    import torch.nn as nn  # 可选依赖
except ImportError:
    torch = None
    nn = None
import numpy as np
from tensorflow.keras.layers import (
    Layer,
    Dense,
    Reshape,
    Flatten,
    Conv1D,
    LayerNormalization,
    Dropout,
)
from tensorflow.keras import Model


##======================================== 第一阶段：神经调制/解调 ================================================##

class ResidualMLPBlock(Layer):
    """预归一化残差MLP块（ResNet思想，He et al., 2016）"""
    def __init__(self, hidden_dim, expansion=4, dropout=0.0):
        super().__init__()
        self.norm = LayerNormalization(epsilon=1e-6)
        self.fc1 = Dense(hidden_dim * expansion, activation="gelu")
        self.drop1 = Dropout(dropout)
        self.fc2 = Dense(hidden_dim, activation=None)
        self.drop2 = Dropout(dropout)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=None):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.drop1(h, training=training)
        h = self.fc2(h)
        h = self.drop2(h, training=training)
        return x + h


class ResidualContextBlock(Layer):
    """预归一化残差上下文块（TCN/Conv思想，Bai et al., 2018）"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.norm = LayerNormalization(epsilon=1e-6)
        self.conv1 = Conv1D(hidden_dim, kernel_size=5, padding="same", activation="gelu")
        self.conv2 = Conv1D(hidden_dim, kernel_size=1, padding="same", activation=None)
        self.drop = Dropout(dropout)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=None):
        h = self.norm(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.drop(h, training=training)
        return x + h


class GatedDilatedTCNBlock(Layer):
    """更现代的门控空洞卷积块，用于长序列上下文建模。"""
    def __init__(self, hidden_dim, dilation_rate=1, dropout=0.0):
        super().__init__()
        self.norm = LayerNormalization(epsilon=1e-6)
        self.conv = Conv1D(
            hidden_dim * 2,
            kernel_size=3,
            padding="same",
            dilation_rate=dilation_rate,
            activation=None,
        )
        self.proj = Dense(hidden_dim, activation=None)
        self.drop = Dropout(dropout)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=None):
        h = self.norm(x)
        h = self.conv(h)
        h_val, h_gate = tf.split(h, num_or_size_splits=2, axis=-1)
        h = h_val * tf.sigmoid(h_gate)
        h = self.proj(h)
        h = self.drop(h, training=training)
        return x + h


class NeuralModulator(Layer):
    """
    用神经网络替代 QAM 调制
    输入: [batch, num_symbols_per_codeword, num_bits_per_symbol]
    输出: [batch, num_symbols_per_codeword] (复数符号)
    """
    def __init__(
        self,
        num_bits_per_symbol=6,
        num_symbols_per_codeword=250,
        hidden_dim=64,
        model_variant="literature_v2",
        num_blocks=3,
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_symbols_per_codeword = num_symbols_per_codeword
        self.model_variant = model_variant
        self.hidden_dim = hidden_dim
        
        # legacy: 兼容旧模型
        self.legacy_dense1 = Dense(hidden_dim, activation="relu")
        self.legacy_dense2 = Dense(hidden_dim, activation="relu")
        # backward compatibility: 旧checkpoint键名
        self.dense1 = self.legacy_dense1
        self.dense2 = self.legacy_dense2

        # literature_v2/v3: 残差上下文结构
        self.in_proj = Dense(hidden_dim, activation=None)
        self.res_blocks = [ResidualMLPBlock(hidden_dim, expansion=4, dropout=dropout) for _ in range(num_blocks)]
        self.context_block = ResidualContextBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.tcn_blocks = [GatedDilatedTCNBlock(hidden_dim, dilation_rate=2 ** i, dropout=dropout) for i in range(num_blocks)]
        self.out_norm = LayerNormalization(epsilon=1e-6)
        self.dense_out = Dense(2, activation=None)

    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, bits, training=None):
        """
        bits: [batch, num_symbols, num_bits_per_symbol]
        """
        if self.model_variant in ["legacy_mlp", "hybrid_residual"]:
            x = self.legacy_dense1(bits)
            x = self.legacy_dense2(x)
        elif self.model_variant == "literature_v3":
            x = self.in_proj(bits)
            for blk in self.tcn_blocks:
                x = blk(x, training=training)
            x = self.out_norm(x)
        else:
            x = self.in_proj(bits)
            for blk in self.res_blocks:
                x = blk(x, training=training)
            x = self.context_block(x, training=training)
            x = self.out_norm(x)

        x = self.dense_out(x)  # [batch, num_symbols, 2]
        
        # 转换为复数：x[..., 0] + 1j * x[..., 1]
        # 并进行功率归一化
        real = x[..., 0]
        imag = x[..., 1]
        symbols = tf.cast(real, tf.complex64) + 1j * tf.cast(imag, tf.complex64)
        
        # 功率归一化
        power = tf.reduce_mean(tf.abs(symbols) ** 2)
        norm = tf.cast(tf.sqrt(power + 1e-8), symbols.dtype)
        symbols = symbols / norm
        
        return symbols


class NeuralDemodulator(Layer):
    """
    用神经网络替代 QAM 解调 + 神经判决反馈
    输入: [batch, num_symbols_per_codeword] (复数符号) + noise_power
    输出: [batch, num_symbols_per_codeword, num_bits_per_symbol] (LLR)
    """
    def __init__(
        self,
        num_bits_per_symbol=6,
        num_symbols_per_codeword=250,
        hidden_dim=128,
        model_variant="literature_v2",
        num_blocks=4,
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_symbols_per_codeword = num_symbols_per_codeword
        self.model_variant = model_variant
        self.hidden_dim = hidden_dim
        
        # legacy: 兼容旧模型
        self.legacy_dense1 = Dense(hidden_dim, activation="relu")
        self.legacy_dense2 = Dense(hidden_dim, activation="relu")
        self.legacy_dense3 = Dense(hidden_dim, activation="relu")
        # backward compatibility: 旧checkpoint键名
        self.dense1 = self.legacy_dense1
        self.dense2 = self.legacy_dense2
        self.dense3 = self.legacy_dense3

        # literature_v2/v3: FiLM + 残差 + 长程上下文
        self.in_proj = Dense(hidden_dim, activation=None)
        self.res_blocks = [ResidualMLPBlock(hidden_dim, expansion=4, dropout=dropout) for _ in range(num_blocks)]
        self.context_block = ResidualContextBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.tcn_blocks = [GatedDilatedTCNBlock(hidden_dim, dilation_rate=2 ** i, dropout=dropout) for i in range(num_blocks)]
        self.out_norm = LayerNormalization(epsilon=1e-6)

        # 噪声条件分支（FiLM，Perez et al., 2018）
        self.cond_dense = Dense(hidden_dim, activation="gelu")
        self.cond_gamma = Dense(hidden_dim, activation=None)
        self.cond_beta = Dense(hidden_dim, activation=None)

        self.llr_out = Dense(num_bits_per_symbol, activation=None)

        # hybrid_residual：在legacy输出上叠加残差（初始化为0，不劣化起步）
        self.hy_in_proj = Dense(hidden_dim, activation=None)
        self.hy_res_blocks = [ResidualMLPBlock(hidden_dim, expansion=4, dropout=dropout) for _ in range(num_blocks)]
        self.hy_out_norm = LayerNormalization(epsilon=1e-6)
        self.hy_llr_delta = Dense(num_bits_per_symbol, activation=None)
        self.hy_delta_scale = self.add_weight(
            name="hy_delta_scale",
            shape=(),
            initializer="zeros",
            trainable=True,
        )

    def build(self, input_shape):
        super().build(input_shape)

    def _broadcast_noise_power(self, noise_power, batch_size):
        noise_power = tf.cast(noise_power, tf.float32)
        rank = noise_power.shape.rank
        if rank == 0:
            return tf.fill([batch_size, 1], noise_power)
        if rank == 1:
            noise_power = tf.reshape(noise_power, [-1, 1])
            return tf.cond(
                tf.equal(tf.shape(noise_power)[0], 1),
                lambda: tf.tile(noise_power, [batch_size, 1]),
                lambda: noise_power,
            )
        noise_power = tf.reshape(noise_power, [tf.shape(noise_power)[0], -1])[:, :1]
        return tf.cond(
            tf.equal(tf.shape(noise_power)[0], 1),
            lambda: tf.tile(noise_power, [batch_size, 1]),
            lambda: noise_power,
        )
    
    def call(self, y_complex, noise_power, training=None):
        """
        y_complex: [batch, num_symbols] (复数接收信号)
        noise_power: [batch, 1] 或 标量 (噪声功率)
        """
        batch_size = tf.shape(y_complex)[0]
        
        # 将复数信号分解为实部和虚部
        real = tf.math.real(y_complex)
        imag = tf.math.imag(y_complex)
        magnitude = tf.abs(y_complex)
        phase = tf.math.angle(y_complex)
        
        # 广播噪声功率到 [batch, 1]
        noise_power = self._broadcast_noise_power(noise_power, batch_size)
        
        noise_db = 10.0 * tf.math.log(noise_power + 1e-8) / tf.math.log(10.0)
        noise_db = tf.tile(noise_db, [1, self.num_symbols_per_codeword])
        
        # 拼接特征：实部、虚部、幅度、相位、噪声功率
        z = tf.stack([real, imag, magnitude, phase, noise_db], axis=2)
        
        if self.model_variant == "legacy_mlp":
            x = self.legacy_dense1(z)
            x = self.legacy_dense2(x)
            x = self.legacy_dense3(x)
        elif self.model_variant == "hybrid_residual":
            # 1) legacy主干
            xb = self.legacy_dense1(z)
            xb = self.legacy_dense2(xb)
            xb = self.legacy_dense3(xb)
            base_llr = self.llr_out(xb)

            # 2) residual修正分支
            xr = self.hy_in_proj(z)
            c = self.cond_dense(noise_db[:, :1])
            gamma = self.cond_gamma(c)[:, tf.newaxis, :]
            beta = self.cond_beta(c)[:, tf.newaxis, :]
            xr = xr * (1.0 + gamma) + beta
            for blk in self.hy_res_blocks:
                xr = blk(xr, training=training)
            xr = self.hy_out_norm(xr)
            llr_delta = self.hy_llr_delta(xr)

            scale = tf.tanh(self.hy_delta_scale)
            llr = base_llr + scale * llr_delta
            return llr
        elif self.model_variant == "literature_v3":
            x = self.in_proj(z)

            # FiLM噪声条件调制
            c = self.cond_dense(noise_db[:, :1])
            gamma = self.cond_gamma(c)[:, tf.newaxis, :]
            beta = self.cond_beta(c)[:, tf.newaxis, :]
            x = x * (1.0 + gamma) + beta

            for blk in self.tcn_blocks:
                x = blk(x, training=training)
            x = self.out_norm(x)
        else:
            x = self.in_proj(z)

            # FiLM噪声条件调制
            c = self.cond_dense(noise_db[:, :1])
            gamma = self.cond_gamma(c)[:, tf.newaxis, :]
            beta = self.cond_beta(c)[:, tf.newaxis, :]
            x = x * (1.0 + gamma) + beta

            for blk in self.res_blocks:
                x = blk(x, training=training)
            x = self.context_block(x, training=training)
            x = self.out_norm(x)

        llr = self.llr_out(x)  # [batch, num_symbols, num_bits_per_symbol]
        
        return llr


##======================================== 第二阶段：神经编码/解码 ================================================##

class NeuralEncoder(Layer):
    """
    用神经网络替代 LDPC 编码
    输入: [batch, k] (信息比特)
    输出: [batch, n] (编码比特)
    
    使用卷积和全连接的组合来实现编码
    """
    def __init__(self, k=7500, n=15000, code_rate=0.5, hidden_dim=512):
        super().__init__()
        self.k = k
        self.n = n
        self.code_rate = code_rate
        
        # 卷积编码部分
        self.conv1 = Conv1D(64, kernel_size=3, padding='same', activation='relu')
        self.conv2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')
        self.conv3 = Conv1D(32, kernel_size=3, padding='same', activation='relu')
        
        # 全连接部分用于最终编码
        # 将 [batch, 32, k] reshape 并处理
        self.flatten = Flatten()
        self.dense1 = Dense(hidden_dim, activation='relu')
        self.dense2 = Dense(hidden_dim, activation='relu')
        self.dense_out = Dense(n, activation='sigmoid')
    
    def call(self, bits, training=None):
        """
        bits: [batch, k]
        返回: [batch, n]
        """
        # 添加通道维度用于卷积
        x = tf.expand_dims(bits, axis=-1)  # [batch, k, 1]
        
        # 卷积处理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # [batch, k, 32]
        
        # 平展并处理
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        encoded = self.dense_out(x)  # [batch, n], 值域(0,1)

        # 训练阶段保留软输出以保证梯度传播；推理阶段再硬判决
        if training:
            return encoded
        return tf.cast(tf.round(encoded), tf.float32)


class NeuralDecoder(Layer):
    """
    用神经网络替代 LDPC 解码
    输入: [batch, n] (接收的编码比特，通常是LLR)
    输出: [batch, k] (恢复的信息比特)
    
    使用迭代解码思想
    """
    def __init__(self, k=7500, n=15000, code_rate=0.5, hidden_dim=512, num_iterations=5):
        super().__init__()
        self.k = k
        self.n = n
        self.code_rate = code_rate
        self.num_iterations = num_iterations
        
        # 迭代解码的处理块
        self.iter_blocks = [
            self._build_iter_block(hidden_dim)
            for _ in range(num_iterations)
        ]
    
    def _build_iter_block(self, hidden_dim):
        """构建单个迭代解码块"""
        class IterBlock(Layer):
            def __init__(self, hidden_dim, n):
                super().__init__()
                self.dense1 = Dense(hidden_dim, activation='relu')
                self.dense2 = Dense(hidden_dim, activation='relu')
                self.dense_out = Dense(n, activation=None)
            
            def call(self, x):
                x = self.dense1(x)
                x = self.dense2(x)
                x = self.dense_out(x)
                return x
        
        return IterBlock(hidden_dim, self.n)
    
    def call(self, llr, training=None):
        """
        llr: [batch, n] (对数似然比)
        返回: [batch, k]
        """
        batch_size = tf.shape(llr)[0]
        
        # 初始化为接收信号
        x = llr
        
        # 迭代解码
        for iter_block in self.iter_blocks:
            x = iter_block(x)
        
        # 提取信息比特（前k个比特）
        logits = x[:, :self.k]

        # 训练阶段返回logits用于BCEWithLogits；推理阶段再硬判决
        if training:
            return logits
        decoded_bits = tf.cast(tf.round(tf.nn.sigmoid(logits)), tf.float32)
        return decoded_bits


##======================================== 完整的神经收发机模型 ================================================##

class NeuralPhyLayer(Model):
    """
    完整的神经物理层模型，支持阶段式替换
    
    mode选项：
    - 'traditional': 保留Sionna (LDPC + QAM + Neural Demod)
    - 'neural_modem': 替换调制/解调 (LDPC + Neural Modem)
    - 'neural_codec': 替换编码/解码 (Neural Encoder/Decoder + QAM)
    - 'full_neural': 全替换 (Neural Encoder/Decoder + Neural Modem)
    """
    def __init__(self, mode='neural_modem', k=7500, n=15000, 
                 num_bits_per_symbol=6, num_symbols_per_codeword=250):
        super().__init__()
        
        self.mode = mode
        self.k = k
        self.n = n
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_symbols_per_codeword = num_symbols_per_codeword
        
        if mode in ['neural_modem', 'full_neural']:
            self.neural_modulator = NeuralModulator(
                num_bits_per_symbol, 
                num_symbols_per_codeword
            )
            self.neural_demodulator = NeuralDemodulator(
                num_bits_per_symbol,
                num_symbols_per_codeword
            )
        
        if mode in ['neural_codec', 'full_neural']:
            self.neural_encoder = NeuralEncoder(k, n)
            self.neural_decoder = NeuralDecoder(k, n, num_iterations=5)
    
    def get_config(self):
        return {
            'mode': self.mode,
            'k': self.k,
            'n': self.n,
            'num_bits_per_symbol': self.num_bits_per_symbol,
            'num_symbols_per_codeword': self.num_symbols_per_codeword
        }
    
    def get_mode(self):
        return self.mode


# 辅助函数：创建不同模式的物理层
def create_phy_layer(mode='neural_modem', **kwargs):
    """
    工厂函数创建不同模式的物理层
    
    示例:
        phy = create_phy_layer('neural_modem')
    """
    return NeuralPhyLayer(mode=mode, **kwargs)


if __name__ == "__main__":
    # 简单测试
    print("创建神经调制器...")
    modulator = NeuralModulator(num_bits_per_symbol=6, num_symbols_per_codeword=250)
    
    # 测试输入：[batch=2, num_symbols=250, num_bits=6]
    test_bits = tf.random.uniform((2, 250, 6), minval=0, maxval=2, dtype=tf.float32)
    symbols = modulator(test_bits)
    print(f"调制输出shape: {symbols.shape}, dtype: {symbols.dtype}")
    
    print("\n创建神经解调器...")
    demodulator = NeuralDemodulator(num_bits_per_symbol=6, num_symbols_per_codeword=250)
    noise_power = tf.constant([0.1, 0.15], dtype=tf.float32)
    llr = demodulator(symbols, noise_power)
    print(f"解调输出shape: {llr.shape}, dtype: {llr.dtype}")
    
    print("\n创建完整物理层...")
    phy = create_phy_layer('neural_modem')
    print(f"物理层模式: {phy.get_mode()}")
    print("✓ 神经物理层模块创建成功")

