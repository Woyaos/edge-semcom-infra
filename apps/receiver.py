##=======================================GPU Configuration and Imports================================================##
import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense

from sionna.phy import Block
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no, log10, expand_to_rank
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource
from sionna.phy.utils import sim_ber

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation

# Import neural modem components
try:
    from edge_semcom.neural_modem import NeuralDemodulator
    NEURAL_MODEM_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Failed to import edge_semcom.neural_modem: {e}")
    NEURAL_MODEM_AVAILABLE = False
    NeuralDemodulator = None

try:
    from edge_semcom.usrp_frontend import USRPReceiverFrontEnd
    USRP_FRONTEND_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Failed to import edge_semcom.usrp_frontend: {e}")
    USRP_FRONTEND_AVAILABLE = False
    USRPReceiverFrontEnd = None

import matplotlib.pyplot as plt
import numpy as np
import pickle
import socket
import tensorflow as tf
import cv2
import json
import time
import torch
from diffusers import AutoencoderKL
import re
import torch.nn as nn
import sys
import subprocess
import importlib

# TensorRT imports (可选)
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

##=======================================Simulation Parameters========================================================##

###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 5.5
ebno_db_max = 6.0

###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6 # Baseline is 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 0.5 # Coderate for the outer code
n = 15000 # Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword

###############################################
# Training configuration
###############################################
num_training_iterations_conventional = 10000 # Number of training iterations for conventional training
# Number of training iterations with RL-based training for the alternating training phase and fine-tuning of the receiver phase
num_training_iterations_rl_alt = 7000
num_training_iterations_rl_finetuning = 3000
training_batch_size = tf.constant(128, tf.int32) # Training batch size
rl_perturbation_var = 0.01 # Variance of the perturbation used for RL-based training of the transmitter
model_weights_path_conventional_training_tx = "awgn_autoencoder_weights_conventional_training_tx" # Filename to save the autoencoder weights once conventional training is done
model_weights_path_conventional_training_rx = "awgn_autoencoder_weights_conventional_training_rx" # Filename to save the autoencoder weights once conventional training is done

###############################################
# Evaluation configuration
###############################################
results_filename = "awgn_autoencoder_results" # Location to save the results

##=======================================Neural Demapper==============================================================##

class NeuralDemapper(Layer):

    def __init__(self):
        super().__init__()

        self._dense_1 = Dense(128, 'relu')
        self._dense_2 = Dense(128, 'relu')
        self._dense_3 = Dense(num_bits_per_symbol, None) # The feature correspond to the LLRs for every bits carried by a symbol

    def call(self, y, no):

        # Using log10 scale helps with the performance
        no_db = log10(no)

        # Stacking the real and imaginary components of the complex received samples
        # and the noise variance
        no_db = tf.tile(no_db, [1, num_symbols_per_codeword]) # [batch size, num_symbols_per_codeword]
        z = tf.stack([tf.math.real(y),
                      tf.math.imag(y),
                      no_db], axis=2) # [batch size, num_symbols_per_codeword, 3]
        llr = self._dense_1(z)
        llr = self._dense_2(llr)
        llr = self._dense_3(llr) # [batch size, num_symbols_per_codeword, num_bits_per_symbol]

        return llr



class RXSystemConventionalTraining(Model):

    def __init__(self, training):
        super().__init__()

        self._training = training

        ################
        ## Receiver
        ################
        # We use the previously defined neural network for demapping
        self._demapper = NeuralDemapper()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not self._training:
            self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    def call(self, batch_size, y, ebno_db):
        # Set the constellation points equal to a complex tensor constructed
        # from two real-valued variables

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)

        ################
        ## Receiver
        ################
        llr = self._demapper(y, no)
        llr = tf.reshape(llr, [batch_size, n])
        # If training, outer decoding is not performed and the BCE is returned
        if self._training:
            return llr
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation


class RXSystemMinimalReplace(Model):
    """工业化最小侵入：仅替换 demapper，可在 neural/app 间切换。"""

    def __init__(self, demapper_backend: str = "neural"):
        super().__init__()
        self._backend = demapper_backend
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

        if demapper_backend == "neural":
            self._demapper = NeuralDemapper()
            self._constellation = None
        elif demapper_backend == "app":
            self._constellation = Constellation("qam", num_bits_per_symbol, trainable=False)
            self._demapper = Demapper("app", constellation=self._constellation)
        else:
            raise ValueError(f"不支持的 demapper_backend: {demapper_backend}")

    def call(self, batch_size, y, ebno_db):
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)

        llr = self._demapper(y, no)
        llr = tf.reshape(llr, [batch_size, n])
        b_hat = self._decoder(llr)
        return b_hat


class RXSystemNeuralModem(Model):
    """神经解调器 RX：使用 NeuralDemodulator 替代传统 APP demapper。"""

    def __init__(self, ckpt_dir: str = None, ckpt_path: str = None):
        super().__init__()
        
        if not NEURAL_MODEM_AVAILABLE:
            raise RuntimeError("[ERROR] NeuralDemodulator not available. Install neural_modem.py first.")
        
        self._demapper = NeuralDemodulator(
            num_bits_per_symbol=num_bits_per_symbol,
            num_symbols_per_codeword=num_symbols_per_codeword,
        )
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        
        # Dummy build for checkpoint restoration
        dummy_y = tf.complex(
            tf.zeros([1, num_symbols_per_codeword], dtype=tf.float32),
            tf.zeros([1, num_symbols_per_codeword], dtype=tf.float32)
        )
        dummy_no = tf.ones([1, 1], dtype=tf.float32)
        _ = self._demapper(dummy_y, dummy_no, training=False)
        
        # Load checkpoint
        # 训练脚本 train_neural_modem.py 中使用的键名是 demodulator
        ckpt = tf.train.Checkpoint(demodulator=self._demapper)
        restore_path = ckpt_path
        if not restore_path and ckpt_dir:
            restore_path = tf.train.latest_checkpoint(ckpt_dir)
        if restore_path:
            ckpt.restore(restore_path).expect_partial()
            print(f"[RX][NEURAL] 已加载神经解调器 checkpoint: {restore_path}")
        else:
            print("[RX][NEURAL] 未提供 checkpoint，使用随机初始化权重")
    
    def call(self, batch_size, y, ebno_db):
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)
        
        # Neural demapper
        # 注意：当前训练得到的 logits 已与本链路LDPC译码约定对齐，保持正向输入。
        llr = self._demapper(y, no, training=False)
        llr = tf.reshape(llr, [batch_size, n])
        
        # LDPC decoding
        b_hat = self._decoder(llr)
        return b_hat


# Utility function to load and set weights of a model
def load_weights_rx(model, model_weights_path):
    model(tf.cast(1, tf.int32),
          tf.complex(
            tf.random.normal([1, n//num_bits_per_symbol]),
            tf.random.normal([1, n//num_bits_per_symbol])),
          tf.constant([10.0], dtype=tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)


def maybe_load_neural_demapper_weights(model, model_weights_path):
    """仅当文件存在时加载神经 demapper 权重；用于最小侵入替换。"""
    if not model_weights_path:
        print("[RX] 未提供神经 demapper 权重路径，使用当前初始化参数")
        return
    if not os.path.isfile(model_weights_path):
        print(f"[RX] 神经 demapper 权重文件不存在，跳过加载: {model_weights_path}")
        return
    load_weights_rx(model, model_weights_path)
    print(f"[RX] 神经 demapper 权重已加载: {model_weights_path}")


def build_pilot_symbols(batch_size, pilot_len):
    """与TX一致的固定导频序列。"""
    pilot_len = int(pilot_len)
    if pilot_len <= 0:
        return None
    pilot = tf.complex(
        tf.ones([batch_size, pilot_len], dtype=tf.float32),
        tf.ones([batch_size, pilot_len], dtype=tf.float32),
    )
    pilot = pilot / tf.cast(tf.sqrt(tf.constant(2.0, dtype=tf.float32)), pilot.dtype)
    return pilot


def frontend_process_burst(y: tf.Tensor, pilot_len: int, mode: str, no: tf.Tensor | None = None):
    """对接 USRP 的前端预处理：同步/估计/均衡；默认返回原始 payload。"""
    pilot_len = int(pilot_len)
    mode = str(mode)
    if pilot_len <= 0:
        return y, {"pilot_len": pilot_len, "mode": mode}

    if mode == "traditional":
        if y.shape.rank != 2:
            raise ValueError(f"前端预处理只支持 [B,S] 张量，当前 shape={y.shape}")
        if y.shape[1] is not None and int(y.shape[1]) <= pilot_len:
            raise ValueError(f"接收序列长度不足以切分导频: shape={y.shape}, pilot_len={pilot_len}")
        return y[:, pilot_len:], {"pilot_len": pilot_len, "mode": mode}

    if y.shape.rank != 2:
        raise ValueError(f"前端预处理只支持 [B,S] 张量，当前 shape={y.shape}")
    if y.shape[1] is not None and int(y.shape[1]) <= pilot_len:
        raise ValueError(f"接收序列长度不足以切分导频: shape={y.shape}, pilot_len={pilot_len}")

    y_pilot = y[:, :pilot_len]
    y_data = y[:, pilot_len:]
    x_pilot = build_pilot_symbols(tf.shape(y)[0], pilot_len)

    if mode == "pilot_sync":
        if not USRP_FRONTEND_AVAILABLE:
            # 退化为简单导频相位校正
            phase_err = tf.math.angle(
                tf.reduce_sum(y_pilot * tf.math.conj(x_pilot), axis=1, keepdims=True) + tf.cast(1e-8, y.dtype)
            )
            rot = tf.exp(tf.complex(tf.zeros_like(phase_err), -phase_err))
            y_data = y_data * rot
            return y_data, {"pilot_len": pilot_len, "mode": mode, "phase_err": phase_err}

        frontend = USRPReceiverFrontEnd(use_residual_ce=False, use_residual_eq=False)
        out = frontend(y_pilot, x_pilot, y_data, no=no)
        return out["x_mmse"], {"pilot_len": pilot_len, "mode": mode, "phase_err": out["phase_err"], "h_hat": out["h_hat"]}

    if mode == "residual":
        if not USRP_FRONTEND_AVAILABLE:
            # 退化为传统 LS + MMSE 思路
            phase_err = tf.math.angle(
                tf.reduce_sum(y_pilot * tf.math.conj(x_pilot), axis=1, keepdims=True) + tf.cast(1e-8, y.dtype)
            )
            rot = tf.exp(tf.complex(tf.zeros_like(phase_err), -phase_err))
            y_pilot = y_pilot * rot
            y_data = y_data * rot
            h_hat = tf.reduce_sum(y_pilot * tf.math.conj(x_pilot), axis=1, keepdims=True) / (
                tf.reduce_sum(tf.abs(x_pilot) ** 2, axis=1, keepdims=True) + tf.cast(1e-8, y.dtype)
            )
            if no is None:
                x_hat = y_data / (h_hat + tf.cast(1e-8, h_hat.dtype))
            else:
                no_f = tf.cast(no, tf.float32)
                if no_f.shape.rank == 0:
                    no_f = tf.fill([tf.shape(y)[0], 1], no_f)
                elif no_f.shape.rank == 1:
                    no_f = tf.reshape(no_f, [-1, 1])
                w = tf.math.conj(h_hat) / (tf.abs(h_hat) ** 2 + no_f)
                x_hat = y_data * w
            return x_hat, {"pilot_len": pilot_len, "mode": mode, "phase_err": phase_err, "h_hat": h_hat}

        frontend = USRPReceiverFrontEnd(use_residual_ce=True, use_residual_eq=True)
        out = frontend(y_pilot, x_pilot, y_data, no=no)
        return out["x_hat"], {"pilot_len": pilot_len, "mode": mode, "phase_err": out["phase_err"], "h_hat": out["h_hat"]}

    raise ValueError(f"不支持的 frontend_mode: {mode}")


def recvall(sock, n):
    """
    保证接收 n 个字节，TCP 接收可能不一次返回全部字节
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket 提前关闭")
        buf.extend(chunk)
    return bytes(buf)


def receive_tensor(host: str, port: int, num_packets=6):
    """
    监听 host:port，接收发来的 Tensor
    返回 tf.Tensor
    """
    """
    连续接收多个 Tensor
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print(f"监听 {host}:{port}，等待连接...")

        for i in range(num_packets):
            conn, addr = srv.accept()
            with conn:
                print(f"\n[连接 {i+1}] 来自 {addr}")
                size_b = recvall(conn, 8)
                size = int.from_bytes(size_b, byteorder='big')
                data = recvall(conn, size)
                tensor = tf.io.parse_tensor(data, out_type=tf.complex64)
                print(f"接收第 {i+1} 个 Tensor，shape={tensor.shape}")

                yield tensor  # 通过 yield 返回多个 tensor


def receive_tensor_stream(host: str, port: int):
    """持续监听 host:port，以长连接方式接收多帧；每帧格式为头部(H)+segments个张量(T)。"""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(1)
            print(f"监听 {host}:{port}，实时接收...")
            conn, addr = srv.accept()
            print(f"连接建立：{addr}，进入长连接模式")
            try:
                with conn:
                    while True:
                        marker = conn.recv(1)
                        if not marker:
                            print("对端关闭连接，等待重连...")
                            break
                        if marker != b'H':
                            print("收到未知标记，跳过该帧。")
                            continue
                        size_b = recvall(conn, 8)
                        size = int.from_bytes(size_b, byteorder='big')
                        hdr_bytes = recvall(conn, size)
                        try:
                            header = json.loads(hdr_bytes.decode('utf-8'))
                        except Exception as e:
                            print("头部解析失败：", e)
                            continue
                        segments = int(header.get("segments", 1))
                        tensors = []
                        for _ in range(segments):
                            m2 = recvall(conn, 1)
                            if m2 != b'T':
                                print("缺少张量标记T，提前结束该帧。")
                                break
                            szb = recvall(conn, 8)
                            sz = int.from_bytes(szb, byteorder='big')
                            data = recvall(conn, sz)
                            tensor = tf.io.parse_tensor(data, out_type=tf.complex64)
                            tensors.append(tensor)
                        if tensors:
                            print(f"收到一帧来自 {addr}：segments={len(tensors)}")
                            yield (header, tensors)
            except Exception as e:
                print(f"连接异常，重建监听：{e}")
                time.sleep(0.2)
                continue


def print_device_report_rx(model_rx):
    """统一打印 RX 侧 TensorFlow/PyTorch 设备与关键模块执行设备。"""
    tf_gpus = tf.config.list_physical_devices('GPU')
    print(f"[Device][RX] TensorFlow GPUs: {[d.name for d in tf_gpus] if tf_gpus else 'None'}")
    print(f"[Device][RX] PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Device][RX] PyTorch CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        y_dummy = tf.complex(tf.random.normal([1, n//num_bits_per_symbol]),
                             tf.random.normal([1, n//num_bits_per_symbol]))
        ebno_dummy = tf.constant([10.0], dtype=tf.float32)
        b_dummy = model_rx(batch_size=tf.cast(1, tf.int32), y=y_dummy, ebno_db=ebno_dummy)
        print(f"[Device][RX] Demap+LDPC output device: {b_dummy.device}")
    except Exception as e:
        print(f"[Device][RX] 模块设备探测失败: {e}")


class VAEDecodeWrapper(nn.Module):
    """用于 Torch-TensorRT 编译的 VAE 解码封装。"""
    def __init__(self, vae_model, vae_scale):
        super().__init__()
        self.vae = vae_model
        self.vae_scale = float(vae_scale)

    def forward(self, z):
        out = self.vae.decode(z / self.vae_scale).sample
        return out


class TRTVAEDecoder:
    """使用 TensorRT engine 执行 VAE decoder 推理。"""
    def __init__(self, engine_path):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("未安装 TensorRT Python 依赖（tensorrt）")
        if not torch.cuda.is_available():
            raise RuntimeError("当前 PyTorch CUDA 不可用，无法运行 TRT engine")
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TRT engine 不存在: {engine_path}")

        self.device = torch.device("cuda")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"反序列化 TRT engine 失败: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("创建 TRT 执行上下文失败")

        self.use_io_api = hasattr(self.engine, "num_io_tensors")
        self.input_idx = None
        self.output_idx = None
        self.input_name = None
        self.output_name = None

        if self.use_io_api:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_name = name
            if self.input_name is None or self.output_name is None:
                raise RuntimeError("TRT engine 未找到输入/输出 tensor")

            in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
            if -1 in in_shape:
                in_shape = (1, 4, 64, 64)
                self.context.set_input_shape(self.input_name, in_shape)
            out_shape = tuple(self.context.get_tensor_shape(self.output_name))
            if -1 in out_shape:
                out_shape = (1, 3, 512, 512)

            self.in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
            self.out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        else:
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.input_idx = i
                else:
                    self.output_idx = i
            if self.input_idx is None or self.output_idx is None:
                raise RuntimeError("TRT engine 未找到输入/输出 binding")

            in_shape = tuple(self.engine.get_binding_shape(self.input_idx))
            if -1 in in_shape:
                in_shape = (1, 4, 64, 64)
                self.context.set_binding_shape(self.input_idx, in_shape)
            out_shape = tuple(self.context.get_binding_shape(self.output_idx))
            if -1 in out_shape:
                out_shape = (1, 3, 512, 512)

            self.in_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
            self.out_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))

        self.in_shape = tuple(int(v) for v in in_shape)
        self.out_shape = tuple(int(v) for v in out_shape)

        def _np2torch(np_dtype):
            if np_dtype == np.float16:
                return torch.float16
            if np_dtype == np.float32:
                return torch.float32
            if np_dtype == np.int32:
                return torch.int32
            return torch.float32

        self.in_torch_dtype = _np2torch(self.in_dtype)
        self.out_torch_dtype = _np2torch(self.out_dtype)
        self.d_input = torch.empty(self.in_shape, device=self.device, dtype=self.in_torch_dtype)
        self.d_output = torch.empty(self.out_shape, device=self.device, dtype=self.out_torch_dtype)

    def decode(self, z_np: np.ndarray) -> np.ndarray:
        if z_np.shape != self.in_shape:
            raise ValueError(f"TRT decoder 输入 shape 不匹配: got={z_np.shape}, expect={self.in_shape}")

        z_t = torch.from_numpy(np.ascontiguousarray(z_np)).to(self.device, dtype=self.in_torch_dtype, non_blocking=True)
        self.d_input.copy_(z_t)
        stream = torch.cuda.current_stream(self.device)

        if self.use_io_api:
            self.context.set_input_shape(self.input_name, tuple(self.in_shape))
            self.context.set_tensor_address(self.input_name, int(self.d_input.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(self.d_output.data_ptr()))
            ok = self.context.execute_async_v3(stream_handle=int(stream.cuda_stream))
        else:
            bindings = [0] * self.engine.num_bindings
            bindings[self.input_idx] = int(self.d_input.data_ptr())
            bindings[self.output_idx] = int(self.d_output.data_ptr())
            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=int(stream.cuda_stream))

        if not ok:
            raise RuntimeError("TRT decoder 执行失败")

        stream.synchronize()
        return self.d_output.float().detach().cpu().numpy()


def build_vae_onnx_session(vae, vae_scale, vae_torch_device, onnx_path, provider="auto"):
    """导出并创建 ONNX Runtime session。仅在需要时调用。"""
    import onnxruntime as ort

    decode_wrapper = VAEDecodeWrapper(vae, vae_scale).to(vae_torch_device).eval()
    example = torch.randn((1, 4, 64, 64), device=vae_torch_device, dtype=torch.float32)
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    if onnx_dir and (not os.path.exists(onnx_dir)):
        os.makedirs(onnx_dir, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            decode_wrapper,
            example,
            onnx_path,
            export_params=True,
            do_constant_folding=True,
            input_names=["z"],
            output_names=["out"],
            dynamic_axes={"z": {0: "batch"}, "out": {0: "batch"}},
            opset_version=17,
        )

    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "auto":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, sess_options=sess_opt, providers=providers)
    return session


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认监听所有网卡")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    parser.add_argument("--reconstruct", action="store_true", help="是否进行VAE图像重建（默认不重建）")
    parser.add_argument("--vae_dir", default="../train_vae/vae", help="VAE 权重目录（包含 config.json/safetensors）")
    parser.add_argument("--reconstruct_count", type=int, default=4, help="每个接收批次最多重建的样本数")
    parser.add_argument("--decode_device", choices=["cpu","cuda","auto"], default="auto", help="VAE 解码设备：cpu/cuda/auto")
    # 额外控制参数
    parser.add_argument("--resize_to_src", type=int, default=1, help="保存时按原图分辨率重采样(1是/0否)")
    parser.add_argument("--latent_clip_sigma", type=float, default=0.0, help="潜变量按均值±sigma剪裁，0禁用")
    parser.add_argument("--latent_smooth_iters", type=int, default=0, help="潜变量3x3盒滤次数，0禁用")
    # Eb/N0 范围（与 TX 默认保持一致）
    parser.add_argument("--ebno_min", type=float, default=0.0, help="最小 Eb/N0(dB)")
    parser.add_argument("--ebno_max", type=float, default=6.0, help="最大 Eb/N0(dB)")
    parser.add_argument("--ebno_step", type=float, default=0.5, help="步长(dB)")
    parser.add_argument("--live", action="store_true", help="实时模式：持续接收并解码视频帧")
    parser.add_argument("--ebno", type=float, default=6.0, help="实时模式固定 Eb/N0")
    parser.add_argument("--show", action="store_true", help="实时显示解码图像窗口")
    parser.add_argument("--use_cuda_vae", action="store_true", help="VAE 解码优先使用 CUDA（不可用则回退 CPU）")
    parser.add_argument("--reconstruct_every", type=int, default=1, help="每 N 帧重建一次，减轻主机 CPU 压力")
    parser.add_argument("--profile_tag", default="", help="实验标签；为空时自动使用TX帧头里的profile_tag")
    parser.add_argument("--gamma", type=float, default=1.0, help="重建图像伽马校正，>1提亮，1为关闭")
    parser.add_argument("--median_ksize", type=int, default=0, help="中值滤波核大小(奇数，0关闭)，默认0(与旧版一致)")
    parser.add_argument("--save_every", type=int, default=1, help="每 N 帧保存一次重建图像，1为每帧保存")
    parser.add_argument("--vae_backend", choices=["torch", "torchtrt", "onnx", "trt_engine"], default="torch", help="VAE解码后端：torch / torchtrt / onnx / trt_engine")
    parser.add_argument("--vae_trt_engine", type=str, default=None, help="TensorRT VAE decoder 引擎路径（vae_backend=trt_engine 时使用）")
    parser.add_argument("--trt_fp16", type=int, default=1, help="torchtrt 模式是否使用 FP16（1启用/0关闭）")
    parser.add_argument("--trt_required", type=int, default=0, help="当请求 torchtrt 时，若初始化失败是否直接退出（1是/0否）")
    parser.add_argument("--trt_auto_install", type=int, default=0, help="缺少 torch_tensorrt 时是否自动 pip 安装（1是/0否）")
    parser.add_argument("--trt_pip_index", default="", help="自动安装 torch_tensorrt 时使用的 pip index-url（可选）")
    parser.add_argument("--onnx_model_path", default="results/vae_decoder.onnx", help="ONNX 模型导出路径")
    parser.add_argument("--onnx_provider", choices=["auto", "cuda", "cpu"], default="auto", help="ONNX Runtime 执行后端")
    parser.add_argument("--tf_graph_mode", type=int, default=1, help="RX demap+ldpc 是否使用 tf.function 图模式（1是/0否）")
    parser.add_argument("--tf_xla", type=int, default=0, help="RX demap+ldpc 图模式是否启用 XLA（1是/0否）")
    parser.add_argument("--modem_backend", default="traditional", choices=["traditional", "neural"], help="解调后端：traditional 或 neural")
    parser.add_argument("--modem_ckpt_dir", default="neural_modem_checkpoints_v2", help="神经解调器 checkpoint 目录")
    parser.add_argument("--modem_ckpt_path", default="", help="神经解调器 checkpoint 完整路径（优先于目录）")
    parser.add_argument("--pilot_len", type=int, default=0, help="可选导频长度；>0 时按前缀剥离并做同步/估计/均衡")
    parser.add_argument("--frontend_mode", default="traditional", choices=["traditional", "pilot_sync", "residual"], help="接收前端模式：traditional/pilot_sync/residual")
    args = parser.parse_args()
    host = args.host
    port = args.port
    reconstruct = args.reconstruct
    vae_dir = args.vae_dir
    # 解析并校验 VAE 本地目录
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        vae_dir,
        os.path.abspath(vae_dir),
        os.path.join(script_dir, "..", "..", "train_vae", "vae"),
        os.path.join(script_dir, "..", "train_vae", "vae"),
        os.path.join(script_dir, "vae"),
    ]
    chosen = None
    for c in candidates:
        cfg = os.path.join(c, "config.json")
        if os.path.isfile(cfg):
            chosen = os.path.abspath(c)
            break
    if chosen is None:
        print("未找到有效的 VAE 目录。已尝试：")
        for c in candidates:
            print(" -", os.path.abspath(c))
        raise FileNotFoundError("请通过 --vae_dir 指向包含 config.json 的本地 VAE 目录")
    else:
        vae_dir = chosen
        print(f"使用本地 VAE 目录：{vae_dir}")
    reconstruct_count = args.reconstruct_count
    decode_device = args.decode_device
    resize_to_src = bool(args.resize_to_src)
    clip_sigma = float(args.latent_clip_sigma)
    smooth_iters = int(args.latent_smooth_iters)
    local_profile_tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(args.profile_tag)) if str(args.profile_tag) else ""

    # 加载 RX 物理层模型（支持 traditional 或 neural 后端）
    if args.modem_backend == "neural":
        model_rx = RXSystemNeuralModem(ckpt_dir=args.modem_ckpt_dir, ckpt_path=args.modem_ckpt_path or None)
    else:
        model_rx = RXSystemConventionalTraining(training=False)
        load_weights_rx(model_rx, model_weights_path_conventional_training_rx)
    print_device_report_rx(model_rx)
    pilot_len = max(0, int(args.pilot_len))
    frontend_mode = str(args.frontend_mode)
    if pilot_len > 0:
        print(f"[RX] 已启用导频前缀处理: pilot_len={pilot_len}, frontend_mode={frontend_mode}")

    # VAE 解码器只加载一次，避免每帧重复加载导致严重开销
    # 设备选择优先级：
    # 1) --use_cuda_vae 明确请求 CUDA
    # 2) 否则按 --decode_device(cpu/cuda/auto)
    cuda_requested = bool(args.use_cuda_vae) or (decode_device == "cuda")
    if cuda_requested:
        vae_torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif decode_device == "auto":
        vae_torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        vae_torch_device = "cpu"

    if cuda_requested and vae_torch_device == "cpu":
        print("[Device][RX] 请求 CUDA VAE，但当前 PyTorch CUDA 不可用（或为CPU版PyTorch），回退到 CPU")
    vae = AutoencoderKL.from_pretrained(vae_dir, local_files_only=True).to(vae_torch_device).eval()
    vae_scale = getattr(vae.config, "scaling_factor", 1.0)
    print(f"[Device][RX] VAE device: {next(vae.parameters()).device}")
    print(f"[PROFILE][RX] tag={local_profile_tag if local_profile_tag else '<from_tx>'}, reconstruct_every={args.reconstruct_every}, save_every={args.save_every}, gamma={args.gamma}, median_ksize={args.median_ksize}")

    # 可选后端：Torch-TensorRT/ONNX Runtime
    vae_decoder_backend = "torch"
    trt_decoder = None
    trt_engine_decoder = None
    onnx_session = None
    if args.vae_backend == "torchtrt":
        trt_err = None
        if vae_torch_device != "cuda":
            trt_err = "当前非 CUDA 设备，无法启用 torchtrt"
        else:
            torch_tensorrt = None
            try:
                torch_tensorrt = importlib.import_module("torch_tensorrt")
            except Exception as e_imp:
                if int(args.trt_auto_install) == 1:
                    try:
                        cmd = [sys.executable, "-m", "pip", "install", "torch_tensorrt"]
                        if str(args.trt_pip_index).strip():
                            cmd.extend(["--index-url", str(args.trt_pip_index).strip()])
                        print(f"[TRT][RX] 未检测到 torch_tensorrt，尝试自动安装：{' '.join(cmd)}")
                        subprocess.check_call(cmd)
                        torch_tensorrt = importlib.import_module("torch_tensorrt")
                        print("[TRT][RX] torch_tensorrt 自动安装成功")
                    except Exception as e_install:
                        trt_err = f"torch_tensorrt 缺失且自动安装失败: {e_install}"
                else:
                    trt_err = f"torch_tensorrt 缺失: {e_imp}"

            if torch_tensorrt is not None:
                try:
                    decode_wrapper = VAEDecodeWrapper(vae, vae_scale).to(vae_torch_device).eval()
                    # 固定 latent shape（本项目默认 [B,4,64,64]）
                    example_shape = (1, 4, 64, 64)
                    dtype = torch.float16 if int(args.trt_fp16) == 1 else torch.float32
                    example = torch.randn(example_shape, device=vae_torch_device, dtype=dtype)
                    with torch.inference_mode():
                        scripted = torch.jit.trace(decode_wrapper, example, strict=False)
                    enabled_precisions = {torch.half} if int(args.trt_fp16) == 1 else {torch.float}
                    trt_decoder = torch_tensorrt.compile(
                        scripted,
                        inputs=[torch_tensorrt.Input(example_shape, dtype=dtype)],
                        enabled_precisions=enabled_precisions,
                        truncate_long_and_double=True,
                    )
                    vae_decoder_backend = "torchtrt"
                    print(f"[TRT][RX] torchtrt 编译成功，backend={vae_decoder_backend}, fp16={int(args.trt_fp16)==1}")
                except Exception as e_compile:
                    trt_err = f"torchtrt 编译失败: {e_compile}"

        if trt_err is not None:
            if int(args.trt_required) == 1:
                raise RuntimeError(f"[TRT][RX] {trt_err}")
            print(f"[TRT][RX] {trt_err}，回退到 torch")

    if args.vae_backend == "onnx":
        onnx_err = None
        try:
            onnx_session = build_vae_onnx_session(
                vae,
                vae_scale,
                vae_torch_device,
                args.onnx_model_path,
                provider=args.onnx_provider,
            )
            vae_decoder_backend = "onnx"
            print(f"[ONNX][RX] ONNX Runtime 初始化成功，backend={vae_decoder_backend}, providers={onnx_session.get_providers()}")
        except Exception as e_onnx:
            onnx_err = f"ONNX 初始化失败: {e_onnx}"
            print(f"[ONNX][RX] {onnx_err}，回退到 torch")

    if args.vae_backend == "trt_engine":
        trt_engine_err = None
        if not args.vae_trt_engine:
            trt_engine_err = "未提供 --vae_trt_engine 路径"
        else:
            try:
                trt_engine_decoder = TRTVAEDecoder(args.vae_trt_engine)
                vae_decoder_backend = "trt_engine"
                print(f"[TRT][RX] 已加载 TRT decoder engine: {args.vae_trt_engine}")
                print(f"[TRT][RX] decoder in={trt_engine_decoder.in_shape}, out={trt_engine_decoder.out_shape}")
            except Exception as e_trt_engine:
                trt_engine_err = f"TRT engine 初始化失败: {e_trt_engine}"

        if trt_engine_err is not None:
            if int(args.trt_required) == 1:
                raise RuntimeError(f"[TRT][RX] {trt_engine_err}")
            print(f"[TRT][RX] {trt_engine_err}，回退到 torch")

    def decode_latents_backend(z_torch):
        """统一执行 VAE 解码；返回 numpy，范围约为 [-1,1]。"""
        if vae_decoder_backend == "torchtrt" and trt_decoder is not None:
            z_in = z_torch.half() if int(args.trt_fp16) == 1 else z_torch.float()
            out = trt_decoder(z_in)
            return out.clamp(-1, 1).detach().cpu().numpy()
        if vae_decoder_backend == "trt_engine" and trt_engine_decoder is not None:
            z_np = (z_torch.float() / vae_scale).detach().cpu().numpy().astype(np.float32, copy=False)
            out_np = trt_engine_decoder.decode(z_np)
            return np.clip(out_np, -1.0, 1.0)
        if vae_decoder_backend == "onnx" and onnx_session is not None:
            z_np = (z_torch.float() / vae_scale).detach().cpu().numpy().astype(np.float32, copy=False)
            out_np = onnx_session.run(["out"], {"z": z_np})[0]
            return np.clip(out_np, -1.0, 1.0)
        out = vae.decode(z_torch.float() / vae_scale).sample
        return out.clamp(-1, 1).detach().cpu().numpy()

    # 后端一致性快速自检：若误用/错配 engine，会直接回退 torch，避免严重花屏
    if vae_decoder_backend != "torch":
        try:
            with torch.inference_mode():
                z_chk = torch.randn((1, 4, 64, 64), device=vae_torch_device, dtype=torch.float32)
                ref = vae.decode(z_chk / vae_scale).sample.clamp(-1, 1).detach().cpu().numpy()
                got = decode_latents_backend(z_chk)
            shape_ok = (got.shape == ref.shape)
            mae = float(np.mean(np.abs(got - ref))) if shape_ok else float("inf")
            if (not shape_ok) or (not np.isfinite(mae)) or (mae > 0.20):
                print(f"[VAE][RX] 后端自检失败：backend={vae_decoder_backend}, shape_ok={shape_ok}, mae={mae:.6f}，回退到 torch")
                vae_decoder_backend = "torch"
                trt_decoder = None
                trt_engine_decoder = None
                onnx_session = None
            else:
                print(f"[VAE][RX] 后端自检通过：backend={vae_decoder_backend}, mae={mae:.6f}")
        except Exception as e_chk:
            print(f"[VAE][RX] 后端自检异常：{e_chk}，回退到 torch")
            vae_decoder_backend = "torch"
            trt_decoder = None
            trt_engine_decoder = None
            onnx_session = None

    # demap+ldpc 图模式（常能降低 Python 调度开销）
    use_graph_mode = int(args.tf_graph_mode) == 1
    use_xla = int(args.tf_xla) == 1
    if use_graph_mode:
        @tf.function(reduce_retracing=True, jit_compile=use_xla)
        def rx_infer_graph(y_batch, ebno_db_use):
            bs_dynamic = tf.shape(y_batch)[0]
            return model_rx(batch_size=tf.cast(bs_dynamic, tf.int32), y=y_batch, ebno_db=ebno_db_use)
    else:
        rx_infer_graph = None

    # 读取元数据（离线路径）；直播路径用每帧头部提供的元数据
    meta = None
    n_bits = None
    fmin = None
    fmax = None
    latent_shape = None
    segments = None
    m_per_seg = None

    indices = None
    if os.path.exists(os.path.join("results", "vae_indices.npy")):
        try:
            indices = np.load(os.path.join("results", "vae_indices.npy"))
        except Exception:
            indices = None

    def bits_to_ints_batch(seg_bits, nbits, out_len):
        """将 [B, out_len*nbits] 比特批量转为 [B, out_len] 整数，避免 Python 双层循环。"""
        usable = np.rint(seg_bits[:, :out_len * nbits]).astype(np.int32, copy=False)
        grouped = usable.reshape(usable.shape[0], out_len, nbits)
        weights = (1 << np.arange(nbits - 1, -1, -1, dtype=np.int32)).reshape(1, 1, nbits)
        return np.sum(grouped * weights, axis=2, dtype=np.int32)

    def reconstruct_frame(seg_bits_list, conn_idx, runtime_meta=None):

        # 运行时元数据（实时模式）优先；否则使用离线加载的元数据
        if runtime_meta is not None:
            n_bits_rt = int(runtime_meta.get("n_bits", 8))
            fmin_rt = float(runtime_meta.get("fmin", -1.0))
            fmax_rt = float(runtime_meta.get("fmax", 1.0))
            latent_shape_rt = tuple(runtime_meta.get("latent_shape", [4,64,64]))
            segments_rt = int(runtime_meta.get("segments", len(seg_bits_list)))
            m_per_seg_rt = int(runtime_meta.get("m_per_seg", seg_bits_list[0].shape[1] // n_bits_rt))
            idx_rt = runtime_meta.get("indices", None)
            tag_rt = runtime_meta.get("profile_tag", "default")
            frame_id_rt = int(runtime_meta.get("frame_id", conn_idx))
        else:
            n_bits_rt, fmin_rt, fmax_rt = n_bits, fmin, fmax
            latent_shape_rt, segments_rt, m_per_seg_rt = latent_shape, segments, m_per_seg
            idx_rt = None
            tag_rt = "default"
            frame_id_rt = int(conn_idx)

        active_tag = local_profile_tag if local_profile_tag else re.sub(r"[^a-zA-Z0-9_.-]", "_", str(tag_rt))

        B = seg_bits_list[0].shape[0]
        ints_per_seg = [bits_to_ints_batch(seg_bits, n_bits_rt, m_per_seg_rt) for seg_bits in seg_bits_list]

        total_latent = int(np.prod(latent_shape_rt))
        latents = np.zeros((B, total_latent), dtype=np.float32)
        for seg_idx in range(segments_rt):
            vals = ints_per_seg[seg_idx].astype(np.float32) / (2 ** n_bits_rt - 1)
            vals = vals * (fmax_rt - fmin_rt) + fmin_rt
            if idx_rt is not None:
                idx_tbl = np.array(idx_rt)[seg_idx]
                for i in range(B):
                    idx_row = np.array(idx_tbl[i % len(idx_tbl)])
                    use_len = min(idx_row.shape[0], vals.shape[1])
                    latents[i, idx_row[:use_len]] = vals[i, :use_len]
            else:
                for i in range(B):
                    start = seg_idx * m_per_seg_rt
                    end = min(start + vals.shape[1], total_latent)
                    latents[i, start:end] = vals[i, :end - start]
        latents = latents.reshape(B, *latent_shape_rt)
        # 兼容旧/错误导出的 encoder：若携带了 8 通道（mean+logvar），仅取 mean 前 4 通道给 decoder
        if latents.ndim == 4 and latents.shape[1] == 8:
            latents = latents[:, :4, :, :]
            if conn_idx == 0:
                print("[TRT][RX] 检测到 8 通道 latent，已自动截取前 4 通道（mean）用于解码")

        with torch.inference_mode():
            z = torch.from_numpy(latents).to(vae_torch_device)
            imgs = decode_latents_backend(z)

        for j in range(imgs.shape[0]):
            img = ((imgs[j] + 1.0) / 2.0 * 255.0).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))

            # 轻量后处理：提亮 + 去噪（可通过参数关闭）
            gamma = float(args.gamma)
            if gamma > 0 and abs(gamma - 1.0) > 1e-6:
                img_f = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
                img_f = np.power(img_f, 1.0 / gamma)
                img = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)

            k = int(args.median_ksize)
            if k >= 3:
                if k % 2 == 0:
                    k += 1
                img = cv2.medianBlur(img, k)

            if args.show:
                cv2.imshow("rx_frame", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            save_every = max(1, int(args.save_every))
            if conn_idx % save_every == 0:
                out_dir = "results/recovered_images"
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{active_tag}_live_frame{frame_id_rt}_idx{j}.png")
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"重建成功，已保存：{out_path}")
            elif j == 0:
                print(f"重建完成但跳过保存 frame={conn_idx}（save_every={save_every}）")

    if args.live:
        # 实时模式：逐帧接收（头部 + segments），按帧重建
        stream = receive_tensor_stream(host, port)
        conn_idx = 0
        while True:
            header, tensors = next(stream)
            t0 = time.time()
            # 关键优化：按帧批量解码，避免每段单独调用 model_rx 带来的高开销
            num_seg = len(tensors)
            y_batch = tf.concat(tensors, axis=0)  # [segments, num_symbols_per_codeword]
            ebno_db_use = tf.fill([num_seg], float(args.ebno))
            no_use = ebnodb2no(ebno_db_use, num_bits_per_symbol, coderate)
            no_use = expand_to_rank(no_use, 2)
            pilot_len_frame = int(header.get("pilot_len", pilot_len))
            if pilot_len_frame > 0:
                y_batch, frontend_meta = frontend_process_burst(y_batch, pilot_len_frame, frontend_mode, no=no_use)
                print(f"[RX] frontend_meta={frontend_meta}")
            if rx_infer_graph is not None:
                b_hat_batch = rx_infer_graph(y_batch, ebno_db_use)
            else:
                b_hat_batch = model_rx(batch_size=tf.cast(num_seg, tf.int32), y=y_batch, ebno_db=ebno_db_use)
            b_hat_np = b_hat_batch.numpy()  # 一次性回传 CPU
            seg_list = [b_hat_np[i:i+1] for i in range(num_seg)]
            t1 = time.time()
            if args.reconstruct_every > 1 and (conn_idx % args.reconstruct_every != 0):
                print(f"跳过重建 frame={conn_idx}（reconstruct_every={args.reconstruct_every}）")
            else:
                reconstruct_frame(seg_list, conn_idx, runtime_meta=header)
            t2 = time.time()
            demap_ms = (t1 - t0) * 1000
            decode_ms = (t2 - t1) * 1000
            total_ms = (t2 - t0) * 1000
            fps = 1000.0 / total_ms if total_ms > 0 else 0
            tag_live = local_profile_tag if local_profile_tag else re.sub(r"[^a-zA-Z0-9_.-]", "_", str(header.get("profile_tag", "default")))
            print(f"RX profile[{tag_live}]: total={total_ms:.1f}ms (demap+ldpc={demap_ms:.1f}ms, vae_decode={decode_ms:.1f}ms), est_fps={fps:.1f}")
            conn_idx += 1
    else:
        # 兼容原有离线评估
        # 离线模式下从本地文件读取元数据
        meta = np.load("results/vae_meta.npz")
        n_bits = int(meta["n_bits"])
        fmin = float(meta["fmin"])
        fmax = float(meta["fmax"])
        latent_shape = tuple(meta["latent_shape"].tolist())
        segments = int(meta["segments"]) if "segments" in meta else 1
        m_per_seg = int(meta["m_per_seg"]) if "m_per_seg" in meta else (latent_shape[0] * latent_shape[1] * latent_shape[2] // segments // n_bits)

        try:
            b_src_all = np.load("results/source_b_vae.npy")
        except Exception as e_src:
            raise RuntimeError(f"无法加载源比特: {e_src}")

        if m_per_seg is None:
            if b_src_all.ndim == 3:
                m_per_seg = b_src_all.shape[-1] // n_bits
            else:
                m_per_seg = b_src_all.shape[1] // n_bits

        ebno_dbs = np.arange(args.ebno_min, args.ebno_max, args.ebno_step)
        total_packets = ebno_dbs.shape[0] * segments
        bers = []

        decoded_segments = {i: [None] * segments for i in range(ebno_dbs.shape[0])}
        bs = int(training_batch_size.numpy()) if hasattr(training_batch_size, 'numpy') else int(training_batch_size)

        def prepare_batch(arr):
            if arr.shape[0] > bs:
                return arr[:bs]
            if arr.shape[0] < bs:
                reps = bs // arr.shape[0] + 1
                return np.tile(arr, (reps, 1))[:bs]
            return arr

        for conn_i, y in enumerate(receive_tensor(host, port, total_packets), start=0):
            ebno_idx = conn_i // segments
            seg_id = conn_i % segments
            ebno_db = ebno_dbs[ebno_idx]
            print(f"\n===== 连接 {conn_i+1}/{total_packets} | Eb/N0={ebno_db} dB | 分段 {seg_id+1}/{segments} =====")
            s = time.time()
            ebno_db_use = tf.fill([training_batch_size], float(ebno_db))
            no_use = ebnodb2no(ebno_db_use, num_bits_per_symbol, coderate)
            no_use = expand_to_rank(no_use, 2)
            if pilot_len > 0:
                y, frontend_meta = frontend_process_burst(y, pilot_len, frontend_mode, no=no_use)
                print(f"[RX] frontend_meta={frontend_meta}")
            b_hat = model_rx(batch_size=training_batch_size, y=y, ebno_db=ebno_db_use)
            e = time.time()
            print('推理时间：', e-s)

            try:
                if b_src_all.ndim == 3:
                    b_src_seg = b_src_all[seg_id]
                else:
                    b_src_seg = b_src_all
                b_src_use = prepare_batch(b_src_seg)
                num_errors = np.sum(b_src_use.astype(np.int32) != b_hat.numpy().astype(np.int32))
                ber = num_errors / b_src_use.size
                print(f"分段 BER = {ber:.6f}")
                bers.append(ber)
            except Exception as e_ber:
                print("BER 计算失败：", e_ber)

            decoded_segments[ebno_idx][seg_id] = b_hat.numpy()

            if reconstruct and all(part is not None for part in decoded_segments[ebno_idx]):
                try:
                    seg_list = decoded_segments[ebno_idx]
                    reconstruct_frame(seg_list, conn_i)
                except Exception as e_img:
                    print("生成恢复图像失败：", e_img)

        plt.figure()
        plt.plot(np.arange(len(bers)), bers)
        plt.yscale('log')
        plt.ylim(1e-4, 1e0)
        plt.xlabel('Packet Index (per segment)')
        plt.ylabel('BER')
        plt.grid(True)
        plt.savefig("results/deep_bers.png")


