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
from edge_semcom.neural_modem import NeuralModulator

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation


import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
import json
import socket
import re

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


class TXSystemConventionalTraining(Model):

    def __init__(self, training):
        super().__init__()

        self._training = training

        ################
        ## Transmitter
        ################
        # Trainable constellation
        # We initialize a custom constellation with qam points
        qam_points = Constellation("qam", num_bits_per_symbol).points
        self.constellation = Constellation("custom",
                                           num_bits_per_symbol,
                                           points=qam_points,
                                           normalize=True,
                                           center=True)
        # To make the constellation trainable, we need to create seperate
        # variables for the real and imaginary parts
        self.points_r = self.add_weight(shape=qam_points.shape,
                                        initializer="zeros")
        self.points_i = self.add_weight(shape=qam_points.shape,
                                        initializer="zeros")
        self.points_r.assign(tf.math.real(qam_points))
        self.points_i.assign(tf.math.imag(qam_points))

        self._mapper = Mapper(constellation=self.constellation)


    def call(self, batch_size, c, ebno_db):
        # Set the constellation points equal to a complex tensor constructed
        # from two real-valued variables
        points = tf.complex(self.points_r, self.points_i)
        self.constellation.points = points
        ################
        ## Transmitter
        ################
        # Modulation
        x = self._mapper(c) # x [batch size, num_symbols_per_codeword]

        return x


class TXSystemBaseline(Model):
    """传统基线 TX：固定 QAM constellation。"""

    def __init__(self):
        super().__init__()
        self.constellation = Constellation("qam", num_bits_per_symbol, trainable=False)
        self._mapper = Mapper(constellation=self.constellation)

    def call(self, batch_size, c, ebno_db):
        x = self._mapper(c)
        return x


class TXSystemNeuralModem(Model):
    """神经调制器 TX：LDPC 编码后用 NeuralModulator 替代 Mapper。"""

    def __init__(self, ckpt_dir: str = None, ckpt_path: str = None):
        super().__init__()
        self._modulator = NeuralModulator(num_bits_per_symbol, num_symbols_per_codeword)

        # 先 build 一次，便于从 checkpoint 恢复
        dummy_bits = tf.zeros([1, num_symbols_per_codeword, num_bits_per_symbol], dtype=tf.float32)
        _ = self._modulator(dummy_bits, training=False)

        ckpt = tf.train.Checkpoint(modulator=self._modulator)
        restore_path = ckpt_path
        if not restore_path and ckpt_dir:
            restore_path = tf.train.latest_checkpoint(ckpt_dir)
        if restore_path:
            ckpt.restore(restore_path).expect_partial()
            print(f"[TX][NEURAL] 已加载神经调制器 checkpoint: {restore_path}")
        else:
            print("[TX][NEURAL] 未提供 checkpoint，使用随机初始化权重")

    def call(self, batch_size, c, ebno_db):
        c = tf.cast(c, tf.float32)
        symbols_bits = tf.reshape(
            c[:, :num_symbols_per_codeword * num_bits_per_symbol],
            [batch_size, num_symbols_per_codeword, num_bits_per_symbol],
        )
        x = self._modulator(symbols_bits, training=False)
        return x


# Utility function to load and set weights of a model
def load_weights_tx(model, model_weights_path):
    model(tf.cast(1, tf.int32), tf.zeros([1, n], dtype=tf.float32), tf.constant(10.0, tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    points = tf.complex(model.points_r, model.points_i)
    model.constellation.points = points
    print(f"\n TX weights have loaded from  {model_weights_path}.")


def maybe_load_trainable_mapper_weights(model, model_weights_path):
    if not model_weights_path:
        print("[TX] 未提供 trainable mapper 权重路径，使用默认初始化")
        return
    if not os.path.isfile(model_weights_path):
        print(f"[TX] 权重文件不存在，跳过加载: {model_weights_path}")
        return
    load_weights_tx(model, model_weights_path)


def build_pilot_symbols(batch_size, pilot_len, pilot_value="1+1j"):
    """生成固定导频，用于未来 USRP/多径链路的同步与信道估计。"""
    pilot_len = int(pilot_len)
    if pilot_len <= 0:
        return None
    pilot = tf.complex(
        tf.ones([batch_size, pilot_len], dtype=tf.float32),
        tf.ones([batch_size, pilot_len], dtype=tf.float32),
    )
    pilot = pilot / tf.cast(tf.sqrt(tf.constant(2.0, dtype=tf.float32)), pilot.dtype)
    return pilot

# 先发送 8 字节表示长度，再发送实际数。连续发送多个张量，保持连接
def send_tensor(tensor: tf.Tensor, host: str, port: int):

    # 1. 序列化 Tensor
    data = tf.io.serialize_tensor(tensor).numpy()  # 转为 bytes
    size = len(data)
    print(f"准备发送 tensor，dtype={tensor.dtype}, shape={tensor.shape}, byte大小={size}")

    # 2. 创建 TCP 连接
    with socket.create_connection((host, port)) as s:
        # 发送长度（8 字节，大端序）
        s.sendall(size.to_bytes(8, byteorder='big'))
        # 发送实际 Tensor 数据
        s.sendall(data)
        print("发送完成，关闭连接。")

def send_frame(sock: socket.socket, header: dict, tensors: list):
    """
    在已建立的 socket 上发送一帧：头部(H) + 若干张量(T)。
    头部为 JSON 字节，前置 1 字节标记 'H' 和 8 字节长度；
    每个张量前置 1 字节标记 'T' 和 8 字节长度。
    """
    hdr_bytes = json.dumps(header).encode('utf-8')
    sock.sendall(b'H')
    sock.sendall(len(hdr_bytes).to_bytes(8, byteorder='big'))
    sock.sendall(hdr_bytes)
    for t in tensors:
        data = tf.io.serialize_tensor(t).numpy()
        sock.sendall(b'T')
        sock.sendall(len(data).to_bytes(8, byteorder='big'))
        sock.sendall(data)
    print(f"本帧发送完成：segments={len(tensors)}")


import tensorflow as tf
import argparse
import time


class TRTVAEEncoder:
    """使用 TensorRT engine 执行 VAE encoder 推理。"""
    def __init__(self, engine_path, vae_scale=1.0):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("未安装 TensorRT Python 依赖（tensorrt）")
        if not torch.cuda.is_available():
            raise RuntimeError("当前 PyTorch CUDA 不可用，无法运行 TRT engine")
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TRT engine 不存在: {engine_path}")

        self.vae_scale = float(vae_scale)
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
                in_shape = (1, 3, 512, 512)
                self.context.set_input_shape(self.input_name, in_shape)
            out_shape = tuple(self.context.get_tensor_shape(self.output_name))
            if -1 in out_shape:
                out_shape = (1, 4, 64, 64)

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
                in_shape = (1, 3, 512, 512)
                self.context.set_binding_shape(self.input_idx, in_shape)
            out_shape = tuple(self.context.get_binding_shape(self.output_idx))
            if -1 in out_shape:
                out_shape = (1, 4, 64, 64)

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

    def encode(self, x_np: np.ndarray) -> torch.Tensor:
        if x_np.shape != self.in_shape:
            raise ValueError(f"TRT encoder 输入 shape 不匹配: got={x_np.shape}, expect={self.in_shape}")

        x_t = torch.from_numpy(np.ascontiguousarray(x_np)).to(self.device, dtype=self.in_torch_dtype, non_blocking=True)
        self.d_input.copy_(x_t)
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
            raise RuntimeError("TRT encoder 执行失败")

        stream.synchronize()
        y = self.d_output.float()
        # 部分 VAE encoder 导出会输出 8 通道（mean+logvar）；语义通信链路仅使用 mean 的前 4 通道
        if y.ndim == 4 and y.shape[1] == 8:
            y = y[:, :4, :, :]
        y = (y * self.vae_scale).detach().cpu()
        return y


def resolve_vae_scale(vae_dir, default=1.0):
    cfg = os.path.join(vae_dir, "config.json")
    scale = None
    try:
        if os.path.isfile(cfg):
            with open(cfg, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # 兼容不同导出格式的字段命名
            for k in ["scaling_factor", "latent_scaling_factor", "vae_scaling_factor"]:
                if k in obj:
                    scale = float(obj[k])
                    break
    except Exception:
        scale = None

    # 某些 VAE 的 config.json 不包含 scaling_factor；从模型配置兜底读取
    if scale is None:
        try:
            vae_tmp = AutoencoderKL.from_pretrained(vae_dir, local_files_only=True)
            scale = float(getattr(vae_tmp.config, "scaling_factor", default))
            del vae_tmp
        except Exception:
            scale = float(default)

    return float(scale)


# VAE 实时编码辅助#
def preprocess_frame(frame_bgr):
    # 期望输入 BGR，高度宽度可变；输出 [1,3,512,512]，范围 [-1,1]
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = tfm(img_rgb).unsqueeze(0)
    return x


def save_original_frame(frame_bgr, out_dir, profile_tag, frame_id):
    os.makedirs(out_dir, exist_ok=True)
    resized = cv2.resize(frame_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    out_path = os.path.join(out_dir, f"{profile_tag}_live_frame{frame_id}_idx0.png")
    cv2.imwrite(out_path, resized)
    return out_path


def select_top_indices(latents, total_m):
    # latents: torch tensor [1,C,H,W]; 按幅值全局取 TopK，返回索引 [1,total_m]
    B, C, H, W = latents.shape
    L = C * H * W
    flat = latents.reshape(B, L)
    energy = flat.abs()
    k_take = min(total_m, L)
    _, idx = torch.topk(energy, k=k_take, dim=1)
    return idx


def quantize_to_bits(latents, n_bits=10, target_k=7500, segments=1, q_low_pct=0.5, q_high_pct=99.5):
    # latents: [1,C,H,W] torch
    B = latents.shape[0]
    flat = latents.reshape(B, -1)
    # 使用分位数范围而非全局 min/max，减少异常值导致的发黑与噪点
    q_low = max(0.0, min(100.0, float(q_low_pct))) / 100.0
    q_high = max(0.0, min(100.0, float(q_high_pct))) / 100.0
    if q_high <= q_low:
        q_low, q_high = 0.0, 1.0
    flat_1d = flat.reshape(-1)
    if q_low == 0.0 and q_high == 1.0:
        fmin = float(flat_1d.min())
        fmax = float(flat_1d.max())
    else:
        fmin = float(torch.quantile(flat_1d, q_low).item())
        fmax = float(torch.quantile(flat_1d, q_high).item())
    if fmax <= fmin:
        fmin = float(flat_1d.min())
        fmax = float(flat_1d.max())
    M_per_seg = target_k // n_bits
    total_M = M_per_seg * segments
    idx_all = select_top_indices(latents, total_M)  # [B,total_M]
    vals_all = torch.gather(flat, 1, idx_all)

    bits_segments = []
    indices_segments = []
    bit_shifts = torch.arange(n_bits - 1, -1, -1, device=latents.device, dtype=torch.long)
    for seg in range(segments):
        start = seg * M_per_seg
        end = start + M_per_seg
        idx_slice = idx_all[:, start:end]
        val_slice = vals_all[:, start:end]
        norm = (val_slice - fmin) / (fmax - fmin + 1e-8)
        q_int = torch.clamp(torch.round(norm * (2 ** n_bits - 1)).long(), 0, 2 ** n_bits - 1)

        # 向量化按位展开，替代 Python 字符串/循环，保持同样的高位到低位顺序
        seg_bits_t = ((q_int.unsqueeze(-1) >> bit_shifts) & 1).reshape(B, -1)
        seg_bits = seg_bits_t.to(dtype=torch.float32).cpu().numpy()  # [B, target_k]
        if seg_bits.shape[1] < target_k:
            pad = np.zeros((B, target_k - seg_bits.shape[1]), dtype=np.float32)
            seg_bits = np.concatenate([seg_bits, pad], axis=1)
        elif seg_bits.shape[1] > target_k:
            seg_bits = seg_bits[:, :target_k]
        bits_segments.append(seg_bits)
        indices_segments.append(idx_slice.cpu().numpy().astype(np.int32))

    bits = np.stack(bits_segments, axis=0)       # [segments,B,target_k]
    indices = np.stack(indices_segments, axis=0) # [segments,B,M_per_seg]
    meta = {
        "n_bits": np.array(n_bits, dtype=np.int32),
        "fmin": np.array(fmin, dtype=np.float32),
        "fmax": np.array(fmax, dtype=np.float32),
        "latent_shape": np.array(latents.shape[1:], dtype=np.int32),
        "segments": np.array(segments, dtype=np.int32),
        "m_per_seg": np.array(M_per_seg, dtype=np.int32),
    }
    return bits, meta, indices


def open_camera(cam_src="0"):
    # cam_src 可以是索引、路径或 gstreamer 字符串
    # Jetson 上优先尝试 CSI 的 nvarguscamerasrc，其次 V4L2
    def _setup_common(c):
        try:
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return c

    # 1) 若用户明确给了 gstreamer 字符串，直接按 GStreamer 打开
    if isinstance(cam_src, str) and ("nvarguscamerasrc" in cam_src or "!" in cam_src):
        cap = cv2.VideoCapture(cam_src, cv2.CAP_GSTREAMER)
        return _setup_common(cap)

    # 2) 索引/路径：先按整数索引打开
    idx = None
    try:
        idx = int(cam_src)
    except Exception:
        idx = None

    if idx is not None:
        # 2.1 先走 V4L2 (某些情况下可用，但 Jetson 通常走 nvarguscamerasrc)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            _setup_common(cap)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()

        # 2.2 Jetson CSI 最优管道：nvarguscamerasrc 直接到 BGR (已验证可用)
        gst = (
            f"nvarguscamerasrc sensor-id={idx} ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1 max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return _setup_common(cap)
        cap.release()
        
        # 2.3 备选方案1：I420 中间格式（轻微格式差异）
        gst2 = (
            f"nvarguscamerasrc sensor-id={idx} ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
            "nvvidconv ! video/x-raw, format=I420 ! videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1 max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst2, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return _setup_common(cap)
        cap.release()
        
        # 2.4 备选方案2：RGBA 路由（某些硬件更稳定）
        gst3 = (
            f"nvarguscamerasrc sensor-id={idx} ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
            "nvvidconv ! video/x-raw, format=RGBA ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst3, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return _setup_common(cap)
        cap.release()
        
        # 2.5 备选方案3：硬件加速 JPEG 编解码
        gst4 = (
            f"nvarguscamerasrc sensor-id={idx} ! "
            "video/x-raw(memory:NVMM), format=NV12 ! "
            "nvjpegenc ! jpegdec ! videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1 max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst4, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return _setup_common(cap)
        cap.release()
        
        # 2.6 最后备选：禁用自动白平衡
        gst5 = (
            f"nvarguscamerasrc sensor-id={idx} awb-mode=0 ! "
            "video/x-raw(memory:NVMM), format=NV12 ! "
            "nvvidconv ! video/x-raw, format=RGBA ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst5, cv2.CAP_GSTREAMER)
        return _setup_common(cap)

    # 3) 普通文件路径或设备路径
    cap = cv2.VideoCapture(cam_src)
    return _setup_common(cap)


def print_device_report_tx(model_tx, encoder):
    """统一打印 TX 侧 TensorFlow/PyTorch 设备与关键模块执行设备。"""
    tf_gpus = tf.config.list_physical_devices('GPU')
    print(f"[Device][TX] TensorFlow GPUs: {[d.name for d in tf_gpus] if tf_gpus else 'None'}")
    print(f"[Device][TX] PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Device][TX] PyTorch CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        b_dummy = tf.zeros([1, k], dtype=tf.float32)
        c_dummy = encoder(b_dummy)
        ebno_dummy = tf.constant([10.0], dtype=tf.float32)
        x_dummy = model_tx(batch_size=tf.cast(1, tf.int32), c=c_dummy, ebno_db=ebno_dummy)
        print(f"[Device][TX] LDPC encoder output device: {c_dummy.device}")
        print(f"[Device][TX] Mapper output device: {x_dummy.device}")
    except Exception as e:
        print(f"[Device][TX] 模块设备探测失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", default="127.0.0.1", required=False,
                        help="接收端 IP 地址")
    parser.add_argument("--port", type=int, default=5000, help="接收端端口，默认 5000")
    parser.add_argument("--batch", type=int, default=128, help="batch_size")
    parser.add_argument("--n", type=int, default=250, help="每 batch 的符号数量")
    parser.add_argument("--live", action="store_true", help="启用实时摄像头模式")
    parser.add_argument("--cam_src", default="0", help="摄像头源，索引/路径/gstreamer 字符串")
    parser.add_argument("--vae_dir", default="../train_vae/vae", help="VAE 权重目录")
    parser.add_argument("--vae_trt_engine", type=str, default=None, help="TensorRT VAE encoder 引擎路径（可选，为None时使用PyTorch）")
    parser.add_argument("--target_k", type=int, default=7500, help="每段比特数，需与k一致")
    parser.add_argument("--segments", type=int, default=1, help="分段数")
    parser.add_argument("--n_bits", type=int, default=10, help="量化位数")
    parser.add_argument("--ebno", type=float, default=6.0, help="实时模式固定 Eb/N0")
    parser.add_argument("--use_cuda_vae", action="store_true", help="VAE 编码优先使用 CUDA（不可用则回退 CPU）")
    parser.add_argument("--profile_tag", default="default", help="实验标签，会写入日志与帧头，便于对照实验")
    parser.add_argument("--modem_backend", default="traditional", choices=["traditional", "neural"], help="调制后端：traditional 或 neural")
    parser.add_argument("--modem_ckpt_dir", default="neural_modem_checkpoints_v2", help="神经调制器 checkpoint 目录")
    parser.add_argument("--modem_ckpt_path", default="", help="神经调制器 checkpoint 完整路径（优先于目录）")
    parser.add_argument("--pilot_len", type=int, default=0, help="可选导频长度；>0 时在每段 payload 前拼接导频")
    parser.add_argument("--q_low_pct", type=float, default=0.0, help="量化下分位数(0-100)，默认0(与旧版一致)")
    parser.add_argument("--q_high_pct", type=float, default=100.0, help="量化上分位数(0-100)，默认100(与旧版一致)")
    parser.add_argument("--warmup_frames", type=int, default=0, help="相机预热丢弃帧数，默认0(与旧版一致)")
    parser.add_argument("--tf_graph_mode", type=int, default=1, help="TX ldpc+map 是否使用 tf.function 图模式（1是/0否）")
    parser.add_argument("--tf_xla", type=int, default=0, help="TX 图模式是否启用 XLA（1是/0否）")
    parser.add_argument("--save_originals", action="store_true", help="save TX-side reference images in live mode")
    parser.add_argument("--save_original_dir", default="results/original_images", help="directory for TX-side reference images")
    parser.add_argument("--save_original_every", type=int, default=1, help="save one TX-side reference image every N frames")
    args = parser.parse_args()
    profile_tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(args.profile_tag))

    # 共用组件
    if args.modem_backend == "neural":
        model_tx = TXSystemNeuralModem(ckpt_dir=args.modem_ckpt_dir, ckpt_path=args.modem_ckpt_path or None)
    else:
        model_tx = TXSystemConventionalTraining(training=True)
    encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
    if args.modem_backend != "neural":
        load_weights_tx(model_tx, model_weights_path_conventional_training_tx)
    channel = AWGN()
    print_device_report_tx(model_tx, encoder)
    pilot_len = max(0, int(args.pilot_len))
    if pilot_len > 0:
        print(f"[TX] 已启用导频前缀: pilot_len={pilot_len}")

    use_graph_mode = int(args.tf_graph_mode) == 1
    use_xla = int(args.tf_xla) == 1
    if use_graph_mode:
        @tf.function(reduce_retracing=True, jit_compile=use_xla)
        def tx_channel_graph(b_tf_all, ebno_scalar):
            batch_items = tf.shape(b_tf_all)[0]
            ebno_db_use = tf.fill([batch_items], ebno_scalar)
            no = ebnodb2no(ebno_db_use, num_bits_per_symbol, coderate)
            no = expand_to_rank(no, 2)
            c_all = encoder(b_tf_all)
            x_all = model_tx(batch_size=batch_items, c=c_all, ebno_db=ebno_db_use)
            if pilot_len > 0:
                pilot = build_pilot_symbols(batch_items, pilot_len)
                x_all = tf.concat([pilot, x_all], axis=1)
            return channel(x_all, no)
    else:
        tx_channel_graph = None

    if args.live:
        # 实时摄像头模式
        cap = open_camera(args.cam_src)
        if not cap.isOpened():
            raise RuntimeError(f"摄像头打开失败: {args.cam_src}")
        trt_encoder = None
        use_trt_encoder = False
        vae = None

        vae_scale = resolve_vae_scale(args.vae_dir, default=1.0)
        if args.vae_trt_engine:
            try:
                trt_encoder = TRTVAEEncoder(args.vae_trt_engine, vae_scale=vae_scale)
                use_trt_encoder = True
                print(f"[TRT][TX] 已启用 TRT encoder: {args.vae_trt_engine}")
                print(f"[TRT][TX] encoder in={trt_encoder.in_shape}, out={trt_encoder.out_shape}, vae_scale={vae_scale}")
            except Exception as e_trt:
                print(f"[TRT][TX] TRT encoder 初始化失败，回退到 PyTorch: {e_trt}")

        if not use_trt_encoder:
            vae_device = "cuda" if args.use_cuda_vae and torch.cuda.is_available() else "cpu"
            if args.use_cuda_vae and vae_device == "cpu":
                print("[Device][TX] 请求 CUDA VAE，但 CUDA 不可用，回退到 CPU")
            vae = AutoencoderKL.from_pretrained(args.vae_dir, local_files_only=True).to(vae_device).eval()
            vae_scale = getattr(vae.config, "scaling_factor", vae_scale)
            print(f"[Device][TX] VAE device: {next(vae.parameters()).device}")
        bs_live = 1

        print("进入实时摄像头推流模式，按 Ctrl+C 结束")
        print(f"[PROFILE][TX] tag={profile_tag}, segments={args.segments}, n_bits={args.n_bits}, ebno={args.ebno}, q=({args.q_low_pct},{args.q_high_pct}), warmup={args.warmup_frames}")
        # 相机预热：丢弃前若干帧，降低首帧过暗/过曝概率
        warmup = max(0, int(args.warmup_frames))
        for _ in range(warmup):
            cap.read()
        if warmup > 0:
            print(f"[TX] 相机预热完成，已丢弃 {warmup} 帧")

        sock = None
        frame_idx = 0
        while True:
            # 确保持长连接；如断开则重连
            if sock is None:
                try:
                    sock = socket.create_connection((args.dst, args.port))
                    print(f"已连接到 {args.dst}:{args.port}，开始长连接推流")
                except Exception as e:
                    print(f"连接失败，重试中: {e}")
                    time.sleep(0.5)
                    continue

            ok, frame = cap.read()
            if not ok:
                print("读取摄像头失败，重试...")
                time.sleep(0.1)
                continue

            t0 = time.time()
            x = preprocess_frame(frame)
            with torch.no_grad():
                if use_trt_encoder and trt_encoder is not None:
                    latents = trt_encoder.encode(x.detach().cpu().numpy().astype(np.float32, copy=False))
                else:
                    x = x.to(vae.device)
                    latents = vae.encode(x).latent_dist.mean * vae_scale  # [1,4,64,64], 按配置缩放
            bits, meta, indices = quantize_to_bits(
                latents,
                n_bits=args.n_bits,
                target_k=args.target_k,
                segments=args.segments,
                q_low_pct=args.q_low_pct,
                q_high_pct=args.q_high_pct,
            )
            t1 = time.time()
            # 逐段编码，打包为一帧发送（携带头部：fmin/fmax/latent_shape/indices等）
            # 逐段改为批量一次性处理，减少 Python/TF 反复调度开销
            b_tf_all = tf.reshape(tf.convert_to_tensor(bits, dtype=tf.float32), [-1, args.target_k])
            if tx_channel_graph is not None:
                x_ch_all = tx_channel_graph(b_tf_all, tf.constant(float(args.ebno), dtype=tf.float32))
            else:
                batch_items = tf.shape(b_tf_all)[0]
                ebno_db_use = tf.fill([batch_items], float(args.ebno))
                no = ebnodb2no(ebno_db_use, num_bits_per_symbol, coderate)
                no = expand_to_rank(no, 2)
                c_all = encoder(b_tf_all)
                x_all = model_tx(batch_size=batch_items, c=c_all, ebno_db=ebno_db_use)
                if pilot_len > 0:
                    pilot = build_pilot_symbols(batch_items, pilot_len)
                    x_all = tf.concat([pilot, x_all], axis=1)
                x_ch_all = channel(x_all, no)
            tensors = tf.split(x_ch_all, num_or_size_splits=args.segments, axis=0)
            t2 = time.time()

            header = {
                "magic": "FRAME1",
                "profile_tag": profile_tag,
                "frame_id": int(frame_idx),
                "n_bits": int(meta["n_bits"]),
                "fmin": float(meta["fmin"]),
                "fmax": float(meta["fmax"]),
                "latent_shape": list(map(int, meta["latent_shape"])),
                "segments": int(meta["segments"]),
                "m_per_seg": int(meta["m_per_seg"]),
                "pilot_len": int(pilot_len),
                # indices: [segments,B,M_per_seg]，B=1；转为列表
                "indices": indices.tolist(),
            }
            try:
                send_frame(sock, header, tensors)
                t3 = time.time()
                if args.save_originals and frame_idx % max(1, int(args.save_original_every)) == 0:
                    ref_path = save_original_frame(frame, args.save_original_dir, profile_tag, frame_idx)
                    print(f"TX reference saved: {ref_path}")
                total_ms = (t3 - t0) * 1000
                encode_ms = (t1 - t0) * 1000
                mod_ms = (t2 - t1) * 1000
                send_ms = (t3 - t2) * 1000
                fps = 1000.0 / total_ms if total_ms > 0 else 0
                print(f"TX profile[{profile_tag}]: total={total_ms:.1f}ms (encode+quant={encode_ms:.1f}ms, ldpc+map={mod_ms:.1f}ms, send={send_ms:.1f}ms), est_fps={fps:.1f}")
                frame_idx += 1
            except Exception as e:
                print(f"发送失败，重连中: {e}")
                try:
                    sock.close()
                except Exception:
                    pass
                sock = None
                time.sleep(0.2)
                continue

            # 控制帧率
            time.sleep(0.05)
    else:
        # 离线 npy 模式（原行为）
        print("正在加载 VAE 生成的比特...")
        b_numpy = np.load("results/source_b_vae.npy")

        if b_numpy.ndim == 2:
            segments = 1
        elif b_numpy.ndim == 3:
            segments = b_numpy.shape[0]
        else:
            raise ValueError(f"source_b_vae.npy 维度异常: {b_numpy.shape}")

        bs = int(training_batch_size.numpy()) if hasattr(training_batch_size, 'numpy') else int(training_batch_size)

        def prepare_batch(arr):
            if arr.shape[0] > bs:
                return arr[:bs]
            if arr.shape[0] < bs:
                reps = bs // arr.shape[0] + 1
                return np.tile(arr, (reps, 1))[:bs]
            return arr

        if segments == 1:
            bit_segments = [prepare_batch(b_numpy)]
        else:
            bit_segments = [prepare_batch(b_numpy[s]) for s in range(segments)]

        model_tx.constellation.show()
        plt.show()
        channel = AWGN()

        ebno_dbs = np.arange(ebno_db_min, ebno_db_max, 0.5)
        for ebno_db in ebno_dbs:
            print(f"\n===== 当前 Eb/N0 = {ebno_db} dB，分段数={segments} =====")
            if len(ebno_db.shape) == 0:
                ebno_db_use = tf.fill([training_batch_size], ebno_db)
            else:
                ebno_db_use = ebno_db
            no = ebnodb2no(ebno_db_use, num_bits_per_symbol, coderate)
            no = expand_to_rank(no, 2)

            for seg_id, b_seg in enumerate(bit_segments):
                print(f"发送分段 {seg_id+1}/{segments}")
                b_tf = tf.convert_to_tensor(b_seg, dtype=tf.float32)
                c = encoder(b_tf)
                x = model_tx(batch_size=training_batch_size, c=c, ebno_db=ebno_db_use)
                if pilot_len > 0:
                    pilot = build_pilot_symbols(tf.shape(x)[0], pilot_len)
                    x = tf.concat([pilot, x], axis=1)
                x = channel(x, no)
                send_tensor(x, args.dst, args.port)
                time.sleep(1.0)










