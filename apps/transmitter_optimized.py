##=======================================GPU Configuration and Imports================================================##
import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sionna.phy
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
from sionna.phy import Block
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no, log10, expand_to_rank
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource
from sionna.phy.utils import sim_ber

sionna.phy.config.seed = 42

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
import time
import argparse

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

##=======================================Simulation Parameters========================================================##
ebno_db_min = 5.5
ebno_db_max = 6.0

num_bits_per_symbol = 6
modulation_order = 2**num_bits_per_symbol
coderate = 0.5
n = 15000
num_symbols_per_codeword = n//num_bits_per_symbol
k = int(n*coderate)

num_training_iterations_conventional = 10000
num_training_iterations_rl_alt = 7000
num_training_iterations_rl_finetuning = 3000
training_batch_size = tf.constant(128, tf.int32)
rl_perturbation_var = 0.01
model_weights_path_conventional_training_tx = "awgn_autoencoder_weights_conventional_training_tx"
model_weights_path_conventional_training_rx = "awgn_autoencoder_weights_conventional_training_rx"
results_filename = "awgn_autoencoder_results"

##=======================================TX System================================================##
class TXSystemConventionalTraining(Model):
    def __init__(self, training):
        super().__init__()
        self._training = training
        qam_points = Constellation("qam", num_bits_per_symbol).points
        self.constellation = Constellation("custom",
                                           num_bits_per_symbol,
                                           points=qam_points,
                                           normalize=True,
                                           center=True)
        self.points_r = self.add_weight(shape=qam_points.shape, initializer="zeros")
        self.points_i = self.add_weight(shape=qam_points.shape, initializer="zeros")
        self.points_r.assign(tf.math.real(qam_points))
        self.points_i.assign(tf.math.imag(qam_points))
        self._mapper = Mapper(constellation=self.constellation)

    def call(self, batch_size, c, ebno_db):
        points = tf.complex(self.points_r, self.points_i)
        self.constellation.points = points
        x = self._mapper(c)
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

def load_weights_tx(model, model_weights_path):
    model(tf.cast(1, tf.int32), tf.zeros([1, n], dtype=tf.float32), tf.constant(10.0, tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    points = tf.complex(model.points_r, model.points_i)
    model.constellation.points = points
    print(f"\n TX weights have loaded from {model_weights_path}.")

def maybe_load_trainable_mapper_weights(model, model_weights_path):
    if not model_weights_path:
        print("[TX] 未提供 trainable mapper 权重路径，使用默认初始化")
        return
    if not os.path.isfile(model_weights_path):
        print(f"[TX] 权重文件不存在，跳过加载: {model_weights_path}")
        return
    load_weights_tx(model, model_weights_path)

def send_tensor(tensor: tf.Tensor, host: str, port: int):
    data = tf.io.serialize_tensor(tensor).numpy()
    size = len(data)
    print(f"准备发送 tensor，dtype={tensor.dtype}, shape={tensor.shape}, byte大小={size}")
    with socket.create_connection((host, port)) as s:
        s.sendall(size.to_bytes(8, byteorder='big'))
        s.sendall(data)
        print("发送完成，关闭连接。")

def send_frame(sock: socket.socket, header: dict, tensors: list):
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
        if y.ndim == 4 and y.shape[1] == 8:
            y = y[:, :4, :, :]
        y = (y * self.vae_scale).detach().cpu()
        return y

def resolve_vae_scale(vae_dir, default=1.0):
    cfg = os.path.join(vae_dir, "config.json")
    if not os.path.isfile(cfg):
        return float(default)
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return float(obj.get("scaling_factor", default))
    except Exception:
        return float(default)

def preprocess_frame(frame_bgr):
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = tfm(img_rgb).unsqueeze(0)
    return x

def select_top_indices(latents, total_m):
    B, C, H, W = latents.shape
    L = C * H * W
    flat = latents.reshape(B, L)
    energy = flat.abs()
    k_take = min(total_m, L)
    _, idx = torch.topk(energy, k=k_take, dim=1)
    return idx

def quantize_to_bits(latents, n_bits=10, target_k=7500, segments=1, q_low_pct=0.5, q_high_pct=99.5):
    B = latents.shape[0]
    flat = latents.reshape(B, -1)
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
    idx_all = select_top_indices(latents, total_M)
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

        seg_bits_t = ((q_int.unsqueeze(-1) >> bit_shifts) & 1).reshape(B, -1)
        seg_bits = seg_bits_t.to(dtype=torch.float32).cpu().numpy()
        if seg_bits.shape[1] < target_k:
            pad = np.zeros((B, target_k - seg_bits.shape[1]), dtype=np.float32)
            seg_bits = np.concatenate([seg_bits, pad], axis=1)
        elif seg_bits.shape[1] > target_k:
            seg_bits = seg_bits[:, :target_k]
        bits_segments.append(seg_bits)
        indices_segments.append(idx_slice.cpu().numpy().astype(np.int32))

    bits = np.stack(bits_segments, axis=0)
    indices = np.stack(indices_segments, axis=0)
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
    def _setup_common(c):
        try:
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return c

    if isinstance(cam_src, str) and ("nvarguscamerasrc" in cam_src or "!" in cam_src):
        cap = cv2.VideoCapture(cam_src, cv2.CAP_GSTREAMER)
        return _setup_common(cap)

    idx = None
    try:
        idx = int(cam_src)
    except Exception:
        idx = None

    if idx is not None:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            _setup_common(cap)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()

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

    cap = cv2.VideoCapture(cam_src)
    return _setup_common(cap)

def print_device_report_tx(model_tx, encoder):
    tf_gpus = tf.config.list_physical_devices('GPU')
    print(f"[Device][TX] TensorFlow GPUs: {[d.name for d in tf_gpus] if tf_gpus else 'None'}")
    print(f"[Device][TX] PyTorch CUDA available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapper_backend", choices=["trainable", "qam"], default="qam", 
                        help="TX mapper 后端：qam(快速基线) / trainable(权重优化)")
    parser.add_argument("--tx_weights", default=model_weights_path_conventional_training_tx, 
                        help="trainable mapper 权重文件路径")
    parser.add_argument("--dst", default="127.0.0.1", help="接收端 IP 地址")
    parser.add_argument("--port", type=int, default=5000, help="接收端端口")
    parser.add_argument("--live", action="store_true", help="启用实时摄像头模式")
    parser.add_argument("--cam_src", default="0", help="摄像头源")
    parser.add_argument("--vae_dir", default="../train_vae/vae", help="VAE 权重目录")
    parser.add_argument("--vae_trt_engine", type=str, default=None, help="TensorRT VAE encoder 引擎路径")
    parser.add_argument("--target_k", type=int, default=7500, help="每段比特数")
    parser.add_argument("--segments", type=int, default=1, help="分段数")
    parser.add_argument("--n_bits", type=int, default=10, help="量化位数")
    parser.add_argument("--ebno", type=float, default=6.0, help="实时模式固定 Eb/N0")
    parser.add_argument("--profile_tag", default="baseline_qam", help="实验标签")
    
    args = parser.parse_args()

    if args.mapper_backend == "trainable":
        model_tx = TXSystemConventionalTraining(training=True)
        maybe_load_trainable_mapper_weights(model_tx, args.tx_weights)
        print(f"[TX] 使用 trainable mapper，weights={args.tx_weights}")
    else:
        model_tx = TXSystemBaseline()
        print("[TX] 使用固定 QAM mapper（快速基线）")

    encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
    channel = AWGN()
    print_device_report_tx(model_tx, encoder)

    if args.live:
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
                print(f"[TRT][TX] 已启用 TRT encoder")
            except Exception as e_trt:
                print(f"[TRT][TX] TRT encoder 初始化失败，回退到 PyTorch: {e_trt}")

        if not use_trt_encoder:
            vae_device = "cuda" if torch.cuda.is_available() else "cpu"
            vae = AutoencoderKL.from_pretrained(args.vae_dir, local_files_only=True).to(vae_device).eval()
            vae_scale = getattr(vae.config, "scaling_factor", vae_scale)

        print("进入实时摄像头推流模式，按 Ctrl+C 结束")
        print(f"[PROFILE][TX] tag={args.profile_tag}, mapper={args.mapper_backend}, ebno={args.ebno}")

        sock = None
        while True:
            if sock is None:
                try:
                    sock = socket.create_connection((args.dst, args.port))
                    print(f"已连接到 {args.dst}:{args.port}")
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
                    latents = trt_encoder.encode(x.detach().cpu().numpy().astype(np.float32))
                else:
                    x = x.to(vae.device)
                    latents = vae.encode(x).latent_dist.mean * vae_scale

            bits, meta, indices = quantize_to_bits(latents, n_bits=args.n_bits, target_k=args.target_k,
                                                    segments=args.segments)
            t1 = time.time()

            b_tf_all = tf.reshape(tf.convert_to_tensor(bits, dtype=tf.float32), [-1, args.target_k])
            batch_items = tf.shape(b_tf_all)[0]
            ebno_db_use = tf.fill([batch_items], float(args.ebno))
            no = ebnodb2no(ebno_db_use, num_bits_per_symbol, coderate)
            no = expand_to_rank(no, 2)
            c_all = encoder(b_tf_all)
            x_all = model_tx(batch_size=batch_items, c=c_all, ebno_db=ebno_db_use)
            x_ch_all = channel(x_all, no)

            tensors = tf.split(x_ch_all, num_or_size_splits=args.segments, axis=0)
            t2 = time.time()

            header = {
                "magic": "FRAME1",
                "profile_tag": args.profile_tag,
                "mapper_backend": args.mapper_backend,
                "n_bits": int(meta["n_bits"]),
                "fmin": float(meta["fmin"]),
                "fmax": float(meta["fmax"]),
                "latent_shape": list(map(int, meta["latent_shape"])),
                "segments": int(meta["segments"]),
                "m_per_seg": int(meta["m_per_seg"]),
                "indices": indices.tolist(),
            }
            try:
                send_frame(sock, header, tensors)
                t3 = time.time()
                total_ms = (t3 - t0) * 1000
                encode_ms = (t1 - t0) * 1000
                mod_ms = (t2 - t1) * 1000
                fps = 1000.0 / total_ms if total_ms > 0 else 0
                print(f"TX[{args.profile_tag}]: total={total_ms:.1f}ms, vae={encode_ms:.1f}ms, ldpc+map={mod_ms:.1f}ms, fps={fps:.1f}")
            except Exception as e:
                print(f"发送失败，重连: {e}")
                try:
                    sock.close()
                except:
                    pass
                sock = None
                time.sleep(0.2)

            time.sleep(0.05)

