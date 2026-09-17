#!/usr/bin/env python3
"""
Train a learnable 16-QAM constellation for the MATLAB OFDM chain.

This version trains against a frame-level OFDM receiver that includes:
- data/pilot/guard/DC subcarrier mapping
- IFFT + cyclic prefix
- residual carrier-frequency-offset surrogate
- AWGN
- FFT
- LS pilot channel estimation + linear interpolation
- nearest-neighbor hard decision on equalized data tones

The optimization uses a straight-through hard-decision surrogate:
forward pass: hard nearest-neighbor bits
backward pass: soft symbol posterior gradients

Output .mat fields:
  constellation   complex64 [M, 1]
  bit_table       uint8 [M, k], MSB-first
  M               int32 scalar
  k               int32 scalar
  avg_power       float32 scalar
  source          string
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.get_logger().setLevel("ERROR")


@dataclass
class TrainConfig:
    m: int = 16
    n_fft: int = 256
    n_cp: int = 16
    n_pilot_seed: int = 16
    n_guard: int = 4
    n_dc: int = 2
    frame_size: int = 10
    snr_min: float = 16.0
    snr_max: float = 18.0
    cfo_max: float = 0.05
    phase_slope_std: float = 2e-4
    batch_size: int = 32
    steps: int = 2500
    lr: float = 2e-3
    eval_every: int = 100
    eval_batches: int = 20
    power_target: float = 10.0
    temp: float = 0.35
    loss_type: str = "bce"
    train_objective: str = "relative"
    advantage_margin: float = 2e-3
    lambda_power: float = 3e-3
    lambda_center: float = 1e-3
    lambda_spread: float = 8e-3
    stage_split: float = 0.7
    stage_relax_ratio: float = 0.3
    init_noise_std: float = 0.12
    init_mode: str = "affine"
    init_escape_strength: float = 0.0
    init_escape_mode: str = "affine"
    init_fresh_scale: float = 1.0
    total_steps: int = 2500


def bits_table_binary(m: int) -> np.ndarray:
    k = int(round(math.log2(m)))
    idx = np.arange(m, dtype=np.int32)
    bits = ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)
    return bits


def standard_qam_binary_indexed(m: int) -> np.ndarray:
    n = int(round(math.sqrt(m)))
    if n * n != m:
        raise ValueError("M must be square")

    k = int(round(math.log2(m)))
    if (1 << k) != m:
        raise ValueError("M must be power of two")

    if k % 2 != 0:
        raise ValueError("M must be 2^(2p) for square QAM")

    ka = k // 2
    levels = np.arange(-(n - 1), n, 2, dtype=np.float32)

    b = np.arange(m, dtype=np.int32)
    i_bin = b >> ka
    q_bin = b & ((1 << ka) - 1)

    i_gray = i_bin ^ (i_bin >> 1)
    q_gray = q_bin ^ (q_bin >> 1)

    pts = levels[i_gray] + 1j * levels[q_gray]
    return pts.astype(np.complex64)


class ConstellationTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.m = cfg.m
        self.k = int(round(math.log2(self.m)))

        if (1 << self.k) != self.m:
            raise ValueError("M must be power of two")

        self._build_layout()
        self._build_interp_matrix()

        self.step_counter = 0
        base = standard_qam_binary_indexed(self.m)
        self.base = tf.constant(base, dtype=tf.complex64)
        self.bits_lut = tf.constant(bits_table_binary(self.m), dtype=tf.float32)

        self.pr = tf.Variable(np.real(base).astype(np.float32), trainable=True)
        self.pi = tf.Variable(np.imag(base).astype(np.float32), trainable=True)

    def _build_layout(self):
        n = self.cfg.n_fft
        ng = self.cfg.n_guard
        np_seed = self.cfg.n_pilot_seed

        dc_sc = np.array([n // 2 - 1, n // 2], dtype=np.int32)
        eff = np.arange(ng, n - ng, dtype=np.int32)
        eff = np.setdiff1d(eff, dc_sc)
        stride = int(np.ceil(len(eff) / np_seed))
        pilot_sc = eff[::stride]

        guard_sc = np.concatenate([np.arange(ng), np.arange(n - ng, n)]).astype(np.int32)
        all_sc = np.arange(n, dtype=np.int32)

        data_sc = np.setdiff1d(all_sc, guard_sc)
        data_sc = np.setdiff1d(data_sc, pilot_sc)
        data_sc = np.setdiff1d(data_sc, dc_sc)

        self.guard_sc = guard_sc
        self.dc_sc = dc_sc
        self.pilot_sc = pilot_sc
        self.data_sc = data_sc
        self.n_pilot = len(pilot_sc)
        self.n_data_per_sym = len(data_sc)
        self.n_data_total = self.n_data_per_sym * self.cfg.frame_size

        data_basis = np.zeros((self.n_data_per_sym, n), dtype=np.complex64)
        data_basis[np.arange(self.n_data_per_sym), self.data_sc] = 1.0 + 0.0j
        pilot_basis = np.zeros((self.n_pilot, n), dtype=np.complex64)
        pilot_basis[np.arange(self.n_pilot), self.pilot_sc] = 1.0 + 0.0j

        pilot_pos = self.pilot_sc.astype(np.float32)
        all_pos = np.arange(n, dtype=np.float32)
        interp = np.zeros((n, self.n_pilot), dtype=np.float32)
        for j in range(self.n_pilot):
            basis = np.zeros(self.n_pilot, dtype=np.float32)
            basis[j] = 1.0
            interp[:, j] = np.interp(all_pos, pilot_pos, basis)

        self.data_basis = tf.constant(data_basis)
        self.pilot_basis = tf.constant(pilot_basis)
        self.interp_w = tf.constant(interp, dtype=tf.float32)

    def _build_interp_matrix(self):
        return

    def randomize_init(self, noise_std: float, escape_strength: float = 0.0, escape_mode: str = "affine"):
        escape_strength = float(np.clip(escape_strength, 0.0, 1.0))

        base = self.base
        base_centered = base - tf.reduce_mean(base)
        base_pow = tf.reduce_mean(tf.abs(base_centered) ** 2)

        if escape_mode == "random":
            rnd = tf.complex(
                tf.random.normal(tf.shape(base_centered), dtype=tf.float32),
                tf.random.normal(tf.shape(base_centered), dtype=tf.float32),
            )
            rnd = rnd - tf.reduce_mean(rnd)
            rnd_pow = tf.reduce_mean(tf.abs(rnd) ** 2)
            rnd = rnd * tf.cast(tf.sqrt((base_pow + 1e-8) / (rnd_pow + 1e-8)), tf.complex64)
            init_pts = (1.0 - escape_strength) * base_centered + escape_strength * rnd
        else:
            # Structure-preserving escape: global affine warp keeps index topology mostly intact.
            x = tf.math.real(base_centered)
            y = tf.math.imag(base_centered)

            max_rot = 0.35 * escape_strength
            theta = tf.random.uniform([], -max_rot, max_rot, dtype=tf.float32)

            sx = 1.0 + tf.random.normal([], stddev=0.25 * escape_strength, dtype=tf.float32)
            sy = 1.0 + tf.random.normal([], stddev=0.25 * escape_strength, dtype=tf.float32)
            shear = tf.random.normal([], stddev=0.20 * escape_strength, dtype=tf.float32)

            x1 = sx * x + shear * y
            y1 = sy * y

            c = tf.cos(theta)
            s = tf.sin(theta)
            xr = c * x1 - s * y1
            yr = s * x1 + c * y1
            init_pts = tf.complex(xr, yr)

            init_pts = init_pts - tf.reduce_mean(init_pts)
            p = tf.reduce_mean(tf.abs(init_pts) ** 2)
            init_pts = init_pts * tf.cast(tf.sqrt((base_pow + 1e-8) / (p + 1e-8)), tf.complex64)

        noise_scale = noise_std * (1.0 + escape_strength)
        noise_r = tf.random.normal(tf.shape(self.pr), stddev=noise_scale, dtype=tf.float32)
        noise_i = tf.random.normal(tf.shape(self.pi), stddev=noise_scale, dtype=tf.float32)

        self.pr.assign(tf.math.real(init_pts) + noise_r)
        self.pi.assign(tf.math.imag(init_pts) + noise_i)

    def fresh_random_init(self, scale: float = 1.0, noise_std: float = 0.0):
        # Start from a fully random constellation (not centered on standard QAM points).
        scale = float(max(scale, 1e-6))
        z = tf.complex(
            tf.random.normal([self.m], dtype=tf.float32),
            tf.random.normal([self.m], dtype=tf.float32),
        )
        z = z - tf.reduce_mean(z)
        p = tf.reduce_mean(tf.abs(z) ** 2)
        z = z * tf.cast(tf.sqrt(tf.constant(self.cfg.power_target, tf.float32) / (p + 1e-8)), tf.complex64)
        z = z * tf.complex(tf.constant(scale, tf.float32), tf.constant(0.0, tf.float32))

        if noise_std > 0:
            z = z + tf.complex(
                tf.random.normal([self.m], stddev=noise_std, dtype=tf.float32),
                tf.random.normal([self.m], stddev=noise_std, dtype=tf.float32),
            )

        self.pr.assign(tf.math.real(z))
        self.pi.assign(tf.math.imag(z))

    def points(self) -> tf.Tensor:
        return tf.complex(self.pr, self.pi)

    def map_bits(self, bits: tf.Tensor, pts: tf.Tensor) -> tf.Tensor:
        w = tf.constant([1 << i for i in range(self.k - 1, -1, -1)], dtype=tf.int32)
        idx = tf.reduce_sum(tf.cast(bits, tf.int32) * w[None, :], axis=1)
        return tf.gather(pts, idx)

    def _sample_impairments(self, batch_size: tf.Tensor, seq_len: int):
        cfo = tf.random.uniform([batch_size, 1], minval=-self.cfg.cfo_max, maxval=self.cfg.cfo_max, dtype=tf.float32)
        phi0 = tf.random.uniform([batch_size, 1], minval=-np.pi, maxval=np.pi, dtype=tf.float32)
        phase_slope = tf.random.normal([batch_size, 1], stddev=self.cfg.phase_slope_std, dtype=tf.float32)
        z_re = tf.random.normal([batch_size, seq_len], dtype=tf.float32)
        z_im = tf.random.normal([batch_size, seq_len], dtype=tf.float32)
        return cfo, phi0, phase_slope, z_re, z_im

    def _sample_channel_state(self, batch_size: tf.Tensor, seq_len: int):
        cfo, phi0, phase_slope, z_re, z_im = self._sample_impairments(batch_size, seq_len)
        snr_db = tf.random.uniform([batch_size, 1], minval=self.cfg.snr_min, maxval=self.cfg.snr_max, dtype=tf.float32)
        return {
            "cfo": cfo,
            "phi0": phi0,
            "phase_slope": phase_slope,
            "z_re": z_re,
            "z_im": z_im,
            "snr_db": snr_db,
        }

    def _build_tx_grid(self, bits: tf.Tensor, pts: tf.Tensor):
        b = tf.shape(bits)[0]
        f = tf.shape(bits)[1]

        weights = tf.constant([1 << i for i in range(self.k - 1, -1, -1)], dtype=tf.int32)
        idx_bin = tf.reduce_sum(tf.cast(bits, tf.int32) * weights[None, None, None, :], axis=-1)
        x_data = tf.gather(pts, idx_bin)

        peak = tf.reduce_max(tf.abs(x_data), axis=[1, 2])
        pilot_amp = peak / tf.sqrt(tf.constant(2.0, dtype=tf.float32))
        pilot_sym = tf.complex(pilot_amp, pilot_amp)

        data_grid = tf.einsum("bfd,dn->bfn", x_data, self.data_basis)
        pilot_vals = tf.cast(tf.reshape(pilot_sym, [b, 1, 1]), tf.complex64) * tf.ones([b, f, self.n_pilot], dtype=tf.complex64)
        pilot_grid = tf.einsum("bfp,pn->bfn", pilot_vals, self.pilot_basis)
        tx_grid = data_grid + pilot_grid
        return tx_grid, pilot_sym, x_data

    def _through_ofdm_chain(self, tx_grid: tf.Tensor, pilot_sym: tf.Tensor, channel_state=None):
        b = tf.shape(tx_grid)[0]
        f = self.cfg.frame_size
        n = self.cfg.n_fft
        ncp = self.cfg.n_cp

        x_t = tf.signal.ifft(tx_grid)
        x_cp = tf.concat([x_t[:, :, -ncp:], x_t], axis=2)
        x_seq = tf.reshape(x_cp, [b, f * (n + ncp)])

        if channel_state is None:
            channel_state = self._sample_channel_state(b, f * (n + ncp))

        cfo = channel_state["cfo"]
        phi0 = channel_state["phi0"]
        phase_slope = channel_state["phase_slope"]
        z_re = channel_state["z_re"]
        z_im = channel_state["z_im"]
        sample_idx = tf.cast(tf.range(f * (n + ncp))[None, :], tf.float32)
        phase = phi0 + (2.0 * np.pi * cfo * sample_idx / tf.cast(n, tf.float32)) + phase_slope * sample_idx
        x_seq = x_seq * tf.exp(tf.complex(tf.zeros_like(phase), phase))

        snr_db = channel_state["snr_db"]
        snr_lin = tf.pow(10.0, snr_db / 10.0)
        p_sig = tf.reduce_mean(tf.abs(x_seq) ** 2, axis=1, keepdims=True)
        n0 = p_sig / snr_lin
        noise = tf.complex(z_re * tf.sqrt(n0 / 2.0), z_im * tf.sqrt(n0 / 2.0))
        y_seq = x_seq + noise

        y_cp = tf.reshape(y_seq, [b, f, n + ncp])
        y_t = y_cp[:, :, ncp:]
        y_grid = tf.signal.fft(y_t)

        y_p = tf.gather(y_grid, self.pilot_sc, axis=2)
        pilot_ref = tf.cast(tf.reshape(pilot_sym, [b, 1, 1]), y_p.dtype)
        h_p = y_p / pilot_ref

        h_r = tf.einsum("np,bfp->bfn", self.interp_w, tf.math.real(h_p))
        h_i = tf.einsum("np,bfp->bfn", self.interp_w, tf.math.imag(h_p))
        h_est = tf.complex(h_r, h_i)

        y_eq = y_grid / (h_est + tf.complex(tf.constant(1e-4, tf.float32), tf.constant(0.0, tf.float32)))
        y_data = tf.gather(y_eq, self.data_sc, axis=2)
        return y_data

    def _decision_loss(self, y_data: tf.Tensor, bits_true: tf.Tensor, pts: tf.Tensor) -> tf.Tensor:
        d2 = tf.abs(tf.expand_dims(y_data, axis=-1) - tf.reshape(pts, [1, 1, 1, self.m])) ** 2
        logits = -d2 / tf.constant(self.cfg.temp, dtype=tf.float32)
        p_sym = tf.nn.softmax(logits, axis=-1)

        hard_idx = tf.argmin(d2, axis=-1, output_type=tf.int32)
        hard_bits = tf.gather(self.bits_lut, hard_idx)
        soft_bits = tf.tensordot(p_sym, self.bits_lut, axes=[[-1], [0]])
        pred_bits = soft_bits + tf.stop_gradient(hard_bits - soft_bits)

        if self.cfg.loss_type == "bce":
            p1 = tf.clip_by_value(pred_bits, 1e-6, 1.0 - 1e-6)
            bce = -(bits_true * tf.math.log(p1) + (1.0 - bits_true) * tf.math.log(1.0 - p1))
            return tf.reduce_mean(bce)

        return tf.reduce_mean(tf.square(pred_bits - bits_true))

    def geometry_reg(self, pts: tf.Tensor) -> tf.Tensor:
        pwr = tf.reduce_mean(tf.abs(pts) ** 2)
        reg_p = tf.square(pwr - tf.constant(self.cfg.power_target, tf.float32))

        ctr = tf.reduce_mean(pts)
        reg_c = tf.square(tf.math.real(ctr)) + tf.square(tf.math.imag(ctr))

        d = tf.abs(tf.expand_dims(pts, 0) - tf.expand_dims(pts, 1))
        d = d + tf.eye(self.m, dtype=d.dtype) * 1e9
        min_d = tf.reduce_min(d)
        reg_s = tf.nn.relu(1.0 - tf.cast(min_d, tf.float32))

        # Two-stage regularization schedule: relax in stage-1, tighten in stage-2
        progress = min(1.0, self.step_counter / max(1, self.cfg.total_steps))
        if progress < self.cfg.stage_split:
            lambda_spread_eff = self.cfg.lambda_spread * self.cfg.stage_relax_ratio
        else:
            lambda_spread_eff = self.cfg.lambda_spread
        
        return self.cfg.lambda_power * reg_p + self.cfg.lambda_center * reg_c + lambda_spread_eff * reg_s

    def train_step(self, opt: tf.keras.optimizers.Optimizer) -> tuple[float, float]:
        bits = tf.cast(
            tf.random.uniform([self.cfg.batch_size, self.cfg.frame_size, self.n_data_per_sym, self.k], minval=0, maxval=2, dtype=tf.int32),
            tf.float32,
        )

        with tf.GradientTape() as tape:
            pts_l = self.points()
            tx_grid, pilot_sym, _ = self._build_tx_grid(bits, pts_l)
            channel_state = self._sample_channel_state(self.cfg.batch_size, self.cfg.frame_size * (self.cfg.n_fft + self.cfg.n_cp))
            y_data = self._through_ofdm_chain(tx_grid, pilot_sym, channel_state)

            loss_decision = self._decision_loss(y_data, bits, pts_l)
            if self.cfg.train_objective == "relative":
                tx_grid_q, pilot_sym_q, _ = self._build_tx_grid(bits, self.base)
                y_data_q = self._through_ofdm_chain(tx_grid_q, pilot_sym_q, channel_state)
                loss_qam = self._decision_loss(y_data_q, bits, self.base)
                # Learned must be better than baseline by margin.
                loss_decision = tf.nn.relu(loss_decision - loss_qam + tf.constant(self.cfg.advantage_margin, tf.float32))

            loss_reg = self.geometry_reg(pts_l)
            loss = loss_decision + loss_reg

        grads = tape.gradient(loss, [self.pr, self.pi])
        opt.apply_gradients(zip(grads, [self.pr, self.pi]))

        self.step_counter += 1
        return float(loss.numpy()), float(loss_decision.numpy())

    def eval_hard_ber(self, pts: tf.Tensor, num_batches: int) -> float:
        err = 0
        total = 0
        for _ in range(num_batches):
            bits = tf.cast(
                tf.random.uniform([self.cfg.batch_size, self.cfg.frame_size, self.n_data_per_sym, self.k], minval=0, maxval=2, dtype=tf.int32),
                tf.float32,
            )
            tx_grid, pilot_sym, _ = self._build_tx_grid(bits, pts)
            y_data = self._through_ofdm_chain(tx_grid, pilot_sym)

            d2 = tf.abs(tf.expand_dims(y_data, axis=-1) - tf.reshape(pts, [1, 1, 1, self.m])) ** 2
            hard_idx = tf.argmin(d2, axis=-1, output_type=tf.int32)
            bh = tf.gather(self.bits_lut, hard_idx)

            err += int(tf.reduce_sum(tf.cast(tf.not_equal(bh, bits), tf.int32)).numpy())
            total += int(self.cfg.batch_size * self.cfg.frame_size * self.n_data_per_sym * self.k)
        return err / max(total, 1)

    def export_mat(self, out_mat: str):
        try:
            from scipy.io import savemat
        except Exception as e:
            raise RuntimeError(f"scipy is required for .mat export: {e}")

        pts = self.points().numpy().astype(np.complex64)
        bit_table = bits_table_binary(self.m)

        savemat(
            out_mat,
            {
                "constellation": pts.reshape(-1, 1),
                "bit_table": bit_table.astype(np.uint8),
                "M": np.array([self.m], dtype=np.int32),
                "k": np.array([self.k], dtype=np.int32),
                "avg_power": np.array([np.mean(np.abs(pts) ** 2)], dtype=np.float32),
                "source": np.array(["train_learned_16qam_matlab_ofdm"], dtype=object),
            },
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train learnable 16-QAM for MATLAB OFDM hard decision")
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--n_fft", type=int, default=256)
    p.add_argument("--n_cp", type=int, default=16)
    p.add_argument("--n_pilot_seed", type=int, default=16)
    p.add_argument("--n_guard", type=int, default=4)
    p.add_argument("--n_dc", type=int, default=2)
    p.add_argument("--frame_size", type=int, default=10)
    p.add_argument("--snr_min", type=float, default=16.0)
    p.add_argument("--snr_max", type=float, default=18.0)
    p.add_argument("--cfo_max", type=float, default=0.05)
    p.add_argument("--phase_slope_std", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--loss_type", type=str, default="bce", choices=["mse", "bce"])
    p.add_argument("--train_objective", type=str, default="relative", choices=["absolute", "relative"])
    p.add_argument("--advantage_margin", type=float, default=2e-3)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--out_mat", type=str, default="trained_constellation_16qam_ofdm.mat")
    p.add_argument("--init_mode", type=str, default="affine", choices=["standard", "affine", "random_fresh"])
    p.add_argument("--init_noise_std", type=float, default=0.12)
    p.add_argument("--init_escape_strength", type=float, default=0.0)
    p.add_argument("--init_escape_mode", type=str, default="affine", choices=["affine", "random"])
    p.add_argument("--init_fresh_scale", type=float, default=1.0)
    p.add_argument("--lambda_spread", type=float, default=8e-3)
    p.add_argument("--stage_split", type=float, default=0.7)
    p.add_argument("--stage_relax_ratio", type=float, default=0.3)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        m=args.M,
        n_fft=args.n_fft,
        n_cp=args.n_cp,
        n_pilot_seed=args.n_pilot_seed,
        n_guard=args.n_guard,
        n_dc=args.n_dc,
        frame_size=args.frame_size,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        cfo_max=args.cfo_max,
        phase_slope_std=args.phase_slope_std,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        loss_type=args.loss_type,
        train_objective=args.train_objective,
        advantage_margin=args.advantage_margin,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        lambda_spread=args.lambda_spread,
        stage_split=args.stage_split,
        stage_relax_ratio=args.stage_relax_ratio,
        init_mode=args.init_mode,
        init_noise_std=args.init_noise_std,
        init_escape_strength=args.init_escape_strength,
        init_escape_mode=args.init_escape_mode,
        init_fresh_scale=args.init_fresh_scale,
        total_steps=args.steps,
    )

    tr = ConstellationTrainer(cfg)
    if cfg.init_mode == "standard":
        pass
    elif cfg.init_mode == "random_fresh":
        tr.fresh_random_init(cfg.init_fresh_scale, cfg.init_noise_std)
    else:
        tr.randomize_init(cfg.init_noise_std, cfg.init_escape_strength, cfg.init_escape_mode)
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

    print(
        f"config: M={cfg.m}, loss_type={cfg.loss_type}, objective={cfg.train_objective}, "
        f"snr=[{cfg.snr_min},{cfg.snr_max}] dB, init_mode={cfg.init_mode}"
    )

    best_learned_ber = 1e9
    best_gap = 1e9
    best_r = tr.pr.numpy().copy()
    best_i = tr.pi.numpy().copy()
    best_gap_r = tr.pr.numpy().copy()
    best_gap_i = tr.pi.numpy().copy()

    for step in range(cfg.steps):
        loss, decision_loss = tr.train_step(opt)

        if (step % 50) == 0:
            print(f"step={step:5d} loss={loss:.3e} decision_loss={decision_loss:.3e}")

        if ((step + 1) % cfg.eval_every == 0) or (step == cfg.steps - 1):
            ber_l = tr.eval_hard_ber(tr.points(), cfg.eval_batches)
            ber_q = tr.eval_hard_ber(tr.base, cfg.eval_batches)
            gap = ber_l - ber_q
            progress_pct = 100.0 * (step + 1) / cfg.steps
            print(f"eval@{step+1:5d} ({progress_pct:5.1f}%): BER_learned={ber_l:.6e} BER_qam={ber_q:.6e} gap={gap:+.3e}")

            if ber_l < best_learned_ber:
                best_learned_ber = ber_l
                best_r = tr.pr.numpy().copy()
                best_i = tr.pi.numpy().copy()

            if gap < best_gap:
                best_gap = gap
                best_gap_r = tr.pr.numpy().copy()
                best_gap_i = tr.pi.numpy().copy()

    tr.pr.assign(best_r)
    tr.pi.assign(best_i)
    ber_l = tr.eval_hard_ber(tr.points(), cfg.eval_batches)
    ber_q = tr.eval_hard_ber(tr.base, cfg.eval_batches)
    print(f"final(best learned): BER_learned={ber_l:.6e} BER_qam={ber_q:.6e} gap={ber_l - ber_q:+.3e}")

    tr.pr.assign(best_gap_r)
    tr.pi.assign(best_gap_i)
    ber_l_gap = tr.eval_hard_ber(tr.points(), cfg.eval_batches)
    ber_q_gap = tr.eval_hard_ber(tr.base, cfg.eval_batches)
    print(f"final(best gap): BER_learned={ber_l_gap:.6e} BER_qam={ber_q_gap:.6e} gap={ber_l_gap - ber_q_gap:+.3e}")

    tr.pr.assign(best_r)
    tr.pi.assign(best_i)
    tr.export_mat(args.out_mat)
    print(f"saved: {args.out_mat}")


if __name__ == "__main__":
    main()

