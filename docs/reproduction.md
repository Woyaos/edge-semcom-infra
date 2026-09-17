# Reproduction guide

## 1. Choose a validation tier

Start with repository checks, then simulation. Jetson and OTA steps are separate
because their package versions are constrained by vendor runtimes and hardware.

### Repository checks

```bash
python -m pip install -e ".[dev]"
pytest
python -m compileall -q apps src training evaluation tools tests
```

### Simulation

```bash
python -m pip install -e ".[simulation,dev]"
python training/train_neural_phy.py --epochs 40 --seed 42 \
  --save_dir artifacts/neural-phy
python evaluation/compare_phy_layers.py \
  --neural_model artifacts/neural-phy/best_model.pt \
  --snr_min 0 --snr_max 15 \
  --output_dir results/generated/phy
```

The TensorFlow/Sionna API changed across releases. The dependency range records
the family used by the code, but a reproducible archival run should also record
the resolved environment with `python -m pip freeze`.

## 2. Constellation experiment

```bash
python training/train_constellation.py \
  --M 16 --n_fft 256 --n_cp 16 --n_pilot_seed 16 \
  --snr_min 16 --snr_max 18 --steps 2500 \
  --out_mat artifacts/trained_constellation_16qam_ofdm.mat
```

Copy the resulting MAT file beside the scripts in `matlab/`, then load it with:

```matlab
model = load_learned_constellation16("trained_constellation_16qam_ofdm.mat");
symbols = learned_qam_mod(bits, model);
bits_hat = learned_qam_hard_demod(symbols, model, numel(bits));
```

## 3. Jetson deployment

1. Install the JetPack-supported CUDA, cuDNN, TensorRT, PyTorch, and TensorFlow
   builds. Do not replace vendor packages with generic wheels blindly.
2. Place the VAE directory and optional checkpoints under `artifacts/`.
3. Export an ONNX model with `tools/export_vae_onnx.py`.
4. Build TensorRT engines on the target Jetson. Engines built elsewhere may be
   incompatible even if the model is identical.
5. Run several warm-up frames before collecting steady-state latency.
6. Record power mode, clock settings, precision, input shape, and software
   versions with every result.

The optimized application variants document the deployment-oriented execution
path. The full apps retain additional backends and experiment switches.

## 4. Controlled network demo

On the receiver:

```bash
python apps/receiver.py --host 0.0.0.0 --port 5000 --live --reconstruct \
  --vae_dir artifacts/vae --save_every 10
```

On the transmitter:

```bash
python apps/transmitter.py --dst RECEIVER_IP --port 5000 --live \
  --vae_dir artifacts/vae --cam_src 0 --profile_tag qam-baseline
```

Use only a trusted lab network. The research framing protocol is not encrypted
or authenticated.

## 5. OTA experiment

An OTA run additionally requires two configured radio endpoints, compatible
UHD/SDR support, agreed center frequency/sample rate/gain, and compliance with
local spectrum regulations. Begin with a cable connection and suitable
attenuation before radiating. Record clock source, antenna/cable path, PA gain,
pilot layout, and capture duration.

## 6. Comparing with archived results

Reference JSON files under `results/` are intentionally small. Compare new runs
only after matching code path, seed, SNR definition, block length, batch size,
precision, and hardware. See `docs/results.md` for interpretation caveats.

