# Architecture

## End-to-end data path

```text
camera/image
   -> VAE encoder
   -> latent importance selection (Top-K)
   -> quantization + frame metadata
   -> LDPC encoder
   -> QAM or learned mapper
   -> pilots / OFDM / USRP
   -> synchronization + channel estimation + equalization
   -> APP or neural soft demapper
   -> LDPC decoder
   -> dequantization + latent placement
   -> VAE decoder
   -> reconstructed image
```

The system deliberately preserves module boundaries. Conventional and learned
blocks use the same frame contract, which makes ablation and fallback possible.

## AI-infrastructure path

The deployment path is organized around four expensive stages: VAE encoding,
TX baseband, RX baseband, and VAE decoding. The research implementation uses:

- exported inference graphs and device-specific TensorRT engines for the VAE;
- TensorFlow graph execution for LDPC and mapping/demapping;
- warm-up before timing to separate compilation from steady-state execution;
- batching and reduced host/device conversions;
- explicit backend selection with a framework fallback;
- per-stage and end-to-end measurement rather than model-only latency.

TensorRT engines are not portable across arbitrary JetPack, CUDA, TensorRT, and
GPU combinations. This repository therefore keeps export code but excludes the
compiled engines.

## Learned physical layer

The repository contains two complementary implementations:

1. `edge_semcom.neural_phy` is a PyTorch encoder/channel-estimator/decoder stack
   for modular experiments.
2. `edge_semcom.neural_modem` and the end-to-end apps use neural modulation and
   soft demodulation alongside the Sionna LDPC chain.

The trainable constellation keeps modulation interpretable: learned complex
points can be plotted, exported, and used by the MATLAB lookup mapper. Radio
front-end residual models are similarly added on top of explicit pilot and
MMSE estimators instead of replacing the complete receiver with a black box.

## Network frame boundary

The camera demonstration sends JSON metadata followed by typed tensor payloads.
Metadata records the latent layout, quantization range, source image shape, and
experiment profile. Production use would require an authenticated, versioned
wire protocol; the research protocol is intended only for controlled networks.

## Validation tiers

| Tier | Purpose | Typical dependencies |
|---|---|---|
| Repository | syntax, metadata, result integrity | Python, pytest |
| Simulation | BER/BLER and component ablation | PyTorch, TensorFlow, Sionna |
| Edge | runtime and latency | Jetson, JetPack, TensorRT |
| OTA | real-channel behavior | two radios / endpoints, UHD or MATLAB SDR support |

Results from one tier should not be silently generalized to another.

