# Models and data

The repository excludes large or non-portable artifacts by design.

| Artifact | Suggested location | Git policy | Reason |
|---|---|---|---|
| VAE directory | `artifacts/vae/` | ignored | large third-party-derived weights |
| Neural PHY weights | `artifacts/neural-phy/` | ignored | generated checkpoint |
| Neural modem checkpoints | `artifacts/neural-modem/` | ignored | generated checkpoint |
| TensorRT engines | `artifacts/tensorrt/` | ignored | device/runtime specific |
| ONNX exports | `artifacts/onnx/` | ignored | generated and often large |
| RF recordings | `captures/` | ignored | large and environment-specific |
| Camera recordings | `captures/` | ignored | privacy and size |
| Generated plots/JSON | `results/generated/` | ignored | reproducible output |

For a public release, publish redistributable weights as versioned GitHub
Release assets or in a model registry. Each artifact should include:

- SHA-256 checksum and byte size;
- model/source license;
- training or conversion command;
- framework, CUDA, TensorRT, and JetPack versions where applicable;
- expected input/output shape and precision;
- the commit that produced it.

Do not publish private camera frames, identifiable recordings, secrets, or raw
RF captures without reviewing consent and spectrum/privacy constraints.

