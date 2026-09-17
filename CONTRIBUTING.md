# Contributing

Thanks for improving EdgeSemCom Infra. Please keep changes focused and explain
which validation tier they affect: simulation, Jetson deployment, or USRP OTA.

1. Create a small branch and include a reproducible command or test.
2. Do not commit checkpoints, TensorRT engines, RF captures, videos, credentials,
   local IP addresses, or generated result directories.
3. Run `pytest` and `python -m compileall -q apps src training evaluation tools tests`.
4. For numerical changes, include the configuration, random seed, hardware, and
   both the old and new metric files.
5. Keep third-party code out of the tree unless its license and attribution are
   documented.

Hardware-only changes are welcome even when CI cannot execute them. In that
case, include device model, JetPack/UHD versions, and a concise test log.

