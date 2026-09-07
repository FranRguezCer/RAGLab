# Raspberry Pi AI Camera

The AI Camera uses the Sony IMX500 sensor. Its image signal processor creates an input tensor and inference runs inside the camera, returning output tensors to the Raspberry Pi. Install the runtime firmware with `sudo apt install imx500-all`; host-side software still post-processes inference output.
