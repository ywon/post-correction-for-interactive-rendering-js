# Post-Correction for Interactive Monte Carlo Denoising using the James-Stein Estimator

This repository contains the implementation and results from my master's thesis, **"Post-Correction for Interactive Monte Carlo Denoising using the James-Stein Estimator"**. 
By leveraging the James-Stein Estimator, this approach achieves effective post-correction, enhancing both image quality and performance in interactive rendering scenarios.

---
## Environment and Setup

This project was developed and tested using the following hardware and software setup:

### Hardware
- **GPU**: NVIDIA GeForce RTX 2080 Ti
- **CPU**: AMD Ryzen Threadripper 2990WX 32-Core Processor

### Software
- **Operating System**: Windows 10
- **WSL**: Windows Subsystem for Linux (Version 2)
- Docker Desktop

## Dataset and Results

### Dataset
The project requires a dataset containing the following buffers, which serve as input for the denoising process:

#### **Required Buffers**
- **`path`**: Contains the path-traced input image.
- **`mvec`**: Stores motion vectors for temporal accumulation.
- **`linearZ`**: Represents linearized depth values.
- **`pnFwidth`**: Holds pixel neighborhood width data for reconstruction.
- **`albedo`**: Albedo buffer for input pixels.
- **`normal`**: Surface normal information for each pixel.
- **`variance`**: Variance estimation of luminance.
- **`ref`**: Ground truth image.
- **`biased_image`**: The denoised image before post-correction.

  
All buffers except for **`biased_image`** were generated using NVIDIA's Falcor version 7. The **`biased_image`** buffer was created based on the official implementations provided by the respective method authors.
- **NVIDIA's Falcor version 7** : https://github.com/NVIDIAGameWorks/Falcor/tree/7.0
- **OptiX Denoiser** : https://github.com/NVIDIAGameWorks/Falcor/tree/7.0/Source/RenderPasses/OptixDenoiser
- **SVGF** : https://github.com/NVIDIAGameWorks/Falcor/tree/7.0/Source/RenderPasses/SVGFPass
- **NBG** : https://github.com/xmeng525/RealTimeDenoisingNeuralBilateralGrid


#### **Dataset Format**
The dataset should be organized in the following structure:
```bash
data
├── scene1
│ ├── path_0000.exr
│ ├── mvec_0000.exr
│ ├── linearZ_0000.exr
│ ├── pnFwidth_0000.exr
│ ├── albedo_0000.exr
│ ├── normal_0000.exr
│ ├── variance_0000.exr
│ ├── ref_0000.exr
│ ├── biased_image_0000.exr
│ ├── ...
├── scene2
├── ...
```

### Results
The results will be stored in the `results/` directory.

## Running the Project

To run this code, follow these steps:

1. **Build the Docker Image**
   Use the provided shell script to build the Docker environment:
   ```bash
   bash build_docker.sh
   ```
   
3. **Build Custom Operations**
   After the Docker image is built, run the following script to compile custom operations:
   ```bash
   bash build_customop.sh
   ```

4. **Run the Main Script**
   Finally, execute the main Python script:
   ```bash
   python scripts/main.py
   ```

## Copyright

This thesis is © 조예원(Yewon Cho), 2025. All rights reserved.  
This work was conducted at **Computer Graphics Lab** within **GIST(Gwangju Institute of Science and Technology)**.  

For institutional and research lab details, please visit:  
- [GIST(Gwangju Institute of Science and Technology)](https://www.gist.ac.kr/kr/main.html)  
- [Computer Graphics Lab](https://cglab.gist.ac.kr/)

Unauthorized distribution or use of this material is prohibited without prior consent.



