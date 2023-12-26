# Galaxy Data Analysis with CUDA

## Overview
This CUDA project analyzes galaxy data, focusing on the angular separation between real and simulated galaxies. It uses parallel processing capabilities of CUDA for efficient computation of histograms representing angular separations.

## Features
- Calculation of angular separation between galaxies.
- Histogram generation for real-real, real-simulated, and simulated-simulated galaxy pairs.
- Efficient parallel processing using CUDA kernels.

## Requirements
- CUDA Toolkit (compatible with the code)
- A CUDA-capable GPU
- Windows operating system (due to the use of Windows.h)

## Installation
Ensure you have the CUDA Toolkit installed and configured on your system.

## Usage
- Compile the code with nvcc: `nvcc -o galaxy_analysis galaxyfinal.cu`
- Run the executable: `./galaxy_analysis`

## Code Structure
- `calculateAngularSeparation`: Function to calculate the angular separation between two points.
- `calculateHistograms`: CUDA kernel to calculate the histograms of angular separations.
- Main function: Handles data reading, memory allocation, kernel execution, and result calculation.

## Performance
The code is optimized for performance using CUDA's parallel processing capabilities, significantly reducing computation time for large datasets.

## Contributing
Contributions to enhance the code, improve performance, or extend functionality are welcome.

## Acknowledgements
Mention any inspirations, data sources, or collaborations.
