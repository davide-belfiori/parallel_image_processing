# Parallel image processing
This application provides a comparison between sequential and parallel approach to image processing using OpenCL for GPU computing.

Implemented algorithms:
- Rotation
- Edge Filtering by 2D Convolution

## Build the application

First clone this repository:

```
git clone https://github.com/davide-belfiori/parallel_image_processing
cd parallel_image_processing
```
Add 2 environment variables to your system, one named `OCL_INCLUDE` containing path to your OpenCL **headers** directory, and one named `OCL_LIB` containing path to the OpenCL **library** directory.
Example for NVIDIA platform:
```
OCL_INCLUDE = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include
OCL_LIB = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64
```
Finally run `make` command inside the repository folder.

## Usage

#### Image rotation:
```
imgp rotate <image_to_rotate>
```
Use `--deg` option to specify rotation degrees, by default this value is 90°.

Use `-p` option for parallel execution, and `--save <filename>` to save rotation result.

#### Image filtering:

```
imgp filter <image_to_filter>
```
Use `-p` option for parallel execution, and `--save <filename>` to save filter result.

This command perform an approximation of Sobel filter using a 3x3 kernel filter


> Run `imgp -h` for the full command and options list
