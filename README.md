# CUDA Programming
This repository is a personal space for practicing CUDA programming and experimenting with GPU development projects.<br>
While it is not primarily designed as a learning resource for others, anyone interested may find the examples useful.<br>

---
## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Folder Structure](#folder-structure)
- [Dependencies](#dependencies)
- [How to Use](#how-to-use)
- [License](#license)

---
## Overview

This repository includes a collection of CUDA projects and exercises for:

- Understanding CUDA threads, blocks, and grids.
- Practicing memory management techniques (shared, global, and constant memory).
- Experimenting with performance optimizations (e.g., memory coalescing).
- Building small applications such as matrix multiplication and basic image processing.

---
## Getting Started

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/mohammed1thabet/CUDA_Programming.git
cd CUDA_Programming
```
Ensure that you have a compatible NVIDIA GPU and the CUDA toolkit installed on your system.

---
## Folder Structure

```plaintext
CUDA_Programming/
├── 001_project_name/    # Folder for the first project, includes source code and VS Studio project.
│   ├── 001_project_name.vcxproj  # Visual Studio project file.
│   ├── 001_project_name.sln  # Visual Studio solution file.
│   ├── kernel.cu        # Main CUDA source file.
│   ├── file1.cpp        # cpp file.
│   ├── file1.hpp        # hpp file.
│   ├── file2.cpp        # cpp file.
│   ├── file2.hpp        # hpp file.
.   .
.   .
.   .
├── 002_project_name/    # Folder for the second project.
│   ├── main.cu
│   ├── helper.cpp
│   ├── project.vcxproj
├── ...                  # Additional numbered project folders.
└── README.md            # Repository documentation.
```
Each project folder contains the following:
- CUDA source files (`.cu`).
- may contain C++ source and header files (`.cpp`, `.hpp`) for supporting code.
- A Visual Studio project file (`.vcxproj`) and solution file (`.sln`) for building the project.
---

## Dependencies

- NVIDIA GPU with CUDA support.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (latest version recommended).
- GCC or Clang compiler.
- Optional: [CMake](https://cmake.org/) for building more complex projects.

---

## How to Use

1. **Navigate to a Project:**
   Go to the folder of the project you want to work on:
   ```bash
   cd 001_project_name
   ```

2. **Build the Project:**
   - If you're using Visual Studio, open the `.vcxproj` file and build the project.
   - If you prefer the command line, compile the `.cu` file using `nvcc`:
     ```bash
     nvcc main.cu -o project_name
     ./project_name
     ```

3. **Experiment:**
   Modify the source code to test different CUDA concepts, configurations, or algorithms. Use each project folder as a playground for trying new ideas.

4. **Track Progress:**
   Keep notes or comments within the code about what you learned or discovered while working on each project.

---

## License

This repository is licensed under the MIT License. Feel free to use or reference the code, but keep in mind that it was primarily created for personal practice.

---
