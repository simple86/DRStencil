# DRStencil
GPU code generator for low-order stencil computations.   
Given the neccessary stencil parameters in a `.stc` file, `DRStencil` generates highly data-reusing GPU codes for stencil computation. 
`DRStencil` provides many optimazation techniques, and users are allowed to adjust the optimazation configurations.

## DEPENDENCIES
  1. python3 (version 3.8.5 tested)
  2. GCC (support for C++ 17 needed, version 9.3.0 tested)
  3. CUDA (version 11.2 tested)
  4. Nsight Compute (version 2020.3.1.0 tested)

## SETUP
  To build `drstencil`, you only need to run "make" in the main directory with the system supporting `C++ 17`.
  ```bash
  cd drstencil
  make
  ```
  Then, you'll get an executable `drstencil`. 
## USAGE
  Copy the executable `drstencil` into the test directory, `2d5pt_star` for instance. By launching the script `starter.sh`, you can start tuning GPU code for the stencil `2d5pt_star`. The script generates hundreds or thousands of optimazation configurations and profiles each of them with `Nsight Compute`. The profiling results will be located in `prof/` directory.
  ```bash
  cd benchmarks/2d5pt_star/
  ./starter.sh
  ```
  To collect the GPU metrics, launch the script `getGpuMetrics.sh`. (Note that the script is only supported by `Nsight Compute 2020.3`. Some modification need to be made for other versions.)
   ```bash
   bash getGpuMetrics.sh
   ```
  Usage of `drstencil`: 
  ```
  ./drstencil [options] <input_stcfile>
  ```   
  Lanuch the following command for more detailed usage of `drstenil`.
  ```bash
  ./drstencil -h
  ```
  To Create a new test, you need a `.stc` file like benchmarks/2d5pt_star/2d5pt_star.stc, which defines the stencil computation.
## PUBLICATION
  [HPCC 2021] 
