# Poisson problem solver implemented using the MFEM library

## Installation of the MFEM library
The parallel build of mfem requires the `HYPRE` and `METIS`library to be installed/built.   
Optionally one can choose to just build the serial version by ommiting the extra cmake-flag in Step 5.
1. Clone the MFEM source from https://github.com/mfem/mfem.org
2. `cd mfem`
3. `mkdir build`
4. `cd build`
5. `cmake .. -DMFEM_USE_MPI=ON` -> Might require the path to the HYPRE/METIS build directory
6. `make -j 8`

## Building
The programm comes with a CMake-File used to create the buildsystem. 
1. `cd mfem_main`
2. `mkdir build`
3. `cd build`
4. `cmake ..`
5. `make`

CMake also takes the optional arguments:
- `-DUSE_TIMING=OFF` -> Controlls the time measurements in the programm (Default: ON)
- `-DUSE_OUTPUT=OFF` -> Controlls the vtk and txt output of the programm (Default: ON)

## Execution
### Serial
Usage: `./mfem_main <p> <max_dof>`
- _p_: Order of elements to be used.
- _max\_dof_: Maximum number of degrees of freedum until the programm should terminate  

### Parallel
Usage: `mpirun -np <num_procs> ./mfem_main <p> <max_dof>`
- _p_: Order of elements to be used.
- _max\_dof_: Maximum number of degrees of freedum until the programm should terminate