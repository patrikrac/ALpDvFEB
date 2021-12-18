# Poisson problem solver implemented using the deal.ii library

## Installation of the deal.ii library
The parallel build of the deal.ii library requires the `PETSc` and `METIS` libraries to be installed/built. 
Optionally one can choose to build only the serial version by ommiting the extra cmake-flags in Step ...
1. Clone the deal.ii source from https://github.com/dealii/dealii.
2. `cd dealii`
3. `mkdir build`
4. `cd build`
5. `cmake .. -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_METIS=ON -DDEAL_II_WITH_PETSC=ON` -> Might require the path to the PETSc/METIS build directories.
6. `make -j 4`
7. Optional: `make test`

## Building
The programm comes with a CMake-File used to create the buildsystem.
1. `cd deal_general`
2. `mkdir build`
3. `cd build`
4. `cmake ..` -> Might require the path to the deal.ii build directory
5. `make`

CMake also takes the optional arguments:
- `-DUSE_TIMING=OFF` -> Controlls the time measurements in the programm (Default: ON)
- `-DUSE_OUTPUT=OFF` -> Controlls the vtk and txt output of the programm (Default: ON)

## Execution
### Serial
Usage: `./deal_general <p> <max_dof> [-hp]`
- _p_: Order of elements to be used.
- _max\_dof_: Maximum number of degrees of freedum until the programm should terminate 
- _-hp_: If this flag is set the programm uses hp-refinement instead of the default h-refinement (Parameter _p_ is required but ommited)

### Parallel
Usage: `mpirun -np <num_procs> ./deal_general <p> <max_dof>`    
Hint: Currently hp is not yet supported in the parallel version