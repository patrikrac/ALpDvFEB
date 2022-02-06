# Poisson problem solver implemented using the NGSPy library

## Installation of Netgen/NGSolve
Detailed installation instructions can be found under https://ngsolve.org/downloads 
The graphical user interface is not used and hence a source build is sufficient.

## Execution
### Serial
Usage: `python3.8 ngsolve_main.py <p> <max_dof>` 
or          `netgen ngsolve_main.py <p> <max_dof>` 
or          `ngspy ngsolve_main.py <p> <max_dof>`
- _p_: Order of elements to be used.
- _max\_dof_: Maximum number of degrees of freedum until the programm should terminate  
