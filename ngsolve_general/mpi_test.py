from netgen.csg import unit_cube
import netgen.meshing

# ngsolve-imports
from ngsolve import *

# initialize MPI
comm = mpi_world
rank = comm.rank
np = comm.size

do_vtk = False

print("Hello from rank "+str(rank)+" of "+str(np))
