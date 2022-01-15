from mpi4py import MPI
from netgen.csg import unit_cube
import netgen.meshing

# ngsolve-imports
from ngsolve import *

# initialize MPI
comm = MPI.COMM_WORLD
MPI_Init()
rank = comm.rank
np = comm.size

do_vtk = False

print("Hello from rank "+str(rank)+" of "+str(np))

mesh = Mesh(unit_cube.GenerateMesh(maxh=0.1))


    
print('rank', str(comm.rank)+"'s part of the mesh has ", mesh.ne, 'elements, ', \
      mesh.nface, 'faces, ', mesh.nedge, 'edges and ', mesh.nv, ' vertices')