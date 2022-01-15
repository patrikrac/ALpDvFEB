from netgen.csg import unit_cube
import netgen.meshing

# ngsolve-imports
from ngsolve import *

# initialize MPI
comm = mpi_world
rank = comm.rank
np = comm.size


print("Hello from rank "+str(rank)+" of "+str(np))

if comm.rank == 0:
    ngmesh = unit_cube.GenerateMesh(maxh=0.1).Distribute(comm)
else:
    ngmesh = netgen.meshing.Mesh.Receive(comm)

mesh = Mesh(ngmesh)

print('rank', str(comm.rank)+"'s part of the mesh has ", mesh.ne, 'elements, ', \
      mesh.nface, 'faces, ', mesh.nedge, 'edges and ', mesh.nv, ' vertices')