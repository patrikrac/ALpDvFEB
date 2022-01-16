# Call with: 
# mpirun -np 5 ngspy mpi_poisson.py

# Solves -laplace(u)=f on [0,1]^3

# netgen-imports
#from netgen.geom2d import unit_square
from queue import Empty
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

if rank==0:
    # master-proc generates mesh
    mesh = unit_cube.GenerateMesh(maxh=0.3)
    # and saves it to file
    mesh.Save("some_mesh.vol")

# wait for master to be done meshing
comm.Barrier()

# now load mesh from file
ngmesh = netgen.meshing.Mesh(dim=3, comm=comm)
ngmesh.Load("some_mesh.vol")

#refine once?
# ngmesh.Refine()

mesh = Mesh(ngmesh)

# build H1-FESpace as usual
V = H1(mesh, order=3, dirichlet=[1,2,3,4])
u = V.TrialFunction()
v = V.TestFunction()

print("rank "+str(rank)+" has "+str(V.ndof)+" of "+str(V.ndofglobal)+" dofs!")

# RHS does not change either!
f = LinearForm (V)
f += SymbolicLFI(32 * (y*(1-y)+x*(1-x)) * v)

# neither does the BLF!
a = BilinearForm (V, symmetric=False)
a += SymbolicBFI(grad(u)*grad(v))

# Some possible preconditioners: 
#c = Preconditioner(a, type="direct", inverse = "masterinverse") # direct solve with mumps
#c = Preconditioner(a, type="bddc", inverse = "mumps")   # BBDC + mumps for coarse inverse
c = Preconditioner(a, type="hypre")                             # BoomerAMG (use only for order 1)
#c = Preconditioner(a, type="bddc", usehypre = True)     # BDDC + BoomerAMG for coarse matrix



# solve the equation

u = GridFunction(V)    
 # use CG-solver with preconditioner c
# u.vec.data = a.mat.Inverse(V.FreeDofs(), inverse="mumps") * f.vec  # use MUMPS parallel inverse
# u.vec.data = a.mat.Inverse(V.FreeDofs(), inverse="masterinverse") * f.vec  # use masterinverse (master proc does all the work!)


#exact solution
exact = 16*x*(1-x)*y*(1-y)


space_flux = HDiv(mesh, order=2, autoupdate=True)
gf_flux = GridFunction(space_flux, "flux", autoupdate=True)

def SolveBVP():
    a.Assemble()
    f.Assemble()
    inv = CGSolver(a.mat, c.mat)
    u.vec.data = inv * f.vec

l = []

def CalcError():
    flux = grad(u)
    # interpolate finite element flux into H(div) space:
    gf_flux.Set (flux)

    # Gradient-recovery error estimator
    err = (flux-gf_flux)*(flux-gf_flux)
    elerr = Integrate (err, mesh, VOL, element_wise=True)
    if elerr is Empty:
        return
    maxerr = max(elerr)
    l.append ( (V.ndof, sqrt(sum(elerr)) ))
    print ("maxerr = ", maxerr)

    for el in mesh.Elements():
        mesh.SetRefinementFlag(el, elerr[el.nr] > 0.25*maxerr)


while V.ndof < 100000:  
        SolveBVP()
        error = Integrate ( (u-exact)*(u-exact) , mesh)
        if rank==0:
            print("L2-error", error )
        CalcError()
        mesh.Refine()
    
SolveBVP()

if do_vtk:
    # do VTK-output
    import os
    output_path = os.path.dirname(os.path.realpath(__file__)) + "/poisson_output"
    if rank==0 and not os.path.exists(output_path):
        os.mkdir(output_path)
    comm.Barrier() #wait until master has created the directory!!

    vtk = VTKOutput(mesh, coefs=[u], names=["sol"], filename=output_path+"/vtkout", subdivision=2)
    vtk.Do()
