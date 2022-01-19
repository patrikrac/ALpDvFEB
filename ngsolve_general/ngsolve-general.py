"""
Created by Patrik Rác
Program using Netgen/NGSolve for the solution of elliptic PDEs using adaptive mesh refinement.
 
The problem is solved using an adaptivity strategy with an gradient recovery error estimator.
The adaptivity and Problem geometry can be easily changed without affecting the performance of the programm.

The program runns with the command "netgen ngsolve-general.py" or "python3.8 ngsolve-general.py"
"""

import sys
import os
from typing import NamedTuple
from ngsolve import *
from netgen.csg import *
from netgen.geom2d import CSG2d, Rectangle

from Timer import Timer

__output__ = True
__timing__ = True

class Metrics(NamedTuple):
    """
    Named Immutable Tuple to store the data collected each cycle
    """
    cycle: int
    cells: int
    dofs: int
    max_error: float
    l2_error: float
<<<<<<< HEAD
    l2_relative_error: float
=======
    refinement_time: float
    solution_time: float
    assembly_time: float
>>>>>>> parallel
    error_p1: float
    error_p2: float
    error_p3: float


class Poisson:
    """
    Class for solving a Poisson equation with adaptive mesh refinement.
    The problem parameters are defined in the __init__ method.
    """
    
    def __init__(self, order, max_dof):
        
        self.order = order
        self.max_dof = max_dof
<<<<<<< HEAD
        
        self.timer = Timer()
        
=======
        
        self.timer = Timer()
        
>>>>>>> parallel
        #User concfiguration (Problem definition)
         #=============================================
        #Define the general parameters and functions for the Problem
        #Define the parameter alpha which is dependant on the used geometry
        #self.alpha = 1.0/2.0
<<<<<<< HEAD
        #self.r = sqrt(x*x + y*y)
        #self.phi = atan2(y,x)
=======
        #self.r = sqrt((x-0.5)*(x-0.5) + y*y)
        #self.phi = atan2(y,(x-0.5))
        #k = 10.0
>>>>>>> parallel

        #Define the boundary function g
        #self.g = CoefficientFunction([(self.r**self.alpha)*sin(self.alpha*self.phi) if bc=="L" else (self.r**self.alpha)*sin(self.alpha*(2*math.pi + self.phi)) if bc=="I" else 0 for bc in self.mesh.GetBoundaries()])
        #self.g = (self.r**self.alpha)*sin(self.alpha*self.phi)* (z**2)
        self.g=exp(-10*(x+y))*(z*z)
        #self.g = sin(k*x) * cos(2*k*y) * exp(z)

        #The exact solution of the problem. The mesh is divided into different materiels through a line. This is necessary in order to define teh function but can be ommited if the errror estimation isn't wanted.
        #self.uexact = CoefficientFunction([(self.r**self.alpha)*sin(self.alpha*self.phi) if m=="upper" else (self.r**self.alpha)*sin(self.alpha*(2*math.pi + self.phi)) if m=="lower" else 0 for m in self.mesh.GetMaterials()])
        #self.uexact = (self.r**self.alpha)*sin(self.alpha*self.phi)*(z**2)
        self.uexact = exp(-10*(x+y))*(z*z)
        #self.uexact = sin(k*x) * cos(2*k*y) * exp(z)

        #Define the right hand side of the poission problem
        self.rhs = -(200*(z*z) + 2)*exp(-10*(x+y))
<<<<<<< HEAD
=======
        #self.rhs = -2.0 * (self.r**self.alpha) * sin(self.alpha*self.phi)
        #self.rhs = (5*k*k - 1) * sin(k * x) * cos(2 * k * y) * exp(z)
>>>>>>> parallel
        #=============================================
        
        #Generate the mesh
        self.mesh = self.make_mesh()

        #Table list used for storing the error in each iteration
        self.table_list = []
        
        #Setup the FE-Space and the Solution vector with proper boundary conditions, as well as the space and solution for the error estimation
        (self.fes, self.gfu, self.space_flux, self.gf_flux) = self.setup_space()
        
        #Setup a grid function corresponding to the exact solution (necessary for error calculation)
        self.solution = GridFunction(self.fes, autoupdate = True)

        #Get the Bilinear and Linear form aswell as the solver.
        (self.a, self.f, self.c) = self.setup_system()
        
        self.bvp = BVP(bf=self.a, lf=self.f, gf=self.gfu, pre=self.c, prec=1e-12, maxsteps=2000)


    def make_mesh(self):
        """
        Initialize the problem mesh.
        Dimensionality has to be taken into account.
        """

        brick = OrthoBrick(Pnt(0.0,0.0,0.0), Pnt(1.0,1.0,1.0)).bc('bnd')
        #rect = Rectangle( pmin=(0,0), pmax=(1.0,1.0), bc="bnd" )

        geo = CSGeometry()
        #geo = CSG2d()
        geo.Add (brick)

        return Mesh(geo.GenerateMesh(maxh=0.125))
        

    def setup_space(self):
        """
        Initialize the problem space and the space for the error estimator.
        Order of the used FE-Elements can be changed here.
        """
        fes = H1(self.mesh, order=self.order, dirichlet="bnd", autoupdate=True)

        #Set up the Solution vector on the Finite Element Space and set the boundary conditions
        gfu = GridFunction(fes, autoupdate=True)
        gfu.Set(self.g, definedon=self.mesh.Boundaries("bnd"))
        
        #Setup the space and solution for the error estimation using a ZZ estimator
        space_flux = HDiv(self.mesh, order=self.order, autoupdate=True)
        gf_flux = GridFunction(space_flux, autoupdate=True)

        return (fes, gfu, space_flux, gf_flux)


    def setup_system(self):
        """
        Define the Problem itself using its weak formulation and therefore the Biliniar and Linear form.
        Initialize the Preconditioner for the solution.
        The Bilinear and Linear Form of the Programm can be changed here. (Right hand side of the equation)
        """
        #Define Test and Trial functions
        u, v = self.fes.TnT()

        #Define the Bilinear form corresponding to the given problem
        a = BilinearForm(self.fes)
        a += grad(u)*grad(v)*dx

        #Define the Linear form corresponding to the given problem
        f = LinearForm(self.fes)
        f += self.rhs*v*dx
        
        #Define the solver to be used to solve the problem
<<<<<<< HEAD
        c = Preconditioner(a, type = "local")
=======
        #c = MultiGridPreconditioner(a, inverse = "sparsecholesky")
        c = Preconditioner(a, type="bddc")
>>>>>>> parallel

        return (a,f,c)
    
    
    def estimate_error(self):
        """
        Estimate the error using an ZZ-error estimator described in the documentation.
        """
        # FEM-flux
        flux =  grad(self.gfu)

        # interpolate into H(div)
        self.gf_flux.Set(flux)

        # compute estimator:
        err = (flux-self.gf_flux)*(flux-self.gf_flux)
        eta2 = Integrate(err, self.mesh, VOL, element_wise=True)

        # mark for refinement:
        maxerr = max(eta2)

        for el in self.mesh.Elements():
<<<<<<< HEAD
            self.mesh.SetRefinementFlag(el, eta2[el.nr] > 0.3*maxerr)

=======
            self.mesh.SetRefinementFlag(el, eta2[el.nr] > 0.2*maxerr)
>>>>>>> parallel

    def assemble(self):
        self.a.Assemble()
        self.f.Assemble()
        
    def solve(self):
        """
        Solve the Problem by assembling the system and using the predefined solver.
        """
        self.c.Update()
        self.bvp.Do()


    def output_vtk(self, cycle):
        """
        Create an vtk output file.
        """
        # VTKOutput object
        vtk = VTKOutput(ma=self.mesh,
                        coefs=[self.gfu,],
                        names = ["u",],
                        filename="output/solution_{}".format(cycle),
                        subdivision=0)
        
        # Exporting the results:
        vtk.Do()

    
    def calculate_max_error(self):
        err = 0.0
<<<<<<< HEAD
        for v in self.mesh.vertices:
            x, y, z = v.point
            ip = self.mesh(x, y, z)
            point_err = abs(self.gfu(ip) - self.uexact(ip))
            if err < point_err: err = point_err
        return err

    def exact_error(self, cycle):
=======
        point_err = 0.0
        for v in self.mesh.vertices:
            ip = self.mesh(*v.point)
            point_err = abs(self.gfu(ip) - self.solution(ip))
            if err < point_err: 
                err = point_err 
        return err


    def exact_error(self, cycle, refinement_time, solution_time, assembly_time):
>>>>>>> parallel
        """
        Takes the current solution and the defined real solution and calculates the exact error.
        The function stores the absolute and relative along with other metrics in the table_list array.
        """

        l2_error = sqrt(Integrate((self.gfu - self.uexact)**2, self.mesh))
        
<<<<<<< HEAD
        l2_relative_error = sqrt(Integrate((self.gfu - self.uexact)**2,self.mesh)) / sqrt(Integrate(self.uexact**2, self.mesh))
=======
        self.solution.Set(self.uexact)
>>>>>>> parallel
        
        max_error = self.calculate_max_error()
        #max_error = max(Integrate(self.gfu, self.mesh, VOL, element_wise=True)-Integrate(self.uexact, self.mesh, VOL, element_wise=True))   

        ip1 = self.mesh(0.125, 0.125, 0.125)
<<<<<<< HEAD
        error_p1 = abs(self.gfu(ip1) - self.uexact(ip1))
        
        ip2 = self.mesh(0.25, 0.25, 0.25)
        error_p2 = abs(self.gfu(ip2) - self.uexact(ip2))
        
        ip3 = self.mesh(0.5, 0.5, 0.5)
        error_p3 = abs(self.gfu(ip3) - self.uexact(ip3))
=======
        error_p1 = abs(self.gfu(ip1) - self.solution(ip1))
        
        ip2 = self.mesh(0.25, 0.25, 0.25)
        error_p2 = abs(self.gfu(ip2) - self.solution(ip2))
        
        ip3 = self.mesh(0.5, 0.5, 0.5)
        error_p3 = abs(self.gfu(ip3) - self.solution(ip3))
>>>>>>> parallel
        
        num_cells = len([el for el in self.mesh.Elements()])

        self.table_list.append(Metrics(cycle, 
                                       num_cells, 
                                       self.fes.ndof, 
                                       max_error,
                                       l2_error,
<<<<<<< HEAD
                                       l2_relative_error,
=======
                                       refinement_time,
                                       solution_time,
                                       assembly_time,
>>>>>>> parallel
                                       error_p1, error_p2, error_p3))
        
        print("Max error: {}".format(max_error))
        print("L2 error: {}".format(l2_error))


    def output_Table(self):
        """
        Uses the table list vector to output an .tex file containing a table with the collected data.
        """
<<<<<<< HEAD
        f =open("output/table.tex", 'w')
        f.write("\\begin{table}[h]\n")
        f.write("\t\\begin{center}\n")
        f.write("\t\t\\begin{tabular}{|c|c|c|c|c|} \hline\n")
        
        plot_l2 = open("output/error_l2.txt", 'w')
        plot_l2.write("$NGSolve$\n")
        plot_l2.write("$n_\\text{dof}$\n")
        plot_l2.write("$\\left\\|u - u_h\\right\\| _{L^2}$\n")
        plot_l2.write("{}\n".format(len(self.table_list)))

        plot_max = open("output/error_max.txt", 'w')
        plot_max.write("$NGSolve$\n")
=======
        f =open("output/table_ngsolve.tex", 'w')
        f.write("\\begin{table}[h]\n")
        f.write("\t\\begin{center}\n")
        f.write("\t\t\\begin{tabular}{|c|c|c|c|c|c|c|c|} \hline\n")
        
        plot_l2 = open("output/error_l2_ngsolve.txt", 'w')
        plot_l2.write("NGSolve\n")
        plot_l2.write("$n_\\text{dof}$\n")
        plot_l2.write("$\\left\\|u_h - I_hu\\right\\| _{L_2}$\n")
        plot_l2.write("{}\n".format(len(self.table_list)))

        plot_max = open("output/error_max_ngsolve.txt", 'w')
        plot_max.write("NGSolve\n")
>>>>>>> parallel
        plot_max.write("$n_\\text{dof}$\n")
        plot_max.write("$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$\n")
        plot_max.write("{}\n".format(len(self.table_list)))
        
<<<<<<< HEAD
        plot_p1 = open("output/error_p1.txt", 'w')
        plot_p1.write("$NGSolve$\n")
=======
        plot_p1 = open("output/error_p1_ngsolve.txt", 'w')
        plot_p1.write("$x_1$\n")
>>>>>>> parallel
        plot_p1.write("$n_\\text{dof}$\n")
        plot_p1.write("$|u(x_i) - u_h(x_i)|$\n")
        plot_p1.write("{}\n".format(len(self.table_list)))

<<<<<<< HEAD
        plot_p2 = open("output/error_p2.txt", 'w')
        plot_p2.write("$NGSolve$\n")
=======
        plot_p2 = open("output/error_p2_ngsolve.txt", 'w')
        plot_p2.write("$x_2$\n")
>>>>>>> parallel
        plot_p2.write("$n_\\text{dof}$\n")
        plot_p2.write("$|u(x_i) - u_h(x_i)|$\n")
        plot_p2.write("{}\n".format(len(self.table_list)))
        
<<<<<<< HEAD
        plot_p3 = open("output/error_p3.txt", 'w')
        plot_p3.write("$NGSolve$\n")
=======
        plot_p3 = open("output/error_p3_ngsolve.txt", 'w')
        plot_p3.write("$x_3$\n")
>>>>>>> parallel
        plot_p3.write("$n_\\text{dof}$\n")
        plot_p3.write("$|u(x_i) - u_h(x_i)|$\n")
        plot_p3.write("{}\n".format(len(self.table_list)))
<<<<<<< HEAD

        f.write("\t\t\tcycle & \# cells & \# dofs & $\\left\\|u - u_h\\right\\| _{L^\\infty}$ & $\dfrac{\\left\\|u - u_h\\right\\| _{L^\\infty}}{\\left\\|u - u_h\\right\\| _{L^\\infty}}$\\\ \hline\n")
        for m in self.table_list:
            f.write("\t\t\t{} & {} & {} & {:.3e} & {:.3e}\\\ \hline\n".format(m.cycle, m.cells, m.dofs, m.l2_error, m.max_error))
=======
        
        plot_time_dof = open("output/time_dof_ngsolve.txt", 'w')
        plot_time_dof.write("NGSolve\n")
        plot_time_dof.write("$n_\\text{dof}$\n")
        plot_time_dof.write("$Time [s]$\n")
        plot_time_dof.write("{}\n".format(len(self.table_list)))
        
        plot_time_l2 = open("output/time_l2_ngsolve.txt", 'w')
        plot_time_l2.write("NGSolve\n")
        plot_time_l2.write("$Time [s]$\n")
        plot_time_l2.write("$\\left\\|u_h - I_hu\\right\\| _{L_2}$\n")
        plot_time_l2.write("{}\n".format(len(self.table_list)))
        
        plot_time_max = open("output/time_max_ngsolve.txt", 'w')
        plot_time_max.write("NGSolve\n")
        plot_time_max.write("$Time [s]$\n")
        plot_time_max.write("$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$\n")
        plot_time_max.write("{}\n".format(len(self.table_list)))
        
        plot_refinement_time_dof = open("output/refinement_time_dof_ngsolve.txt", 'w')
        plot_refinement_time_dof.write("NGSolve\n")
        plot_refinement_time_dof.write("$n_\\text{dof}$\n")
        plot_refinement_time_dof.write("$Time [s]$\n")
        plot_refinement_time_dof.write("{}\n".format(len(self.table_list)))
        
        plot_refinement_time_l2 = open("output/refinement_time_l2_ngsolve.txt", 'w')
        plot_refinement_time_l2.write("NGSolve\n")
        plot_refinement_time_l2.write("$Time [s]$\n")
        plot_refinement_time_l2.write("$\\left\\|u_h - I_hu\\right\\| _{L_2}$\n")
        plot_refinement_time_l2.write("{}\n".format(len(self.table_list)))
        
        plot_refinement_time_max = open("output/refinement_time_max_ngsolve.txt", 'w')
        plot_refinement_time_max.write("NGSolve\n")
        plot_refinement_time_max.write("$Time [s]$\n")
        plot_refinement_time_max.write("$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$\n")
        plot_refinement_time_max.write("{}\n".format(len(self.table_list)))
        
        plot_assembly_time_dof = open("output/assembly_time_dof_ngsolve.txt", 'w')
        plot_assembly_time_dof.write("NGSolve\n")
        plot_assembly_time_dof.write("$n_\\text{dof}$\n")
        plot_assembly_time_dof.write("$Time [s]$\n")
        plot_assembly_time_dof.write("{}\n".format(len(self.table_list)))
        
        plot_assembly_time_l2 = open("output/assembly_time_l2_ngsolve.txt", 'w')
        plot_assembly_time_l2.write("NGSolve\n")
        plot_assembly_time_l2.write("$Time [s]$\n")
        plot_assembly_time_l2.write("$\\left\\|u_h - I_hu\\right\\| _{L_2}$\n")
        plot_assembly_time_l2.write("{}\n".format(len(self.table_list)))
        
        plot_assembly_time_max = open("output/assembly_time_max_ngsolve.txt", 'w')
        plot_assembly_time_max.write("NGSolve\n")
        plot_assembly_time_max.write("$Time [s]$\n")
        plot_assembly_time_max.write("$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$\n")
        plot_assembly_time_max.write("{}\n".format(len(self.table_list)))

        f.write("\t\t\tcycle & $n_{cells} $ & $n_{dof}$ & $\\left\\|u_h - I_hu\\right\\| _{L_2}$ & $\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$ & $t_{solve}$ & $t_{refine}$ & $t_{assembly}$\\\\ \\hline\n")
        for m in self.table_list:
            f.write("\t\t\t{} & {} & {} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e}\\\ \hline\n".format(m.cycle, m.cells, m.dofs, m.l2_error, m.max_error, m.solution_time, m.refinement_time, m.assembly_time))
>>>>>>> parallel
            plot_l2.write("{} {}\n".format(m.dofs, m.l2_error))
            plot_max.write("{} {}\n".format(m.dofs, m.max_error))
            plot_p1.write("{} {}\n".format(m.dofs, m.error_p1))
            plot_p2.write("{} {}\n".format(m.dofs, m.error_p2))
            plot_p3.write("{} {}\n".format(m.dofs, m.error_p3))
<<<<<<< HEAD
=======
            
            plot_time_dof.write("{} {}\n".format(m.dofs, m.solution_time))
            plot_refinement_time_dof.write("{} {}\n".format(m.dofs, m.refinement_time))
            plot_assembly_time_dof.write("{} {}\n".format(m.dofs, m.assembly_time))
            
            
            plot_time_l2.write("{} {}\n".format(m.solution_time, m.l2_error))
            plot_refinement_time_l2.write("{} {}\n".format(m.refinement_time, m.l2_error))
            plot_assembly_time_l2.write("{} {}\n".format(m.assembly_time, m.l2_error))
            
            plot_time_max.write("{} {}\n".format(m.solution_time, m.max_error))
            plot_refinement_time_max.write("{} {}\n".format(m.refinement_time, m.max_error))
            plot_assembly_time_max.write("{} {}\n".format(m.assembly_time, m.max_error))
>>>>>>> parallel
            

        f.write("\t\t\end{tabular}\n")
        f.write("\t\end{center}\n")
        f.write("\\end{table}")
        f.close()
        
        plot_l2.close()
        plot_max.close()
        plot_p1.close()
        plot_p2.close()
        plot_p3.close()
        
        plot_time_dof.close()
        plot_refinement_time_dof.close()
        plot_assembly_time_dof.close()
        
        
        plot_time_l2.close()
        plot_refinement_time_l2.close()
        plot_assembly_time_l2.close()
        
        plot_time_max.close()
        plot_refinement_time_max.close()
        plot_assembly_time_max.close()


    def do(self):
        """
        Execute the computation of the solution. Solve the problem using an adaptive mesh until a certain number of degrees of freedom are reached.

        """
        cycle = 0
<<<<<<< HEAD
        while self.fes.ndof < self.max_dof:
            self.mesh.Refine()

            self.gfu.Set(self.g, definedon=self.mesh.Boundaries("bnd"))
            
            if __timing__:
                self.timer.startTimer()

            self.solve()
            
            if __timing__:
                self.timer.printTimer()

            if __output__:
                self.exact_error(cycle)
                self.output_vtk(cycle)

            self.estimate_error()

            print("Cycle: {}, DOFs: {}".format(cycle, self.fes.ndof))
            cycle += 1
        
        if __output__:
            self.output_vtk(cycle)
            self.output_Table()
=======
        ref_time = 0.0
        sol_time = 0.0
        assem_time = 0.0
        with TaskManager():
            while True:
                self.gfu.Set(self.g, definedon=self.mesh.Boundaries("bnd"))
                
                if __timing__:
                    self.timer.startTimer()

                self.assemble()
                
                if __timing__:
                    assem_time = self.timer.printTimer()
                
                if __timing__:
                    self.timer.startTimer()

                self.solve()
                
                if __timing__:
                    sol_time = self.timer.printTimer()

                if __output__:
                    self.exact_error(cycle, ref_time, sol_time, assem_time)
                
                if self.fes.ndof >= self.max_dof:
                    break
                
                if __timing__:
                    self.timer.startTimer()
           
                self.estimate_error()
                self.mesh.Refine()
                
                if __timing__:
                    ref_time = self.timer.printTimer()


                print("Cycle: {}, DOFs: {}".format(cycle, self.fes.ndof))
                cycle += 1
        
        if __output__:
            self.output_vtk(cycle)
            
        self.output_Table()
>>>>>>> parallel
        


if __name__ == "__main__":
    if len(sys.argv) == 3:
        current_directory = os.getcwd()
        result_directory = os.path.join(current_directory, r'output')
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        e = Poisson(int(sys.argv[1]), int(sys.argv[2]))
        e.do()
<<<<<<< HEAD
    else:    
        print("usage: python3.8 ngsolve-general.py <order> <max_dof>")
=======
    else: 
        print("usage: python3.8/ngspy ngsolve-general.py <order> <max_dof>")
>>>>>>> parallel
