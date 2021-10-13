"""
Created by Patrik Rác
Solution of the following problem
-Solving the Laplace on a defined mesh. 
The problem is solved using an adaptivity strategy with an gradient recovery error estimator.
The adaptivity and Problem geometry can be easily changed without affecting the performance of the programm.
The programm runns with the command "netgen ngsolve-general.py or python3.8 ngsolve-general.py"
"""
from ngsolve import *
from netgen.csg import *

class Problem:

    def __init__(self):
        
        #Define the general parameters and functions for the Problem
        #Define the parameter alpha which is dependant on the used geometry
        self.alpha = 1.0/2.0
        self.r = sqrt(x*x + y*y)
        self.phi = atan2(y,x)

        #Define the boundary function g
        #self.g = CoefficientFunction([(self.r**self.alpha)*sin(self.alpha*self.phi) if bc=="L" else (self.r**self.alpha)*sin(self.alpha*(2*math.pi + self.phi)) if bc=="I" else 0 for bc in self.mesh.GetBoundaries()])
        self.g = (self.r**self.alpha)*sin(self.alpha*self.phi)*(z**2)

        #The exact solution of the problem. The mesh is divided into different materiels through a line. This is necessary in order to define teh function but can be ommited if the errror estimation isn't wanted.
        #self.uexact = CoefficientFunction([(self.r**self.alpha)*sin(self.alpha*self.phi) if m=="upper" else (self.r**self.alpha)*sin(self.alpha*(2*math.pi + self.phi)) if m=="lower" else 0 for m in self.mesh.GetMaterials()])
        self.uexact = (self.r**self.alpha)*sin(self.alpha*self.phi)*(z**2)

        #Generate the mesh
        self.mesh = self.make_mesh()

        #Table list used for storing the error in each iteration
        self.table_list = []
        
        #Setup the FE-Space and the Solution vector with proper boundary conditions, as well as the space and solution for the error estimation
        (self.fes, self.gfu, self.space_flux, self.gf_flux) = self.setup_space()

        #Get the Bilinear and Linear form aswell as the solver.
        (self.a, self.f, self.c) = self.setup_system()


    def make_mesh(self):
        """
        Initialize the problem mesh.
        Dimensionality has to be taken into account.
        """

        brick = OrthoBrick(Pnt(-0.5,0.0,0.0), Pnt(0.5,1.0,1.0)).bc('bnd')

        geo = CSGeometry()
        geo.Add (brick)

        return Mesh(geo.GenerateMesh(maxh=0.25))
        

    def setup_space(self):
        """
        Initialize the problem space and the space for the error estimator.
        Order of the used FE-Elements can be changed here.
        """
        fes = H1(self.mesh, order=2, dirichlet="bnd", autoupdate=True)

        #Set up the Solution vector on the Finite Element Space and set the boundary conditions
        gfu = GridFunction(fes, autoupdate=True)
        gfu.Set(self.g, definedon=self.mesh.Boundaries("bnd"))
        
        #Setup the space and solution for the error estimation using a ZZ estimator
        space_flux = HDiv(self.mesh, order=2, autoupdate=True)
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
        f += -2.0*v*dx

        #Define the solver to be used to solve the problem
        c = Preconditioner(a, "local")

        return (a,f,c)
    
    
    def calculate_error(self):
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
        print ("maxerr: ", maxerr)

        for el in self.mesh.Elements():
            self.mesh.SetRefinementFlag(el, eta2[el.nr] > 0.25*maxerr)


    def solve(self):
        """
        Solve the Problem by assembling the system and using the predefined solver.
        """
        self.a.Assemble()

        self.f.Assemble()

        self.c.Update()

        solvers.BVP(bf=self.a, lf=self.f, gf=self.gfu, pre=self.c)


    def output(self):
        """
        Create an vtk output file.
        """
        # VTKOutput object
        vtk = VTKOutput(ma=self.mesh,
                        coefs=[self.gfu,],
                        names = ["u",],
                        filename="result",
                        subdivision=0)


        # Exporting the results:
        vtk.Do()


    def exact_error(self, cycle):
        """
        Takes the current solution and the defined real solution and calculates the exact error.
        The function stores the absolute and relative along with other metrics in the table_list array.
        """
        values = []
        values.append(cycle)

        values.append(len([el for el in self.mesh.Elements()]))

        values.append(self.fes.ndof)

        l2_error = sqrt(Integrate((self.gfu - self.uexact)*(self.gfu - self.uexact), self.mesh))
        #values.append(l2_error)

        max_error = max(Integrate((self.gfu - self.uexact), self.mesh, VOL, element_wise=True))
        print("Max-error: ",  max_error)
        values.append(max_error)

        l2_relative_error = sqrt(Integrate((self.gfu - self.uexact)*(self.gfu - self.uexact),self.mesh)) / sqrt(Integrate(self.uexact*self.uexact, self.mesh))
        #values.append(l2_relative_error)
        #print("Relative error: ", l2_relative_error)

        max_relative_error = max_error / max(Integrate(self.uexact,  self.mesh, VOL, element_wise=True))
        values.append(max_relative_error)

        self.table_list.append(values)


    def output_Table(self):
        """
        Uses the table list vector to output an .tex file containing a table with the collected data.
        """
        f =open("table.tex", 'w')
        f.write("\\begin{table}[h]\n")
        f.write("\t\\begin{center}\n")
        f.write("\t\t\\begin{tabular}{|c|c|c|c|c|} \hline\n")

        plot_f = open("errorNG.txt", 'w')
        plot_f.write("$NGSolve$\n")
        plot_f.write("$n_\\text{dof}$\n")
        plot_f.write("$\\norm{u - u_h}_{L^\\infty}$\n")
        plot_f.write("{}\n".format(len(self.table_list)))

        f.write("\t\t\tcycle & \# cells & \# dofs & $\\norm{u - u_h}_{L^\\infty}$ & $\dfrac{\\norm{u - u_h}_{L^\\infty}}{\\norm{u}_{L^\\infty}}$\\\ \hline\n")
        for cycle, cells, dofs, error, relative_error in self.table_list:
            f.write("\t\t\t{} & {} & {} & {:.3e} & {:.3e}\\\ \hline\n".format(cycle, cells, dofs, error, relative_error))
            plot_f.write("{} {}\n".format(dofs, error))


        f.write("\t\t\end{tabular}\n")
        f.write("\t\end{center}\n")
        f.write("\\end{table}")
        f.close()
        plot_f.close()


    def do(self):
        """
        Execute the computation of the solution. Solve the problem using an adaptive mesh until a certain number of degrees of freedom are reached.

        """
        cycle = 0
        while self.fes.ndof < 100000:
            self.mesh.Refine()

            self.gfu.Set(self.g, definedon=self.mesh.Boundaries("bnd"))

            self.solve()

            self.exact_error(cycle)

            self.calculate_error()

            print(self.fes.ndof)
            cycle += 1
            
        self.output()
        self.output_Table()
        


if __name__ == "__main__":
    e = Problem()
    e.do()
