//Created by Patrik Rác
//This is a general MFEM program designed to solve Laplace-equations using the MFEM Library.
//The dimensionality of the programm can be changed by altering the input grid.

#include "mfem.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;
using namespace mfem;

// Boundary and right-hand side functions that determine the general behaviour of the programm.
double bdr_func(const Vector &p);
double rhs_func(const Vector &p);

class PointValueEvaluation
{
public:
   PointValueEvaluation(const vector<double> &evaluation_point);
   double operator()(GridFunction &x,  Mesh &mesh) const;

private:
   const vector<double> evaluation_point;
};

PointValueEvaluation::PointValueEvaluation(const vector<double> &evaluation_point) : evaluation_point(evaluation_point)
{
}

double PointValueEvaluation::operator()(GridFunction &x, Mesh &mesh) const
{
   int NE = mesh.GetNE();
   Vector vert_vals;
   DenseMatrix vert_coords;
   for (int i = 0; i < NE; i++)
   {
      Element *e = mesh.GetElement(i);
      int nv = e->GetNVertices();
      const IntegrationRule &ir = *Geometries.GetVertices(e->GetGeometryType());
      x.GetValues(i, ir, vert_vals, vert_coords);
      for (int j = 0; j < nv; j++) 
      {
         bool coords_match = true;
         for (int k = 0; k <evaluation_point.size(); k++)
         {
            if (evaluation_point[k] != vert_coords(k,j)) 
            {
               coords_match = false;
               break;
            }
         }

         if(coords_match)
         {
            return vert_vals(j);
         }
 
      }
   }

   return 1e20; 
}


//Metrics to be collected for later graphing.
typedef struct error_values
{
   int cycle;
   int cells;
   int dofs;
   double error;
   double error_p1;
   double error_p2;
   double error_p3;
   double relative_error;
} error_values;

//----------------------------------------------------------------
//Class for the specification of the problem
//----------------------------------------------------------------
class Problem
{
public:
   Problem() : hysteresis(0.2), max_elem_error(1.0e-10), order(2),
                     postprocessor1({0.125,0.125,0.125}), postprocessor2({0.25,0.25,0.25}), postprocessor3({0.5,0.5,0.5})
   {
   }
   void run();

private:
   void make_mesh();

   void update(BilinearForm &a, LinearForm &f, FiniteElementSpace &fespace, GridFunction &x, GridFunction &error_zero);
   void solve(BilinearForm &a, LinearForm &f, FiniteElementSpace &fespace, GridFunction &x, Array<int> &ess_bdr, FunctionCoefficient &bdr);
   bool refine(BilinearForm &a, LinearForm &f, FiniteElementSpace &fespace, GridFunction &x, GridFunction &error_zero, ThresholdRefiner &refiner);

   void exact_error(int cycle, int dofs, GridFunction &x, GridFunction &error_zero, FunctionCoefficient &u);

   void output_table();
   void glvis_output(GridFunction &x);
   void vtk_output(GridFunction &x);

   //Configuration parameters
   double hysteresis; //derefinement safety coefficient
   double max_elem_error;
   int order;
   PointValueEvaluation postprocessor1, postprocessor2, postprocessor3;

   //Data parameters
   Mesh mesh;
   vector<error_values> table_vector;
};

//----------------------------------------------------------------
//Create the mesh the problem will be solved on
//----------------------------------------------------------------
void Problem::make_mesh()
{
   const char *mesh_file = "unit_cube.mesh";
   mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < 3; i++)
   {
      mesh.UniformRefinement();
   }

   //Turns the Quad mesh (which supports hanging nodes) to a tetrahedral mesh without hanging nodes
   mesh = Mesh::MakeSimplicial(mesh);

   mesh.Finalize(true);

   cout << "Mesh generated." << endl;
}

//----------------------------------------------------------------
//Update all variables to adabt to the recently adapted mesh
//----------------------------------------------------------------
void Problem::update(BilinearForm &a, LinearForm &f, FiniteElementSpace &fespace, GridFunction &x, GridFunction &error_zero)
{
   fespace.Update();

   x.Update();

   error_zero.Update();
   error_zero = 0.0;

   fespace.UpdatesFinished();

   // Inform the linear and bilinear forms that the space has changed.
   a.Update();
   f.Update();
}

//----------------------------------------------------------------
//Solve the Problem on the current mesh
//----------------------------------------------------------------
void Problem::solve(BilinearForm &a, LinearForm &f, FiniteElementSpace &fespace, GridFunction &x, Array<int> &ess_bdr, FunctionCoefficient &bdr)
{
   a.Assemble();
   f.Assemble();

   // Project the exact solution to the essential boundary DOFs.
   x.ProjectBdrCoefficient(bdr, ess_bdr);

   //Create and solve the linear system.
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   SparseMatrix A;
   Vector B, X;

   a.FormLinearSystem(ess_tdof_list, x, f, A, X, B);

   GSSmoother M(A);
   PCG(A, M, B, X, 0, 500, 1e-12, 0.0);

   a.RecoverFEMSolution(X, f, x);
}

//----------------------------------------------------------------
//Execute one refinement step and call the update funciton to adapt the other variables
//----------------------------------------------------------------
bool Problem::refine(BilinearForm &a, LinearForm &f, FiniteElementSpace &fespace, GridFunction &x, GridFunction &error_zero, ThresholdRefiner &refiner)
{
   refiner.Apply(mesh);

   if (refiner.Stop())
   {
      return false;
   }

   // 20. Update the space and interpolate the solution.
   update(a, f, fespace, x, error_zero);

   return true;
}

//----------------------------------------------------------------
//This method initializes a lot of the functionality of the programm
//Run the problem and terminate after a given condition
//----------------------------------------------------------------
void Problem::run()
{
   //Mesh Generation
   make_mesh();

   //Declaration
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   //Setup the necessary spaces and solutions for the problem
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   BilinearForm a(&fespace);
   LinearForm f(&fespace);

   GridFunction x(&fespace);

   //Grid function for calculation of the relative norm
   GridFunction error_zero(&fespace);
   error_zero = 0.0;

   FunctionCoefficient u(bdr_func);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   //Setup
   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs(rhs_func);
   FunctionCoefficient bdr(bdr_func);

   //Specify the Problem
   BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
   a.AddDomainIntegrator(integ);
   f.AddDomainIntegrator(new DomainLFIntegrator(rhs));

   KellyErrorEstimator *estimator{nullptr};
   L2_FECollection flux_fec(order, dim);

   auto flux_fes = new FiniteElementSpace(&mesh, &flux_fec, sdim);
   estimator = new KellyErrorEstimator(*integ, x, flux_fes);

   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(0.0); // use purely local threshold
   refiner.SetLocalErrorGoal(max_elem_error);
   refiner.PreferConformingRefinement();
   refiner.SetNCLimit(0);

   ThresholdDerefiner derefiner(*estimator);
   derefiner.SetThreshold(hysteresis * max_elem_error);
   derefiner.SetNCLimit(0);

   x = 0.0;

   refiner.Reset();
   derefiner.Reset();

   int step = 0;
   while (true)
   {

      cout << "Step: " << step << endl
           << "DOF: " << fespace.GetNDofs() << endl;

      solve(a, f, fespace, x, ess_bdr, bdr);

      exact_error(step, fespace.GetNDofs(), x, error_zero, u);

      //Stop the loop if no more elements are marked for refinement or the desired number of DOFs is reached.
      if (!refine(a, f, fespace, x, error_zero, refiner) || fespace.GetNDofs() > 1000000)
      {
         break;
      }
      step++;
   }

   if (derefiner.Apply(mesh))
   {
      cout << "\nDerefined elements." << endl;

      update(a, f, fespace, x, error_zero);
   }

   cout << "Final: " << step << endl
        << "DOF: " << fespace.GetNDofs() << endl;

   a.Update();
   f.Update();

   delete estimator;

   vtk_output(x);
   output_table();
}

//----------------------------------------------------------------
//Calculate the exact error
//----------------------------------------------------------------
void Problem::exact_error(int cycle, int dofs, GridFunction &x, GridFunction &error_zero, FunctionCoefficient &u)
{
   error_values values = {};
   values.cycle = cycle;
   values.dofs = dofs;
   values.cells = mesh.GetNE();
   values.error = x.ComputeMaxError(u);
   values.relative_error = values.error / error_zero.ComputeMaxError(u);

   values.error_p1 = abs(postprocessor1(x, mesh) - bdr_func(Vector(new double[] {0.125,0.125,0.125}, 3)));
   values.error_p2 = abs(postprocessor2(x, mesh) - bdr_func(Vector(new double[] {0.25,0.25,0.25}, 3)));
   values.error_p3 = abs(postprocessor3(x, mesh) - bdr_func(Vector(new double[] {0.5,0.5,0.5}, 3)));
   table_vector.push_back(values);

   cout << "Error for step " << cycle << ": " << setprecision(3) << scientific << values.error << endl;
}

//----------------------------------------------------------------
//Output the table containing the calcualted errors
//----------------------------------------------------------------
void Problem::output_table()
{
   std::ofstream output("table.tex");
   std::ofstream output_custom("error_mfem.txt");
   std::ofstream output_p1("error_p1.txt");
   std::ofstream output_p2("error_p2.txt");
   std::ofstream output_p3("error_p3.txt");

   output_custom << "MFEM" << endl;
   output_custom << "$n_\\text{dof}$" << endl;
   output_custom << "$\\left\\|u - u_h\\right\\| _{L^\\infty}$" << endl;
   output_custom << table_vector.size() << endl;

   output << "\\begin{table}[h]" << endl;
   output << "\t\\begin{center}" << endl;
   output << "\t\t\\begin{tabular}{|c|c|c|c|c|} \\hline" << endl;

   output << "\t\t\tcycle & \\# cells & \\# dofs & $\\norm{u - u_h}_{L^\\infty}$ & $\\dfrac{\\norm{u - u_h}_{L^\\infty}}{\\norm{u}_{L^\\infty}}$\\\\ \\hline" << endl;
   for (int i = 0; i < table_vector.size(); i++)
   {
      output << "\t\t\t" << table_vector[i].cycle << " & " << table_vector[i].cells << " & " << table_vector[i].dofs << " & " << setprecision(3) << scientific << table_vector[i].error << " & " << setprecision(3) << scientific << table_vector[i].relative_error << "\\\\ \\hline" << endl;
      output_custom << table_vector[i].dofs << " " << table_vector[i].error << endl;
      output_p1 << table_vector[i].dofs << " " << table_vector[i].error_p1 << endl;
      output_p2 << table_vector[i].dofs << " " << table_vector[i].error_p2 << endl;
      output_p3 << table_vector[i].dofs << " " << table_vector[i].error_p3 << endl;
   }
   //TODO: Title for point wise error files.
   output << "\t\t\\end{tabular}" << endl;
   output << "\t\\end{center}" << endl;
   output << "\\end{table}" << endl;
}

//----------------------------------------------------------------
//Create a vtk Output for the current solution
//----------------------------------------------------------------
void Problem::vtk_output(GridFunction &x)
{
   std::ofstream output("solution.vtk");

   mesh.PrintVTK(output, 0);
   x.SaveVTK(output, "u", 0);
}

int main(int argc, char *argv[])
{
   Problem l;
   l.run();
}

// Exact solution, used for the Dirichlet BC.
double bdr_func(const Vector &p)
{
   /*
   double radius = sqrt((p(0)-0.5)*(p(0)-0.5) + p(1)*p(1));
   double phi;
   double alpha = 1.0/2.0;

   if(p(1) < 0)
   {
      phi = 2*M_PI+atan2(p(1),p(0)-0.5);
   }
   else
   {
      phi = atan2(p(1),p(0)-0.5);
   }

   return pow(radius,alpha) * sin(alpha * phi) *  (p(2)*p(2));
   */
   /*
   return exp(-10 * (p(0) + p(1))) * (p(2) * p(2));
   */
   double k = 8.0;
   return sin(k*p(0)) * cos(2*k*p(1)) * exp(p(2)); 
}

// Right hand side function
double rhs_func(const Vector &p)
{
   /*
   return -(200 * (p(2) * p(2)) + 2) * exp(-10 * (p(0) + p(1)));
   */
   double k = 8.0;
   return (k * k + 4 * k - 1) * sin(k * p(0)) * cos(2 * k * p(1)) * exp(p(2));
}
