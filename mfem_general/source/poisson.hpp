//Created by Patrik Rác
//This file contains the class definition for the problem
//Definition of the Poisson class containing the necessarystructures to silve the problem using the MFEM Library
//The class is written inside the AspDEQuFEL namespace
#include "mfem.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>

#include "problem.hpp"
#include "evaluation.hpp"
#pragma once

namespace AspDEQuFEL
{

   using namespace mfem;
   using namespace std;
   //----------------------------------------------------------------
   //Class for the Poisson AMR solver
   //----------------------------------------------------------------
   class Poisson
   {
   public:
      Poisson(int num_procs, int myid, int order, int max_iters) : num_procs(num_procs), myid(myid),
                                                                   max_dofs(max_iters), reorder_mesh(0), nc_simplices(false),
                                                                   max_elem_error(1.0e-12), order(order),
                                                                   postprocessor1({0.125, 0.125, 0.125}), postprocessor2({0.25, 0.25, 0.25}), postprocessor3({0.5, 0.5, 0.5})
      {
      }
      void run();

   private:
      void make_mesh();

      void update(ParBilinearForm &a, ParLinearForm &f, ParFiniteElementSpace &fespace, ParGridFunction &x);
      void assemble(ParBilinearForm &a, ParLinearForm &f);
      void solve(ParBilinearForm &a, ParLinearForm &f, ParFiniteElementSpace &fespace, ParGridFunction &x, Array<int> &ess_bdr, FunctionCoefficient &bdr);
      bool refine(ThresholdRefiner &refiner);

      void exact_error(int cycle, int dofs, double solution_time, double refinement_time,  ParGridFunction &x, FunctionCoefficient &u);

      void output_table();
      void vtk_output(ParGridFunction &x);

      //Configuration parameters

      //Derefinement safety coefficient
      double max_elem_error;
      int order;
      PointValueEvaluation postprocessor1, postprocessor2, postprocessor3;
      int max_dofs;

      //Data parameters
      vector<error_values> table_vector;
      ParMesh *pmesh;

      //Parallel Parameters
      //Reorder elements
      int reorder_mesh;
      bool nc_simplices;

      //MPI
      int num_procs;
      int myid;
   };

}