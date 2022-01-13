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
   //Class for the specification of the problem
   //Initialized with the hysteresis (Factor for derefinement), the max element error used for Refinement, the order of elements to be used,
   //and the three postprocessors required to calculate the pointwise error.
   //----------------------------------------------------------------
   class Poisson
   {
   public:
      Poisson(int num_procs, int myid, int order, int max_iters) : num_procs(num_procs), myid(myid),
                                                                   max_dofs(max_iters), reorder_mesh(0), nc_simplices(false),
                                                                   hysteresis(0.2), max_elem_error(1.0e-12), order(order),
                                                                   postprocessor1({0.125, 0.125, 0.125}), postprocessor2({0.25, 0.25, 0.25}), postprocessor3({0.5, 0.5, 0.5})
      {
      }
      void run();

   private:
      ParMesh *make_mesh();

      void solve(ParLinearForm &f, ParFiniteElementSpaceHierarchy& fespaces, ParGridFunction &x, FunctionCoefficient &bdr);
      bool refine(ParLinearForm &f, ParFiniteElementSpaceHierarchy& fespaces, ParGridFunction &x, ParGridFunction &error_zero);

      void exact_error(int cycle, int dofs, ParFiniteElementSpaceHierarchy& fespaces, ParGridFunction &x, ParGridFunction &error_zero, FunctionCoefficient &u);

      //void output_table();
      void vtk_output(ParFiniteElementSpaceHierarchy& fespaces, ParGridFunction &x);

      //Configuration parameters

      //Derefinement safety coefficient
      double hysteresis;
      double max_elem_error;
      int order;
      PointValueEvaluation postprocessor1, postprocessor2, postprocessor3;
      int max_dofs;

      //Data parameters
      vector<error_values> table_vector;

      //Parallel Parameters
      //Reorder elements
      int reorder_mesh;
      bool nc_simplices;

      //MPI
      int num_procs;
      int myid;
   };

}