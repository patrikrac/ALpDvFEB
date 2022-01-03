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
      Poisson(int order, int max_dof) : hysteresis(0), max_elem_error(1.0e-12), order(order), max_dofs(max_dof),
                                        postprocessor1({0.125, 0.125, 0.125}), postprocessor2({0.25, 0.25, 0.25}), postprocessor3({0.5, 0.5, 0.5})
      {
      }
      void run();

   private:
      Mesh *make_mesh();

      void update(LinearForm &f, FiniteElementSpaceHierarchy& fespaces, GridFunction &x, GridFunction &error_zero);
      void solve(LinearForm &f, FiniteElementSpaceHierarchy& fespaces, GridFunction &x, Array<int> &ess_bdr, FunctionCoefficient &bdr);
      bool refine(LinearForm &f, FiniteElementSpaceHierarchy& fespaces, GridFunction &x, GridFunction &error_zero);

      void exact_error(int cycle, int dofs, FiniteElementSpaceHierarchy &fespaces, GridFunction &x, GridFunction &error_zero, FunctionCoefficient &u);

      void output_table();
      void vtk_output(FiniteElementSpaceHierarchy &fespaces, GridFunction &x, int &cycle);

      //Configuration parameters
      double hysteresis; //derefinement safety coefficient
      double max_elem_error;
      int order;
      int max_dofs;
      PointValueEvaluation postprocessor1, postprocessor2, postprocessor3;

      //Data parameters
      vector<error_values> table_vector;
   };
}