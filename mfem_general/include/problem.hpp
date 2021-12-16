//Created by Patrik Rác
//This file contains the class definition for the problem
#include "mfem.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>

#include "data.hpp"
#include "evaluation.hpp"

using namespace mfem;
using namespace std;
//----------------------------------------------------------------
//Class for the specification of the problem
//Initialized with the hysteresis (Factor for derefinement), the max element error used for Refinement, the order of elements to be used,
//and the three postprocessors required to calculate the pointwise error.
//----------------------------------------------------------------
class Problem
{
public:
   Problem(int order, int iters) : hysteresis(0), max_elem_error(1.0e-12), order(order), max_iters(iters),
               postprocessor1({0.125, 0.125, 0.125}), postprocessor2({0.25, 0.25, 0.25}), postprocessor3({0.5, 0.5, 0.5})
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
   void vtk_output(GridFunction &x, int &cycle);

   //Configuration parameters
   double hysteresis; //derefinement safety coefficient
   double max_elem_error;
   int order;
   int max_iters;
   PointValueEvaluation postprocessor1, postprocessor2, postprocessor3;

   //Data parameters
   Mesh mesh;
   vector<error_values> table_vector;
};