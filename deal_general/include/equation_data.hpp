#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/convergence_table.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/refinement.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/numerics/smoothness_estimator.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#pragma once

using namespace dealii;
//----------------------------------------------------------------
//Define the Boundary condition function that is to be applied on this Problem.
//----------------------------------------------------------------
template <int dim>
class BoundaryValues : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
    //Formulate the boundary function
    
    /*
    //Problem with singularity 
    double alpha = 1.0/2.0;
    double radius = sqrt((p(0)-0.5)*(p(0)-0.5) + p(1)*p(1));
    double phi;
    if(p(1) < 0)
    {
        phi = 2*M_PI + atan2(p(1), p(0)-0.5);
    }
    else
    {
        phi = atan2(p(1), p(0)-0.5);
    }

    return pow(radius, alpha) * sin(alpha * phi) * (p(2)*p(2));
    */

    /*
   //Problem using the highly oszilating function
   double k = 8.0;
   return sin(k*p(0)) * cos(2*k*p(1)) * exp(p(2));
   */

    
    //Problem using the exponential function
    return exp(-10 * (p(0) + p(1))) * (p(2) * p(2));
    
}

//------------------------------
//Define the the right hand side function of the Problem.
//------------------------------
template <int dim>
class RHS_function : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double RHS_function<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
    //Formulate right hand side function

    /*
    return -2.0;
    */

    /*
    double k = 8.0;
    return (k * k + 4 * k - 1) * sin(k * p(0)) * cos(2 * k * p(1)) * exp(p(2));
    */

   
    return -(200 * (p(2) * p(2)) + 2) * exp(-10 * (p(0) + p(1)));
    
}

//------------------------------
//Define the exact soliution of the Problem., in order to calculate the exact error.
//------------------------------
template <int dim>
class Solution : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double Solution<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
    //Formulate the boundary function
    
    /*
    //Problem with singularity 
    double alpha = 1.0/2.0;
    double radius = sqrt((p(0)-0.5)*(p(0)-0.5)+ p(1)*p(1));
    double phi;
    if(p(1) < 0)
    {
        phi = 2*M_PI + atan2(p(1), p(0)-0.5);
    }
    else
    {
        phi = atan2(p(1), p(0)-0.5);
    }

    return pow(radius, alpha) * sin(alpha * phi) * (p(2)*p(2));
    */

    /*
   //Problem using the highly oszilating function
   double k = 8.0;
   return sin(k*p(0)) * cos(2*k*p(1)) * exp(p(2));
   */
   
    //Problem using the exponential function
    return exp(-10 * (p(0) + p(1))) * (p(2) * p(2));
    
}
