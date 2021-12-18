//Created by Patrik Rác
//Function Headers definitions for the poisson problem definition

#include <deal.II/base/function.h>
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


//------------------------------
//Define the the right hand side function of the Problem.
//------------------------------
template <int dim>
class RHS_function : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};


//------------------------------
//Define the exact soliution of the Problem., in order to calculate the exact error.
//------------------------------
template <int dim>
class Solution : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};
