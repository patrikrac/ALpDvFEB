//Created by Patrik Rác
//Implementation of the Boundary value, right hand side and exact solution functions of the problem definition
#include "problem.hpp"

using namespace dealii;

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
    //Formulate the boundary function

    //Problem with singularity 
    double alpha = 1.0/2.0;
    double radius = sqrt((p(0)-0.5)*(p(0)-0.5) + p(1)*p(1));
    double phi;
    if(p(1) < 0)
    {
        phi = 2*M_PI + atan2(p(1), (p(0)-0.5));
    }
    else
    {
        phi = atan2(p(1), (p(0)-0.5));
    }

    return pow(radius, alpha) * sin(alpha * phi) * (p(2)*p(2));
    

    /*
   //Problem using the highly oszilating function
   double k = 8.0;
   return sin(k*p(0)) * cos(2*k*p(1)) * exp(p(2));
   */

    /*
    //Problem using the exponential function
    return exp(-10 * (p(0) + p(1))) * (p(2) * p(2));
    */
    
}


template <int dim>
double RHS_function<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
    //Formulate right hand side function

    double alpha = 1.0/2.0;
    double radius = sqrt((p(0)-0.5)*(p(0)-0.5) + p(1)*p(1));
    double phi;
    if(p(1) < 0)
    {
        phi = 2*M_PI + atan2(p(1), (p(0)-0.5));
    }
    else
    {
        phi = atan2(p(1), (p(0)-0.5));
    }

    return -2.0*pow(radius, alpha) * sin(alpha * phi);
    

    /*
    double k = 8.0;
    return (5*k*k- 1) * sin(k * p(0)) * cos(2 * k * p(1)) * exp(p(2));
    */

   /*
    return -(200 * (p(2) * p(2)) + 2) * exp(-10 * (p(0) + p(1)));
    */
}


template <int dim>
double Solution<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
    //Formulate the boundary function
    
    //Problem with singularity
    double alpha = 1.0/2.0;
    double radius = sqrt((p(0)-0.5)*(p(0)-0.5) + p(1)*p(1));
    double phi;
    if(p(1) < 0)
    {
        phi = 2*M_PI + atan2(p(1), (p(0)-0.5));
    }
    else
    {
        phi = atan2(p(1), (p(0)-0.5));
    }

    return pow(radius, alpha) * sin(alpha * phi) * (p(2)*p(2));

    /*
   //Problem using the highly oszilating function
   double k = 8.0;
   return sin(k*p(0)) * cos(2*k*p(1)) * exp(p(2));
   */

    /*
    //Problem using the exponential function
    return exp(-10 * (p(0) + p(1))) * (p(2) * p(2));
    */
}