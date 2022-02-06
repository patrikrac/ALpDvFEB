//Created by Patrik Rác
//Implementation of the problem defintitions containing the dirichlet boundary condition, as well as the right hand side function.
#include "problem.hpp"

// Exact solution, used for the dirichlet BC.
double bdr_func(const Vector &p)
{
   /*
   double radius = sqrt((p(0)-0.5) * (p(0)-0.5) + p(1) * p(1));
   double phi;
   double alpha = 1.0/2.0;

   if (p(1) < 0)
   {
      phi = 2 * M_PI + atan2(p(1), (p(0)-0.5));
   }
   else
   {
      phi = atan2(p(1), (p(0)-0.5));
   }

   return pow(radius, alpha) * sin(alpha * phi) * (p(2) * p(2));
   */

   /*
   return exp(-10 * (p(0) + p(1))) * (p(2) * p(2));
   */

   
   double k = 10.0;
   return sin(k*p(0)) * cos(2*k*p(1)) * exp(p(2)); 
   
   /*
   //Experimental atan problem
   double radius = sqrt(p(0) * p(0)+ p(1) * p(1));
   double r_0 = 1.0/4.0;
   double alpha = 100;
   return atan(alpha*(radius-r_0)) * (p(2) * p(2));
   */
}

// Right hand side function
double rhs_func(const Vector &p)
{

   /*
   return -(200 * (p(2) * p(2)) + 2) * exp(-10 * (p(0) + p(1)));
   */

   
   double k = 10.0;
   return (5*k*k - 1) * sin(k * p(0)) * cos(2 * k * p(1)) * exp(p(2));
   

   /*
   double radius = sqrt((p(0)-0.5) * (p(0)-0.5) + p(1) * p(1));
   double phi;
   double alpha = 1.0/2.0;

   if (p(1) < 0)
   {
      phi = 2 * M_PI + atan2(p(1), (p(0)-0.5));
   }
   else
   {
      phi = atan2(p(1), (p(0)-0.5));
   }
   
   return -2.0 * pow(radius, alpha) * sin(alpha * phi);
   */
  /*
   double radius = sqrt(p(0)*p(0)+ p(1)*p(1));
   double r_0 = 1.0/4.0;
   double alpha = 100;
   double diff = radius - r_0;
   return -(((1.0/radius)*(alpha / (1+pow(alpha,2)*pow(diff,2)))
                  + ((2.0*pow(alpha,3)*diff) / (pow(1+pow(alpha,2)*pow(diff,2),2))))*(p(2) * p(2)) 
                  +2.0*atan(alpha*diff));
  */
}