//Created by Patrik Rác
//This file contains the problem definitions

using namespace mfem;
// Exact solution, used for the Dirichlet BC.
double bdr_func(const Vector &p)
{
   
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
   

   /*
   return exp(-10 * (p(0) + p(1))) * (p(2) * p(2));
   */

   /*
   double k = 8.0;
   return sin(k*p(0)) * cos(2*k*p(1)) * exp(p(2)); 
   */
}

// Right hand side function
double rhs_func(const Vector &p)
{
   /*
   return -(200 * (p(2) * p(2)) + 2) * exp(-10 * (p(0) + p(1)));
   */

   /*
   double k = 8.0;
   return (5*k*k - 1) * sin(k * p(0)) * cos(2 * k * p(1)) * exp(p(2));
   */
  
   
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
   
}