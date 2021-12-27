//Created by Patrik Rác
//Implementation of the evaluation class
#include "evaluation.hpp"

using namespace mfem;
using namespace std;

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
         for (int k = 0; k < evaluation_point.size(); k++)
         {
            if (evaluation_point[k] != vert_coords(k, j))
            {
               coords_match = false;
               break;
            }
         }

         if (coords_match)
         {
            return vert_vals(j);
         }
      }
   }

   return 1e20;
}