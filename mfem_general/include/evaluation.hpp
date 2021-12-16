//Created by Patrik Rác
//This file contains the cleasses and structs neccessary for evaluation

using namespace mfem;
using namespace std;
//--------------------------------------------------------------
//Class for evaluating the approximate solution at a given point. Only applicable if the point is a node of the grid.
//--------------------------------------------------------------
class PointValueEvaluation
{
public:
   PointValueEvaluation(const vector<double> &evaluation_point);
   double operator()(GridFunction &x, Mesh &mesh) const;

private:
   const vector<double> evaluation_point;
};

PointValueEvaluation::PointValueEvaluation(const vector<double> &evaluation_point) : evaluation_point(evaluation_point)
{
}

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

//Metrics to be collected for later graphing.
typedef struct error_values
{
   int cycle;
   int cells;
   int dofs;
   double max_error;
   double l2_error;
   double error_p1;
   double error_p2;
   double error_p3;
   double relative_error;
} error_values;