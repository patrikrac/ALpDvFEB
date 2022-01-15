//Created by Patrik Rác
//This file contains the cleasses and structs neccessary for evaluation
#include "mfem.hpp"
#pragma once

using namespace mfem;
using namespace std;
//--------------------------------------------------------------
//Class for evaluating the approximate solution at a given point. Only applicable if the point is a node of the grid.
//--------------------------------------------------------------
class PointValueEvaluation
{
public:
   PointValueEvaluation(const vector<double> &evaluation_point) : evaluation_point(evaluation_point) {}
   double operator()(ParGridFunction &x, ParMesh &mesh) const;

private:
   const vector<double> evaluation_point;
};

//Metrics to be collected for later graphing.
typedef struct error_values
{
   int cycle;
   int cells;
   int dofs;
   double solution_time;
   double refinement_time;
   double max_error;
   double l2_error;
   double error_p1;
   double error_p2;
   double error_p3;
} error_values;
