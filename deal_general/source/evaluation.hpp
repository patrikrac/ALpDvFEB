//Created by Patrik Rác
//Definition of the PointValueEvaluation class and the metrics data structure
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/vector.h>
#pragma once

using namespace dealii;

//----------------------------------------------------------------
//Class used to evaluate the approx. solution at a given point (If node exists).
//----------------------------------------------------------------
template <int dim>
class PointValueEvaluation
{
public:
    PointValueEvaluation(const Point<dim> &evaluation_point)  : evaluation_point(evaluation_point) {}
    double operator()(const DoFHandler<dim> &dof_handler, const Vector<double> &solution) const;

private:
    const Point<dim> evaluation_point;
};


//Metrics to be collected fot later plots or diagramms
typedef struct metrics
{
    double max_error;
    double l2_error;
    double relative_error;
    double error_p1;
    double error_p2;
    double error_p3;
    int cycle;
    int n_dofs;
} metrics;