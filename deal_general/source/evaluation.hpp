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
    PointValueEvaluation(const Point<dim> &evaluation_point) : evaluation_point(evaluation_point) {}
    double operator()(const DoFHandler<dim> &dof_handler, const LinearAlgebraPETSc::MPI::Vector &solution) const;

private:
    const Point<dim> evaluation_point;
};

//Metrics to be collected fot later plots or diagramms
typedef struct metrics
{
    double max_error;
    double l2_error;
    double solution_time;
    double refinement_time;
    double total_time;
    double error_p1;
    double error_p2;
    double error_p3;
    int cycle;
    int cells;
    int n_dofs;
} metrics;

template <int dim>
double PointValueEvaluation<dim>::operator()(const DoFHandler<dim> &dof_handler, const LinearAlgebraPETSc::MPI::Vector &solution) const
{
    double point_value = 1e20;

    bool eval_point_found = false;
    for (const auto &cell : dof_handler.active_cell_iterators())
        if (!eval_point_found)
            if (cell->is_locally_owned())
            {
                for (const auto vertex : cell->vertex_indices())
                    if (cell->vertex(vertex) == evaluation_point)
                    {
                        point_value = solution(cell->vertex_dof_index(vertex, 0));
                        eval_point_found = true;
                        break;
                    }
            }

    return point_value;
}