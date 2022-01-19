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
<<<<<<< HEAD
    PointValueEvaluation(const Point<dim> &evaluation_point)  : evaluation_point(evaluation_point) {}
    double operator()(const DoFHandler<dim> &dof_handler, const Vector<double> &solution) const;
=======
    PointValueEvaluation(const Point<dim> &evaluation_point) : evaluation_point(evaluation_point) {}
    double operator()(const DoFHandler<dim> &dof_handler, const LinearAlgebraPETSc::MPI::Vector &solution) const;
>>>>>>> parallel

private:
    const Point<dim> evaluation_point;
};

//Metrics to be collected fot later plots or diagramms
typedef struct metrics
{
    double max_error;
    double l2_error;
<<<<<<< HEAD
    double relative_error;
=======
    double solution_time;
    double refinement_time;
    double assembly_time;
>>>>>>> parallel
    double error_p1;
    double error_p2;
    double error_p3;
    int cycle;
<<<<<<< HEAD
    int n_dofs;
} metrics;


template <int dim>
double PointValueEvaluation<dim>::operator()(const DoFHandler<dim> &dof_handler, const Vector<double> &solution) const
=======
    int cells;
    int n_dofs;
} metrics;

template <int dim>
double PointValueEvaluation<dim>::operator()(const DoFHandler<dim> &dof_handler, const LinearAlgebraPETSc::MPI::Vector &solution) const
>>>>>>> parallel
{
    double point_value = 1e20;

    bool eval_point_found = false;
    for (const auto &cell : dof_handler.active_cell_iterators())
        if (!eval_point_found)
<<<<<<< HEAD
            for (const auto vertex : cell->vertex_indices())
                if (cell->vertex(vertex) == evaluation_point)
                {
                    point_value = solution(cell->vertex_dof_index(vertex, 0));
                    eval_point_found = true;
                    break;
                }
=======
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
>>>>>>> parallel

    return point_value;
}