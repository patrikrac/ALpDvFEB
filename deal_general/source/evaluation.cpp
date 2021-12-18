//Created by Patrik Rác
//Implementation of the evaulation class used for evaluating the estimated solution at one specific node
#include "evaluation.hpp"

using namespace dealii;

template <int dim>
double PointValueEvaluation<dim>::operator()(const DoFHandler<dim> &dof_handler, const Vector<double> &solution) const
{
    double point_value = 1e20;

    bool eval_point_found = false;
    for (const auto &cell : dof_handler.active_cell_iterators())
        if (!eval_point_found)
            for (const auto vertex : cell->vertex_indices())
                if (cell->vertex(vertex) == evaluation_point)
                {
                    point_value = solution(cell->vertex_dof_index(vertex, 0));
                    eval_point_found = true;
                    break;
                }

    return point_value;
}