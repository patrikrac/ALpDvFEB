#pragma once
using namespace dealii;
//----------------------------------------------------------------
//Class used to evaluate the approx. solution at a given point (If node exists).
//----------------------------------------------------------------
template <int dim>
class PointValueEvaluation
{
public:
    PointValueEvaluation(const Point<dim> &evaluation_point);
    double operator()(const DoFHandler<dim> &dof_handler, const Vector<double> &solution) const;

private:
    const Point<dim> evaluation_point;
};

template <int dim>
PointValueEvaluation<dim>::PointValueEvaluation(const Point<dim> &evaluation_point) : evaluation_point(evaluation_point)
{
}

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

//Metrics to be collected fot later plots or diagramms
typedef struct metrics
{
    double error;
    double relative_error;
    double error_p1;
    double error_p2;
    double error_p3;
    int cycle;
    int n_dofs;
} metrics;