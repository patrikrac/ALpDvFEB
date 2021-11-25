/*  ----------------------------------------------------------------
Created by Patrik Rac
Programm solving a partial differential equation of arbitrary dimensionality using the functionality of the deal.ii FE-library.
The classes defined here can be modified in order to solve each specific Problem.
---------------------------------------------------------------- */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/convergence_table.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/refinement.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/numerics/smoothness_estimator.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/meshworker/mesh_loop.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>

using namespace dealii;

template <int dim>
struct ScratchData
{
    ScratchData(const Mapping<dim> &mapping,
                const FiniteElement<dim> &fe,
                const unsigned int quadrature_degree,
                const UpdateFlags update_flags)
        : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags)
    {
    }
    ScratchData(const ScratchData<dim> &scratch_data)
        : fe_values(scratch_data.fe_values.get_mapping(),
                    scratch_data.fe_values.get_fe(),
                    scratch_data.fe_values.get_quadrature(),
                    scratch_data.fe_values.get_update_flags())
    {
    }
    FEValues<dim> fe_values;
};

struct CopyData
{
    unsigned int level;
    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    template <class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_rhs.reinit(dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        cell->get_active_or_mg_dof_indices(local_dof_indices);
        level = cell->level();
    }
};

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

//----------------------------------------------------------------
//Define the Boundary condition function that is to be applied on this Problem.
//----------------------------------------------------------------
template <int dim>
class BoundaryValues : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
    //Formulate the boundary function

    /*
    //Problem with singularity 
    double alpha = 1.0/2.0;
    double radius = sqrt((p(0)-0.5)*(p(0)-0.5) + p(1)*p(1));
    double phi;
    if(p(1) < 0)
    {
        phi = 2*M_PI + atan2(p(1), p(0)-0.5);
    }
    else
    {
        phi = atan2(p(1), p(0)-0.5);
    }

    return pow(radius, alpha) * sin(alpha * phi) * (p(2)*p(2));
    */

    //Problem using the highly oszilating function
    double k = 8.0;
    return sin(k * p(0)) * cos(2 * k * p(1)) * exp(p(2));

    /*
    //Problem using the exponential function
    return exp(-10 * (p(0) + p(1))) * (p(2) * p(2));
    */
}

//------------------------------
//Define the the right hand side function of the Problem.
//------------------------------
template <int dim>
class RHS_function : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double RHS_function<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
    //Formulate right hand side function

    /* 
    return -2.0;
    */

    double k = 8.0;
    return (k * k + 4 * k - 1) * sin(k * p(0)) * cos(2 * k * p(1)) * exp(p(2));

    /*
    return -(200 * (p(2) * p(2)) + 2) * exp(-10 * (p(0) + p(1)));
    */
}

//------------------------------
//Define the exact soliution of the Problem., in order to calculate the exact error.
//------------------------------
template <int dim>
class Solution : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double Solution<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
    //Formulate the boundary function

    /*
    //Problem with singularity 
    double alpha = 1.0/2.0;
    double radius = sqrt((p(0)-0.5)*(p(0)-0.5)+ p(1)*p(1));
    double phi;
    if(p(1) < 0)
    {
        phi = 2*M_PI + atan2(p(1), p(0)-0.5);
    }
    else
    {
        phi = atan2(p(1), p(0)-0.5);
    }

    return pow(radius, alpha) * sin(alpha * phi) * (p(2)*p(2));
    */

    //Problem using the highly oszilating function
    double k = 8.0;
    return sin(k * p(0)) * cos(2 * k * p(1)) * exp(p(2));

    /*
    //Problem using the exponential function
    return exp(-10 * (p(0) + p(1))) * (p(2) * p(2));
    */
}

//------------------------------
//Class which stores all vital parameters and has all functions to solve the problem using hp-adaptivity
//------------------------------
template <int dim>
class ProblemHP
{
public:
    ProblemHP();
    ~ProblemHP();

    void run();

private:
    void make_grid();
    void setup_system();
    void assemble_system();
    int get_n_dof();
    void solve();
    void refine_grid();
    void calculate_exact_error(const unsigned int cycle);
    void output_results();

    Triangulation<dim> triangulation;

    DoFHandler<dim> dof_handler;
    hp::FECollection<dim> fe_collection;
    hp::QCollection<dim> quadrature_collection;
    hp::QCollection<dim - 1> face_quadrature_collection;

    AffineConstraints<double> constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;

    const unsigned int max_degree;

    ConvergenceTable convergence_table;
    std::vector<metrics> convergence_vector;
};

//------------------------------
//The dof_handler manages enumeration and indexing of all degrees of freedom (relating to the given triangulation).
//Set an adequate maximum degree.
//------------------------------
template <int dim>
ProblemHP<dim>::ProblemHP() : dof_handler(triangulation), max_degree(dim <= 2 ? 7 : 5)
{
    for (unsigned int degree = 2; degree <= max_degree; degree++)
    {
        fe_collection.push_back(FE_Q<dim>(degree));
        quadrature_collection.push_back(QGauss<dim>(degree + 1));
        face_quadrature_collection.push_back(QGauss<dim - 1>(degree + 1));
    }
}

//------------------------------
//Destructor for the ProblemHP class
//------------------------------
template <int dim>
ProblemHP<dim>::~ProblemHP()
{
    dof_handler.clear();
}

//------------------------------
//Construct the Grid the ProblemHP is beeing solved on.
//Define the default coarsaty / refinement of the grid
//------------------------------
template <int dim>
void ProblemHP<dim>::make_grid()
{
    //Appropriate grid generation has to be implemented in here!
    //The default grid generated will be a unit square/cube depending on the dimensionality of the problem.
    GridGenerator::hyper_rectangle(triangulation, Point<3>(-0.5, 0.0, 0.0), Point<3>(0.5, 1.0, 1.0));

    triangulation.refine_global(2);

    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

//------------------------------
//Setup the system by initializing the solution and ProblemHP vectors with the right dimensionality and apply bounding constraints.
//Althogh system calls are equal to the ones of the non hp-version of the program, it has to be noted that the dof_handler is in hp-mode and therefore the calls differ internally.
//------------------------------
template <int dim>
void ProblemHP<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe_collection);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0, BoundaryValues<dim>(), constraints);

    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
}

//------------------------------
//Assemble the system by creating a quadrature rule for integeration, calculate local matrices using the appropriate weak formulations and assamble the global matrices.
//------------------------------
template <int dim>
void ProblemHP<dim>::assemble_system()
{
    hp::FEValues<dim> hp_fe_values(fe_collection,
                                   quadrature_collection,
                                   update_values | update_gradients | update_quadrature_points | update_JxW_values);

    RHS_function<dim> rhs;

    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;

    std::vector<types::global_dof_index> local_dof_indices;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        //Call for dofs is pushed until now, beceause each cell might differ
        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_matrix = 0;

        cell_rhs.reinit(dofs_per_cell);
        cell_rhs = 0;

        hp_fe_values.reinit(cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        rhs.value_list(fe_values.get_quadrature_points(), rhs_values);

        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; q_point++)
        {
            for (unsigned int i = 0; i < dofs_per_cell; i++)
            {
                for (unsigned int j = 0; j < dofs_per_cell; j++)
                {
                    cell_matrix(i, j) +=
                        (fe_values.shape_grad(i, q_point) * // grad phi_i(x_q)
                         fe_values.shape_grad(j, q_point) * // grad phi_j(x_q)
                         fe_values.JxW(q_point));           //dx
                }
                cell_rhs(i) += (fe_values.shape_value(i, q_point) * // phi_i(x_q)
                                rhs_values[q_point] *               // f(x_q)
                                fe_values.JxW(q_point));            //dx
            }
        }

        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}

//------------------------------
//Return the number of degrees of freedom of the current ProblemHP state.
//------------------------------
template <int dim>
int ProblemHP<dim>::get_n_dof()
{
    return dof_handler.n_dofs();
}

//------------------------------
//Set solving conditinos and define the solver. Then solve the given system.
//------------------------------
template <int dim>
void ProblemHP<dim>::solve()
{
    SolverControl solver_control(system_rhs.size(), 1e-12 * system_rhs.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);
}

//------------------------------
//Refine the Grid using a built in error estimator.
//------------------------------
template <int dim>
void ProblemHP<dim>::refine_grid()
{
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       face_quadrature_collection,
                                       std::map<types::boundary_id, const Function<dim> *>(),
                                       solution,
                                       estimated_error_per_cell);

    Vector<float> smoothness_indicators(triangulation.n_active_cells());
    FESeries::Fourier<dim> fourier = SmoothnessEstimator::Fourier::default_fe_series(fe_collection);
    SmoothnessEstimator::Fourier::coefficient_decay(fourier,
                                                    dof_handler,
                                                    solution,
                                                    smoothness_indicators);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);

    hp::Refinement::p_adaptivity_from_relative_threshold(dof_handler, smoothness_indicators, 0.2, 0.2);

    hp::Refinement::choose_p_over_h(dof_handler);

    triangulation.prepare_coarsening_and_refinement();
    hp::Refinement::limit_p_level_difference(dof_handler);

    triangulation.execute_coarsening_and_refinement();
}

//------------------------------
//Output the result using a vtk file format
//------------------------------
template <int dim>
void ProblemHP<dim>::output_results()
{
    DataOut<dim> data_out;

    Vector<float> fe_degrees(triangulation.n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_degrees(cell->active_cell_index()) = fe_collection[cell->active_fe_index()].degree;
    }

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u");
    data_out.add_data_vector(fe_degrees, "fe_degree");

    data_out.build_patches();

    std::ofstream output("solution.vtk");
    data_out.write_vtk(output);

    convergence_table.set_precision("Linfty", 3);
    convergence_table.set_precision("relativeLinfty", 3);
    convergence_table.set_scientific("Linfty", true);
    convergence_table.set_scientific("relativeLinfty", true);

    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("Linfty", "$L^\\infty$");
    convergence_table.set_tex_caption("relativeLinfty", "relative");

    std::ofstream error_table_file("error.tex");
    convergence_table.write_tex(error_table_file);

    std::ofstream output_custom("error_deal2.txt");

    output_custom << "$deal.ii$" << std::endl;
    output_custom << "$n_\\text{dof}$" << std::endl;
    output_custom << "$\\left\\|u - u_h\\right\\| _{L^\\infty}$" << std::endl;
    output_custom << convergence_vector.size() << std::endl;
    for (size_t i = 0; i < convergence_vector.size(); i++)
    {
        output_custom << convergence_vector[i].n_dofs << " " << convergence_vector[i].error << std::endl;
    }
}

//-------------------------------------------------------------
//Calculate the exact error using the Solution class at the given cycle
//-------------------------------------------------------------
template <int dim>
void ProblemHP<dim>::calculate_exact_error(const unsigned int cycle)
{
    Vector<float> difference_per_cell(triangulation.n_active_cells());

    const QTrapezoid<1> q_trapez;
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      quadrature_collection,
                                      VectorTools::Linfty_norm);
    const double Linfty_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::Linfty_norm);

    Vector<double> zero_vector(dof_handler.n_dofs());
    Vector<float> norm_per_cell(triangulation.n_active_cells());

    VectorTools::integrate_difference(dof_handler,
                                      zero_vector,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      quadrature_collection,
                                      VectorTools::Linfty_norm);

    const double Linfty_norm = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::Linfty_norm);

    const double relative_Linfty_error = Linfty_error / Linfty_norm;

    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();

    std::cout << "Cycle " << cycle << ':' << std::endl
              << "   Number of active cells:       " << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: " << n_dofs << std::endl;

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("Linfty", Linfty_error);
    convergence_table.add_value("relativeLinfty", relative_Linfty_error);

    metrics values = {};
    values.error = Linfty_error;
    values.relative_error = relative_Linfty_error;
    values.n_dofs = n_dofs;
    values.cycle = cycle;

    convergence_vector.push_back(values);
}

//------------------------------
//Execute the solving process with cylce refinement steps.
//------------------------------
template <int dim>
void ProblemHP<dim>::run()
{

    int cycle = 0;
    while (true)
    {
        if (cycle == 0)
        {
            make_grid();
        }
        else
        {
            refine_grid();
        }
        setup_system();
        assemble_system();
        solve();

        calculate_exact_error(cycle);

        //Netgen similar condition to reach desired number of degrees of freedom
        if (get_n_dof() > 1000000)
        {
            break;
        }

        cycle++;
    }

    output_results();
}

//------------------------------
//Class which stores and solves the problem using only h-adaptivity.
//------------------------------
template <int dim>
class Problem
{
public:
    Problem(const unsigned int degree);
    void run();

private:
    template <class Iterator>
    void cell_worker(const Iterator &cell, ScratchData<dim> &scratch_data, CopyData &copy_data);

    void make_grid();
    void setup_system();
    void assemble_system();
    void assemble_multigrid();
    int get_n_dof();
    void solve();
    void refine_grid();
    void calculate_exact_error(const unsigned int cycle);
    void output_results();

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;

    ConvergenceTable convergence_table;
    PointValueEvaluation<dim> postprocessor1;
    PointValueEvaluation<dim> postprocessor2;
    PointValueEvaluation<dim> postprocessor3;
    std::vector<metrics> convergence_vector;

    const unsigned int degree;

    MGLevelObject<SparsityPattern> mg_sparsity_patterns;
    MGLevelObject<SparsityPattern> mg_interface_sparsity_patterns;
    MGLevelObject<SparseMatrix<double>> mg_matrices;
    MGLevelObject<SparseMatrix<double>> mg_interface_matrices;
    MGConstrainedDoFs mg_constrained_dofs;
};

//------------------------------
//Initialize the problem with first order finite elements
//The dof_handler manages enumeration and indexing of all degrees of freedom (relating to the given triangulation)
//------------------------------
template <int dim>
Problem<dim>::Problem(const unsigned int degree) : triangulation(Triangulation<dim>::limit_level_difference_at_vertices),
                                                   fe(degree),
                                                   dof_handler(triangulation),
                                                   degree(degree),
                                                   postprocessor1(Point<dim>(0.125, 0.125, 0.125)), postprocessor2(Point<dim>(0.25, 0.25, 0.25)), postprocessor3(Point<dim>(0.5, 0.5, 0.5))
{
}

//------------------------------
//Construct the Grid the problem is beeing solved on.
//Define the default coarsaty / refinement of the grid
//------------------------------
template <int dim>
void Problem<dim>::make_grid()
{
    //Appropriate grid generation has to be implemented in here!
    //The default grid generated will be a unit square/cube depending on the dimensionality of the problem.

    GridGenerator::hyper_rectangle(triangulation, Point<3>(0.0, 0.0, 0.0), Point<3>(1.0, 1.0, 1.0));

    triangulation.refine_global(3);

    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

//------------------------------
//Setup the system by initializing the solution and problem vectors with the right dimensionality and apply bounding constraints.
//------------------------------
template <int dim>
void Problem<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();
    //std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(dof_handler, 0, BoundaryValues<dim>(), constraints);

    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof_handler);

    const std::set<types::boundary_id> boundary_ids = {0};
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, boundary_ids);

    const unsigned int n_levels = triangulation.n_levels();
    mg_interface_matrices.resize(0, n_levels - 1);
    mg_matrices.resize(0, n_levels - 1);
    mg_sparsity_patterns.resize(0, n_levels - 1);
    mg_interface_sparsity_patterns.resize(0, n_levels - 1);

    for (unsigned int level = 0; level < n_levels; ++level)
    {
        {
            DynamicSparsityPattern dsp(dof_handler.n_dofs(level),
                                       dof_handler.n_dofs(level));
            MGTools::make_sparsity_pattern(dof_handler, dsp, level);
            mg_sparsity_patterns[level].copy_from(dsp);
            mg_matrices[level].reinit(mg_sparsity_patterns[level]);
        }
        {
            DynamicSparsityPattern dsp(dof_handler.n_dofs(level),
                                       dof_handler.n_dofs(level));
            MGTools::make_interface_sparsity_pattern(dof_handler,
                                                     mg_constrained_dofs,
                                                     dsp,
                                                     level);
            mg_interface_sparsity_patterns[level].copy_from(dsp);
            mg_interface_matrices[level].reinit(
                mg_interface_sparsity_patterns[level]);
        }
    }
}

//------------------------------
//Helper class to enable multigrid
//------------------------------
template <int dim>
template <class Iterator>
void Problem<dim>::cell_worker(const Iterator &cell, ScratchData<dim> &scratch_data, CopyData &copy_data)
{
    FEValues<dim> &fe_values = scratch_data.fe_values;
    fe_values.reinit(cell);

    RHS_function<dim> rhs;

    const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points = fe_values.get_quadrature().size();
    copy_data.reinit(cell, dofs_per_cell);
    const std::vector<double> &JxW = fe_values.get_JxW_values();
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                copy_data.cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q) * // grad phi_j(x_q)
                     JxW[q]);                     //dx
            }
            copy_data.cell_rhs(i) += (fe_values.shape_value(i, q) *              // phi_i(x_q)
                                      rhs.value(fe_values.quadrature_point(q)) * // f(x_q)
                                      JxW[q]);
        }
    }
}

template <int dim>
void Problem<dim>::assemble_multigrid()
{
    MappingQ1<dim> mapping;
    const unsigned int n_levels = triangulation.n_levels();
    std::vector<AffineConstraints<double>> boundary_constraints(n_levels);
    for (unsigned int level = 0; level < n_levels; ++level)
    {
        IndexSet dofset;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, dofset);
        boundary_constraints[level].reinit(dofset);
        boundary_constraints[level].add_lines(mg_constrained_dofs.get_refinement_edge_indices(level));
        boundary_constraints[level].add_lines(mg_constrained_dofs.get_boundary_indices(level));
        boundary_constraints[level].close();
    }
    auto cell_worker =
        [&](const typename DoFHandler<dim>::level_cell_iterator &cell,
            ScratchData<dim> &scratch_data,
            CopyData &copy_data)
    {
        this->cell_worker(cell, scratch_data, copy_data);
    };
    auto copier = [&](const CopyData &cd)
    {
        boundary_constraints[cd.level].distribute_local_to_global(cd.cell_matrix, cd.local_dof_indices, mg_matrices[cd.level]);
        const unsigned int dofs_per_cell = cd.local_dof_indices.size();

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                if (mg_constrained_dofs.is_interface_matrix_entry(cd.level, cd.local_dof_indices[i], cd.local_dof_indices[j]))
                {
                    mg_interface_matrices[cd.level].add(cd.local_dof_indices[i], cd.local_dof_indices[j], cd.cell_matrix(i, j));
                }
    };

    const unsigned int n_gauss_points = degree + 1;
    ScratchData<dim> scratch_data(mapping, fe, n_gauss_points,
                                  update_values | update_gradients | update_JxW_values | update_quadrature_points);
    MeshWorker::mesh_loop(dof_handler.begin_mg(),
                          dof_handler.end_mg(),
                          cell_worker,
                          copier,
                          scratch_data,
                          CopyData(),
                          MeshWorker::assemble_own_cells);
}

//------------------------------
//Assemble the system by creating a quadrature rule for integeration, calculate local matrices using the appropriate weak formulations and assamble the global matrices.
//------------------------------
template <int dim>
void Problem<dim>::assemble_system()
{
    MappingQ1<dim> mapping;

    auto cell_worker =
        [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
            ScratchData<dim> &scratch_data,
            CopyData &copy_data)
    {
        this->cell_worker(cell, scratch_data, copy_data);
    };

    auto copier = [&](const CopyData &cd)
    {
        this->constraints.distribute_local_to_global(cd.cell_matrix,
                                                     cd.cell_rhs,
                                                     cd.local_dof_indices,
                                                     system_matrix,
                                                     system_rhs);
    };

    const unsigned int n_gauss_points = degree + 1;
    ScratchData<dim> scratch_data(mapping,
                                  fe,
                                  n_gauss_points,
                                  update_values | update_gradients | update_JxW_values | update_quadrature_points);

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          CopyData(),
                          MeshWorker::assemble_own_cells);
}

//------------------------------
//Return the number of degrees of freedom of the current problem state.
//------------------------------
template <int dim>
int Problem<dim>::get_n_dof()
{
    return dof_handler.n_dofs();
}

//------------------------------
//Set solving conditinos and define the solver. Then solve the given system.
//------------------------------
template <int dim>
void Problem<dim>::solve()
{
    MGTransferPrebuilt<Vector<double>> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    FullMatrix<double> coarse_matrix;
    coarse_matrix.copy_from(mg_matrices[0]);
    MGCoarseGridHouseholder<double, Vector<double>> coarse_grid_solver;
    coarse_grid_solver.initialize(coarse_matrix);

    using Smoother = PreconditionSOR<SparseMatrix<double>>;
    mg::SmootherRelaxation<Smoother, Vector<double>> mg_smoother;
    mg_smoother.initialize(mg_matrices);
    mg_smoother.set_steps(2);
    mg_smoother.set_symmetric(true);

    mg::Matrix<Vector<double>> mg_matrix(mg_matrices);
    mg::Matrix<Vector<double>> mg_interface_up(mg_interface_matrices);
    mg::Matrix<Vector<double>> mg_interface_down(mg_interface_matrices);
    Multigrid<Vector<double>> mg(mg_matrix, coarse_grid_solver, mg_transfer, mg_smoother, mg_smoother);

    mg.set_edge_matrices(mg_interface_down, mg_interface_up);

    PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>> preconditioner(dof_handler, mg, mg_transfer);

    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);

    solution = 0;
    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);
}

//------------------------------
//Refine the Grid using a built in error estimator
//------------------------------
template <int dim>
void Problem<dim>::refine_grid()
{
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(fe.degree + 1), {}, solution, estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);

    triangulation.execute_coarsening_and_refinement();
}

//------------------------------
//Output the result using a vtk file format
//------------------------------
template <int dim>
void Problem<dim>::output_results()
{
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u");

    data_out.build_patches();

    std::ofstream output("solution.vtk");
    data_out.write_vtk(output);

    convergence_table.set_precision("Linfty", 3);
    convergence_table.set_precision("relativeLinfty", 3);
    convergence_table.set_precision("error_p1", 3);
    convergence_table.set_precision("error_p2", 3);
    convergence_table.set_precision("error_p3", 3);
    convergence_table.set_scientific("Linfty", true);
    convergence_table.set_scientific("relativeLinfty", true);
    convergence_table.set_scientific("error_p1", true);
    convergence_table.set_scientific("error_p2", true);
    convergence_table.set_scientific("error_p3", true);

    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("Linfty", "$\\left\\|u_h - I_hu\\right\\| _{L^\\infty}$");
    convergence_table.set_tex_caption("relativeLinfty", "$\\frac{\\left\\|u_h - I_hu\\right\\| _{L^\\infty}}{\\left\\|I_hu\\right\\| _{L^\\infty}}$");
    convergence_table.set_tex_caption("error_p1", "$\\left\\|u_h(x_1) - I_hu(x_1)\\right\\| $");
    convergence_table.set_tex_caption("error_p2", "$\\left\\|u_h(x_2) - I_hu(x_2)\\right\\| $");
    convergence_table.set_tex_caption("error_p3", "$\\left\\|u_h(x_3) - I_hu(x_3)\\right\\| $");

    std::ofstream error_table_file("error.tex");
    convergence_table.write_tex(error_table_file);

    std::ofstream output_custom1("error_deal2.txt");

    output_custom1 << "$deal.ii$" << std::endl;
    output_custom1 << "$n_\\text{dof}$" << std::endl;
    output_custom1 << "$\\left\\|u_h - I_hu\\right\\| $" << std::endl;
    output_custom1 << convergence_vector.size() << std::endl;
    for (size_t i = 0; i < convergence_vector.size(); i++)
    {
        output_custom1 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error << std::endl;
    }
    output_custom1.close();

    std::ofstream output_custom2("error_deal2_p1.txt");

    output_custom2 << "$\\left\\|u_h(x_1) - I_hu(x_1)\\right\\| $" << std::endl;
    output_custom2 << "$n_\\text{dof}$" << std::endl;
    output_custom2 << "$\\left\\|u_h(x) - I_hu(x)\\right\\|$" << std::endl;
    output_custom2 << convergence_vector.size() << std::endl;
    for (size_t i = 0; i < convergence_vector.size(); i++)
    {
        output_custom2 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p1 << std::endl;
    }
    output_custom2.close();

    std::ofstream output_custom3("error_deal2_p2.txt");

    output_custom3 << "$\\left\\|u_h(x_2) - I_hu(x_2)\\right\\|$" << std::endl;
    output_custom3 << "$n_\\text{dof}$" << std::endl;
    output_custom3 << "$\\left\\|u_h(x) - I_hu(x)\\right\\| $" << std::endl;
    output_custom3 << convergence_vector.size() << std::endl;
    for (size_t i = 0; i < convergence_vector.size(); i++)
    {
        output_custom3 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p2 << std::endl;
    }
    output_custom3.close();

    std::ofstream output_custom4("error_deal2_p3.txt");

    output_custom4 << "$\\left\\|u_h(x_3) - I_hu(x_3)\\right\\|$" << std::endl;
    output_custom4 << "$n_\\text{dof}$" << std::endl;
    output_custom4 << "$\\left\\|u_h(x) - I_hu(x)\\right\\| $" << std::endl;
    output_custom4 << convergence_vector.size() << std::endl;
    for (size_t i = 0; i < convergence_vector.size(); i++)
    {
        output_custom4 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p3 << std::endl;
    }
    output_custom4.close();
}

//----------------------------------------------------------------
//Calculate the exact error usign the solution class.
//----------------------------------------------------------------
template <int dim>
void Problem<dim>::calculate_exact_error(const unsigned int cycle)
{
    Vector<float> difference_per_cell(triangulation.n_active_cells());

    const QTrapezoid<1> q_trapez;
    const QIterated<dim> q_iterated(q_trapez, fe.degree * 2 + 1);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      q_iterated,
                                      VectorTools::Linfty_norm);
    const double Linfty_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::Linfty_norm);

    Vector<double> zero_vector(dof_handler.n_dofs());
    Vector<float> norm_per_cell(triangulation.n_active_cells());

    VectorTools::integrate_difference(dof_handler,
                                      zero_vector,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      q_iterated,
                                      VectorTools::Linfty_norm);

    const double Linfty_norm = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::Linfty_norm);

    const double relative_Linfty_error = Linfty_error / Linfty_norm;

    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();

    double error_p1 = abs(postprocessor1(dof_handler, solution) - Solution<dim>().value(Point<dim>(0.125, 0.125, 0.125)));
    double error_p2 = abs(postprocessor2(dof_handler, solution) - Solution<dim>().value(Point<dim>(0.25, 0.25, 0.25)));
    double error_p3 = abs(postprocessor3(dof_handler, solution) - Solution<dim>().value(Point<dim>(0.5, 0.5, 0.5)));

    std::cout << "Cycle " << cycle << ':' << std::endl
              << "   Number of active cells:       " << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: " << n_dofs << std::endl;

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("Linfty", Linfty_error);
    convergence_table.add_value("relativeLinfty", relative_Linfty_error);
    convergence_table.add_value("error_p1", error_p1);
    convergence_table.add_value("error_p2", error_p2);
    convergence_table.add_value("error_p3", error_p3);

    metrics values = {};
    values.error = Linfty_error;
    values.relative_error = relative_Linfty_error;
    values.error_p1 = error_p1;
    values.error_p2 = error_p2;
    values.error_p3 = error_p3;
    values.n_dofs = n_dofs;
    values.cycle = cycle;

    convergence_vector.push_back(values);
}

//------------------------------
//Run the problem.
//------------------------------
template <int dim>
void Problem<dim>::run()
{

    int cycle = 0;
    while (true)
    {
        if (cycle == 0)
        {
            make_grid();
        }
        else
        {
            refine_grid();
        }
        setup_system();
        assemble_system();
        assemble_multigrid();
        solve();

        calculate_exact_error(cycle);

        if (get_n_dof() > 100000)
        {
            break;
        }

        cycle++;
    }

    output_results();
}

int main(int argc, char **argv)
{
    Problem<3> l(2);
    l.run();
    return 0;
}