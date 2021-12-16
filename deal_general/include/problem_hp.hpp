#include "data.hpp"
#include "evaluation.hpp"
#pragma once

using namespace dealii;
//------------------------------
//Class which stores all vital parameters and has all functions to solve the problem using hp-adaptivity
//------------------------------
template <int dim>
class ProblemHP
{
public:
    ProblemHP(int);
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
    void output_vtk(const unsigned int cycle);
    void output_results();

    int max_iterations;

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
    PointValueEvaluation<dim> postprocessor1;
    PointValueEvaluation<dim> postprocessor2;
    PointValueEvaluation<dim> postprocessor3;
    std::vector<metrics> convergence_vector;
};