/*  ----------------------------------------------------------------
Created by Patrik Rac
Programm solving a partial differential equation of arbitrary dimensionality using the functionality of the deal.ii FE-library.
The classes defined here can be modified in order to solve each specific Problem.
---------------------------------------------------------------- */
#include "data.hpp"
#include "evaluation.hpp"
#pragma once

using namespace dealii;
//------------------------------
//Class which stores and solves the problem using only h-adaptivity.
//------------------------------
template <int dim>
class Problem
{
public:
    Problem(int, int);
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
};