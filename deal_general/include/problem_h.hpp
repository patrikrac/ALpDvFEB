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
    Problem();
    void run();

private:
    void make_grid();
    void setup_system();
    void assemble_system();
    int get_n_dof();
    void solve();
    void refine_grid();
    void calculate_exact_error(const unsigned int cycle);
    void output_results(const unsigned int cycle);
    void output_error();

    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    ConditionalOStream pcout;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;

    PETScWrappers::MPI::SparseMatrix system_matrix;

    PETScWrappers::MPI::Vector solution;
    PETScWrappers::MPI::Vector system_rhs;

    ConvergenceTable convergence_table;
    PointValueEvaluation<dim> postprocessor1;
    PointValueEvaluation<dim> postprocessor2;
    PointValueEvaluation<dim> postprocessor3;
    std::vector<metrics> convergence_vector;
};