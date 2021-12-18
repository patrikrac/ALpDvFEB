//Created by Patrik Rác
//Definition of the Poisson problem solver class using hp-refinement
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

#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>

#include "evaluation.hpp"
#include "Timer.hpp"
#pragma once

namespace AspDEQuFEL
{
    using namespace dealii;
    //------------------------------
    //Class which stores all vital parameters and has all functions to solve the problem using hp-adaptivity
    //------------------------------
    template <int dim>
    class PoissonHP
    {
    public:
        PoissonHP(int);
        ~PoissonHP();

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

        void startTimer();
        double printTimer();

        int max_dofs;

        timing::Timer timer; 

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


    
}
