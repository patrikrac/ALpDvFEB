/*  ----------------------------------------------------------------
Created by Patrik Rac
Programm solving a partial differential equation of arbitrary dimensionality using the functionality of the deal.ii FE-library.
The classes defined here can be modified in order to solve each specific Problem.
---------------------------------------------------------------- */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/base/index_set.h>

#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>

#include "evaluation.hpp"
#include "problem.hpp"

#ifdef USE_TIMING
#include "Timer.hpp"
#endif
#pragma once

namespace AspDEQuFEL
{
    using namespace dealii;
    //------------------------------
    //Class which stores and solves the problem using only h-adaptivity.
    //------------------------------
    template <int dim>
    class Poisson
    {
    public:
        Poisson(int, int);
        void run();

    private:
        void make_grid();

        void setup_system();
        void assemble_system();
        int get_n_dof();
        void solve();
        void refine_grid();
        void calculate_exact_error(const unsigned int cycle, double solution_time, double refinement_time);
        void output_results(const unsigned int cycle) const;
        void output_error();

#ifdef USE_TIMING
        void startTimer();
        double printTimer();

        timing::Timer timer;
#endif

        int max_dofs;

        MPI_Comm mpi_communicator;
        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;

        ConditionalOStream pcout;

        parallel::distributed::Triangulation<dim> triangulation;
        FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;
        AffineConstraints<double> constraints;

        LinearAlgebraPETSc::MPI::SparseMatrix system_matrix;

        LinearAlgebraPETSc::MPI::Vector local_solution;
        LinearAlgebraPETSc::MPI::Vector system_rhs;

        ConvergenceTable convergence_table;
        PointValueEvaluation<dim> postprocessor1;
        PointValueEvaluation<dim> postprocessor2;
        PointValueEvaluation<dim> postprocessor3;
        std::vector<metrics> convergence_vector;
    };

    //------------------------------
    //Initialize the problem with first order finite elements
    //The dof_handler manages enumeration and indexing of all degrees of freedom (relating to the given triangulation)
    //------------------------------
    template <int dim>
    Poisson<dim>::Poisson(int order, int max_dof) : max_dofs(max_dof),
                                                    mpi_communicator(MPI_COMM_WORLD),
                                                    triangulation(mpi_communicator, typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::smoothing_on_refinement | Triangulation<dim>::smoothing_on_coarsening)),
                                                    n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
                                                    this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
                                                    pcout(std::cout, (this_mpi_process == 0)),
                                                    fe(order),
                                                    dof_handler(triangulation),
                                                    postprocessor1(Point<3>(0.125, 0.125, 0.125)),
                                                    postprocessor2(Point<3>(0.25, 0.25, 0.25)),
                                                    postprocessor3(Point<3>(0.5, 0.5, 0.5))
    {
    }

#ifdef USE_TIMING
    //--------------------------------
    // Starts or resets the current clock.
    //--------------------------------
    template <int dim>
    void Poisson<dim>::startTimer()
    {
        timer.reset();
    }

    //--------------------------------
    // Prints the current value of the clock
    //--------------------------------
    template <int dim>
    double Poisson<dim>::printTimer()
    {
        double time = timer.elapsed();
        std::cout << "Calculation took " << time << " seconds." << std::endl;
        return time;
    }
#endif

    //------------------------------
    //Construct the Grid the problem is beeing solved on.
    //Define the default coarsaty / refinement of the grid
    //------------------------------
    template <int dim>
    void Poisson<dim>::make_grid()
    {
        //The default grid generated will be a unit square/cube depending on the dimensionality of the problem.

        GridGenerator::hyper_cube(triangulation, 0, 1);

        triangulation.refine_global(3);

        pcout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
    }

    //------------------------------
    //Setup the system by initializing the solution and problem vectors with the right dimensionality and apply bounding constraints.
    //------------------------------
    template <int dim>
    void Poisson<dim>::setup_system()
    {
        dof_handler.distribute_dofs(fe);
        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        local_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        system_rhs.reinit(locally_owned_dofs, mpi_communicator);

        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        VectorTools::interpolate_boundary_values(dof_handler, 0, BoundaryValues<dim>(), constraints);

        constraints.close();

        DynamicSparsityPattern dsp(locally_relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
        SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

        system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
    }

    //------------------------------
    //Assemble the system by creating a quadrature rule for integeration, calculate local matrices using the appropriate weak formulations and assamble the global matrices.
    //------------------------------
    template <int dim>
    void Poisson<dim>::assemble_system()
    {
        const QGauss<dim> quadrature_formula(fe.degree + 1);
        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values | update_gradients | update_quadrature_points | update_JxW_values);

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        RHS_function<dim> rhs;

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                cell_matrix = 0;
                cell_rhs = 0;
                fe_values.reinit(cell);

                for (const unsigned int q_index : fe_values.quadrature_point_indices())
                {

                    const auto &x_q = fe_values.quadrature_point(q_index);

                    for (const unsigned int i : fe_values.dof_indices())
                    {
                        for (const unsigned int j : fe_values.dof_indices())
                        {
                            cell_matrix(i, j) +=
                                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                                 fe_values.JxW(q_index));           //dx
                        }

                        cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                        rhs.value(x_q) *                    // f(x_q)
                                        fe_values.JxW(q_index));            // dx
                    }
                }

                //Now add the calculated local matrix to the global sparse matrix
                cell->get_dof_indices(local_dof_indices);
                constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
            }
        }

        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
    }

    //------------------------------
    //Return the number of degrees of freedom of the current problem state.
    //------------------------------
    template <int dim>
    int Poisson<dim>::get_n_dof()
    {
        return dof_handler.n_dofs();
    }

    //------------------------------
    //Set solving conditinos and define the solver. Then solve the given system.
    //------------------------------
    template <int dim>
    void Poisson<dim>::solve()
    {
        LinearAlgebraPETSc::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

        SolverControl solver_control(2000, 1e-12);
        LinearAlgebraPETSc::SolverCG solver(solver_control, mpi_communicator);

        LinearAlgebraPETSc::MPI::PreconditionAMG preconditioner;
        LinearAlgebraPETSc::MPI::PreconditionAMG::AdditionalData data;
        data.symmetric_operator = true;
        preconditioner.initialize(system_matrix, data);

        solver.solve(system_matrix, completely_distributed_solution, system_rhs, preconditioner);
        pcout << "   Solved in " << solver_control.last_step() << " iterations."
              << std::endl;

        constraints.distribute(completely_distributed_solution);

        local_solution = completely_distributed_solution;
    }

    //------------------------------
    //Refine the Grid using a built in error estimator
    //------------------------------
    template <int dim>
    void Poisson<dim>::refine_grid()
    {
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

        KellyErrorEstimator<dim>::estimate(dof_handler,
                                           QGauss<dim - 1>(fe.degree + 1),
                                           std::map<types::boundary_id, const Function<dim> *>(),
                                           local_solution,
                                           estimated_error_per_cell);

        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.15, 0);

        triangulation.execute_coarsening_and_refinement();
    }

    //------------------------------
    //Output the result using a vtk file format
    //------------------------------
    template <int dim>
    void Poisson<dim>::output_results(const unsigned int cycle) const
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(local_solution, "u");

        Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();

        data_out.add_data_vector(subdomain, "subdomain");
        data_out.build_patches();

        data_out.write_vtu_with_pvtu_record("./", "solution", cycle, mpi_communicator, 2, 8);
    }

    template <int dim>
    void Poisson<dim>::output_error()
    {
        if (this_mpi_process == 0)
        {
            convergence_table.set_precision("L2", 3);
            convergence_table.set_precision("Linfty", 3);
            convergence_table.set_precision("sTime", 3);
            convergence_table.set_precision("rTime", 3);

            convergence_table.set_scientific("L2", true);
            convergence_table.set_scientific("Linfty", true);
            convergence_table.set_scientific("rTime", true);
            convergence_table.set_scientific("sTime", true);

            convergence_table.set_tex_caption("cycle", "cycle");
            convergence_table.set_tex_caption("cells", "$n_{cells}$");
            convergence_table.set_tex_caption("dofs", "$n_{dof}$");
            convergence_table.set_tex_caption("L2", "$\\left\\|u_h - I_hu\\right\\| _{L_2}$");
            convergence_table.set_tex_caption("Linfty", "$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$");
            convergence_table.set_tex_caption("sTime", "$t_{solve}$");
            convergence_table.set_tex_caption("rTime", "$t_{refine}$");

            std::ofstream error_table_file("error.tex");
            convergence_table.write_tex(error_table_file);

            std::ofstream output_customMax("error_max_dealii.txt");

            output_customMax << "Deal.ii" << std::endl;
            output_customMax << "$n_\\text{dof}$" << std::endl;
            output_customMax << "$\\left\\|u_h - I_hu\\right\\|_{L_\\infty}$" << std::endl;
            output_customMax << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customMax << convergence_vector[i].n_dofs << " " << convergence_vector[i].max_error << std::endl;
            }
            output_customMax.close();

            std::ofstream output_customL2("error_l2_dealii.txt");

            output_customL2 << "Deal.ii" << std::endl;
            output_customL2 << "$n_\\text{dof}$" << std::endl;
            output_customL2 << "$\\left\\|u_h - I_hu\\right\\|_{L_2}$" << std::endl;
            output_customL2 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customL2 << convergence_vector[i].n_dofs << " " << convergence_vector[i].l2_error << std::endl;
            }
            output_customL2.close();

            std::ofstream output_custom1("error_dealii_p1.txt");

            output_custom1 << "$x_1$" << std::endl;
            output_custom1 << "$n_\\text{dof}$" << std::endl;
            output_custom1 << "$|u(x_i) - u_h(x_i)|$" << std::endl;
            output_custom1 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_custom1 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p1 << std::endl;
            }
            output_custom1.close();

            std::ofstream output_custom2("error_dealii_p2.txt");

            output_custom2 << "$x_2$" << std::endl;
            output_custom2 << "$n_\\text{dof}$" << std::endl;
            output_custom2 << "$|u(x_i) - u_h(x_i)| $" << std::endl;
            output_custom2 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_custom2 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p2 << std::endl;
            }
            output_custom2.close();

            std::ofstream output_custom3("error_dealii_p3.txt");

            output_custom3 << "$x_3$" << std::endl;
            output_custom3 << "$n_\\text{dof}$" << std::endl;
            output_custom3 << "$|u(x_i) - u_h(x_i)| $" << std::endl;
            output_custom3 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_custom3 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p3 << std::endl;
            }
            output_custom3.close();

            std::ofstream output_customTimeDOF("error_dealii_time_dof.txt");

            output_customTimeDOF << "Deal.ii" << std::endl;
            output_customTimeDOF << "$n_\\text{dof}$" << std::endl;
            output_customTimeDOF << "Time [s]" << std::endl;
            output_customTimeDOF << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeDOF << convergence_vector[i].n_dofs << " " << convergence_vector[i].solution_time << std::endl;
            }
            output_customTimeDOF.close();

            std::ofstream output_customTimeL2("error_dealii_time_l2.txt");

            output_customTimeL2 << "Deal.ii" << std::endl;
            output_customTimeL2 << "$\\left\\|u_h - I_hu\\right\\|_{L_2}$" << std::endl;
            output_customTimeL2 << "Time [s]" << std::endl;
            output_customTimeL2 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeL2 << convergence_vector[i].l2_error << " " << convergence_vector[i].solution_time << std::endl;
            }
            output_customTimeL2.close();

            std::ofstream output_customTimeMax("error_dealii_time_max.txt");

            output_customTimeMax << "Deal.ii" << std::endl;
            output_customTimeMax << "$\\left\\|u_h - I_hu\\right\\|_{L_\\infty}$" << std::endl;
            output_customTimeMax << "Time [s]" << std::endl;
            output_customTimeMax << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeMax << convergence_vector[i].max_error << " " << convergence_vector[i].solution_time << std::endl;
            }
            output_customTimeMax.close();

            std::ofstream output_customTimeRef("error_dealii_refinement_time.txt");

            output_customTimeRef << "Deal.ii" << std::endl;
            output_customTimeRef << "$n_\\text{dof}$" << std::endl;
            output_customTimeRef << "Time [s]" << std::endl;
            output_customTimeRef << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeMax << convergence_vector[i].n_dofs << " " << convergence_vector[i].refinement_time << std::endl;
            }
            output_customTimeMax.close();
        }
    }

    //----------------------------------------------------------------
    //Calculate the exact error usign the solution class.
    //----------------------------------------------------------------
    template <int dim>
    void Poisson<dim>::calculate_exact_error(const unsigned int cycle, double solution_time, double refinement_time)
    {
        Vector<float> difference_per_cell(triangulation.n_active_cells());
        VectorTools::integrate_difference(dof_handler,
                                          local_solution,
                                          Solution<dim>(),
                                          difference_per_cell,
                                          QGauss<dim>(fe.degree + 1),
                                          VectorTools::L2_norm);
        const double L2_error = VectorTools::compute_global_error(triangulation,
                                                                  difference_per_cell,
                                                                  VectorTools::L2_norm);

        const QTrapezoid<1> q_trapez;
        const QIterated<dim> q_iterated(q_trapez, fe.degree * 2 + 1);
        VectorTools::integrate_difference(dof_handler,
                                          local_solution,
                                          Solution<dim>(),
                                          difference_per_cell,
                                          q_iterated,
                                          VectorTools::Linfty_norm);
        const double Linfty_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::Linfty_norm);

        const unsigned int n_active_cells = triangulation.n_active_cells();
        const unsigned int n_dofs = dof_handler.n_dofs();

        double local_error_p1 = abs(postprocessor1(dof_handler, local_solution) - Solution<dim>().value(Point<dim>(0.125, 0.125, 0.125)));
        double local_error_p2 = abs(postprocessor2(dof_handler, local_solution) - Solution<dim>().value(Point<dim>(0.25, 0.25, 0.25)));
        double local_error_p3 = abs(postprocessor3(dof_handler, local_solution) - Solution<dim>().value(Point<dim>(0.5, 0.5, 0.5)));
        double error_p1 = Utilities::MPI::min(local_error_p1, mpi_communicator);
        double error_p2 = Utilities::MPI::min(local_error_p2, mpi_communicator);
        double error_p3 = Utilities::MPI::min(local_error_p3, mpi_communicator);

        pcout << "Cycle " << cycle << ':' << std::endl
              << "   Number of active cells:       " << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: " << n_dofs << std::endl
              << "L2 error: " << L2_error << std::endl
              << "Max error: " << Linfty_error << std::endl;

        convergence_table.add_value("cycle", cycle);
        convergence_table.add_value("cells", n_active_cells);
        convergence_table.add_value("dofs", n_dofs);
        convergence_table.add_value("L2", L2_error);
        convergence_table.add_value("Linfty", Linfty_error);
        convergence_table.add_value("sTime", solution_time);
        convergence_table.add_value("rTime", refinement_time);

        metrics values = {};
        values.max_error = Linfty_error;
        values.error_p1 = error_p1;
        values.error_p2 = error_p2;
        values.error_p3 = error_p3;
        values.n_dofs = n_dofs;
        values.cycle = cycle;
        values.cells = n_active_cells;
        values.solution_time = solution_time;
        values.refinement_time = refinement_time;

        convergence_vector.push_back(values);
    }

    //------------------------------
    //Run the problem.
    //------------------------------
    template <int dim>
    void Poisson<dim>::run()
    {
        pcout << "Running on " << n_mpi_processes << " MPI rank(s)..." << std::endl;

        int cycle = 0;
        double solution_time = 0.0;
        double refinement_time = 0.0;
        make_grid();
        while (true)
        {
            setup_system();
            assemble_system();
#ifdef USE_TIMING
            if (this_mpi_process == 0)
            {
                startTimer();
            }
#endif

            solve();

#ifdef USE_TIMING
            if (this_mpi_process == 0)
            {
                solution_time = printTimer();
            }
#endif

            calculate_exact_error(cycle, solution_time, refinement_time);
            pcout << "Cycle " << cycle << std::endl;
            pcout << "DOFs: " << get_n_dof() << std::endl;

            //Condition to reach desired number of degrees of freedom
            if (get_n_dof() > max_dofs)
            {
                break;
            }
#ifdef USE_TIMING
            if (this_mpi_process == 0)
            {
                startTimer();
            }
#endif
            refine_grid();
#ifdef USE_TIMING
            if (this_mpi_process == 0)
            {
                refinement_time = printTimer();
            }
#endif
            cycle++;
        }
#ifdef USE_OUTPUT
        output_results(cycle);
#endif
        output_error();
    }

}