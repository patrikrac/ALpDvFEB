/*  ----------------------------------------------------------------
Created by Patrik Rac
Programm solving a partial differential equation of arbitrary dimensionality using the functionality of the deal.ii FE-library.
This class explicitly uses hp-adaptivity.
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
#include <deal.II/numerics/smoothness_estimator.h>

#include <deal.II/base/utilities.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/base/index_set.h>

#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/refinement.h>
#include <deal.II/fe/fe_series.h>

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
        void calculate_exact_error(const unsigned int cycle, double solution_time, double refinement_time, double assembly_time);
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

        DoFHandler<dim> dof_handler;
        hp::FECollection<dim> fe_collection;
        hp::QCollection<dim> quadrature_collection;
        hp::QCollection<dim - 1> face_quadrature_collection;

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;
        AffineConstraints<double> constraints;

        LinearAlgebraPETSc::MPI::SparseMatrix system_matrix;

        LinearAlgebraPETSc::MPI::Vector local_solution;
        LinearAlgebraPETSc::MPI::Vector system_rhs;

        const unsigned int max_degree;

        ConvergenceTable convergence_table;
        PointValueEvaluation<dim> postprocessor1;
        PointValueEvaluation<dim> postprocessor2;
        PointValueEvaluation<dim> postprocessor3;
        std::vector<metrics> convergence_vector;
    };

    //------------------------------
    //The dof_handler manages enumeration and indexing of all degrees of freedom (relating to the given triangulation).
    //Set an adequate maximum degree.
    //------------------------------
    template <int dim>
    PoissonHP<dim>::PoissonHP(int max_dof) : max_dofs(max_dof),
                                             mpi_communicator(MPI_COMM_WORLD),
                                             triangulation(mpi_communicator, typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::smoothing_on_refinement | Triangulation<dim>::smoothing_on_coarsening)),
                                             n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
                                             this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
                                             pcout(std::cout, (this_mpi_process == 0)),
                                             dof_handler(triangulation),
                                             max_degree(dim <= 2 ? 7 : 5),
                                             postprocessor1(Point<3>(0.5, 0.125, 0.875)),
                                             postprocessor2(Point<3>(0.5, 0.25, 0.875)),
                                             postprocessor3(Point<3>(0.5, 0.5, 0.875))
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
    PoissonHP<dim>::~PoissonHP()
    {
        dof_handler.clear();
    }

#ifdef USE_TIMING
    //------------------------------
    //Start the timer
    //------------------------------
    template <int dim>
    void PoissonHP<dim>::startTimer()
    {
        timer.reset();
    }

    //------------------------------
    // Prints the current value of the clock
    //------------------------------
    template <int dim>
    double PoissonHP<dim>::printTimer()
    {
        double time = timer.elapsed();
        std::cout << "Calculation took " << time << " seconds." << std::endl;
        return time;
    }
#endif

    //------------------------------
    //Construct the Grid the ProblemHP is beeing solved on.
    //Define the default coarsaty / refinement of the grid
    //------------------------------
    template <int dim>
    void PoissonHP<dim>::make_grid()
    {
        //The default grid generated will be a unit square/cube depending on the dimensionality of the problem.
        GridGenerator::hyper_cube(triangulation, 0, 1);

        triangulation.refine_global(3);

        pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
    }

    //------------------------------
    //Setup the system by initializing the solution and ProblemHP vectors with the right dimensionality and apply bounding constraints.
    //Althogh system calls are equal to the ones of the non hp-version of the program, it has to be noted that the dof_handler is in hp-mode and therefore the calls differ internally.
    //------------------------------
    template <int dim>
    void PoissonHP<dim>::setup_system()
    {
        dof_handler.distribute_dofs(fe_collection);
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
    void PoissonHP<dim>::assemble_system()
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
            if (cell->is_locally_owned())
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

        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
    }

    //------------------------------
    //Return the number of degrees of freedom of the current ProblemHP state.
    //------------------------------
    template <int dim>
    int PoissonHP<dim>::get_n_dof()
    {
        return dof_handler.n_dofs();
    }

    //------------------------------
    //Set solving conditinos and define the solver. Then solve the given system.
    //------------------------------
    template <int dim>
    void PoissonHP<dim>::solve()
    {
        LinearAlgebraPETSc::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

        SolverControl solver_control(2000, 1e-12);
        LinearAlgebraPETSc::SolverCG solver(solver_control, mpi_communicator);

        LinearAlgebraPETSc::MPI::PreconditionAMG preconditioner;
        LinearAlgebraPETSc::MPI::PreconditionAMG::AdditionalData data;
        //data.symmetric_operator = true;
        preconditioner.initialize(system_matrix, data);

        solver.solve(system_matrix, completely_distributed_solution, system_rhs, preconditioner);
        pcout << "   Solved in " << solver_control.last_step() << " iterations."
              << std::endl;

        constraints.distribute(completely_distributed_solution);

        local_solution = completely_distributed_solution;
    }

    //------------------------------
    //Refine the Grid using a built in error estimator.
    //------------------------------
    template <int dim>
    void PoissonHP<dim>::refine_grid()
    {
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

        KellyErrorEstimator<dim>::estimate(dof_handler,
                                           face_quadrature_collection,
                                           std::map<types::boundary_id, const Function<dim> *>(),
                                           local_solution,
                                           estimated_error_per_cell);

        Vector<float> smoothness_indicators(triangulation.n_active_cells());
        FESeries::Fourier<dim> fourier = SmoothnessEstimator::Fourier::default_fe_series(fe_collection);
        SmoothnessEstimator::Fourier::coefficient_decay(fourier,
                                                        dof_handler,
                                                        local_solution,
                                                        smoothness_indicators);

        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.15, 0);

        hp::Refinement::p_adaptivity_from_relative_threshold(dof_handler, smoothness_indicators, 0.15, 0);

        hp::Refinement::choose_p_over_h(dof_handler);

        triangulation.prepare_coarsening_and_refinement();
        hp::Refinement::limit_p_level_difference(dof_handler);

        triangulation.execute_coarsening_and_refinement();
    }

    //------------------------------
    //Output the result using a vtk file format
    //------------------------------
    template <int dim>
    void PoissonHP<dim>::output_results(const unsigned int cycle) const
    {
        DataOut<dim> data_out;

        Vector<float> fe_degrees(triangulation.n_active_cells());
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
                fe_degrees(cell->active_cell_index()) = fe_collection[cell->active_fe_index()].degree;
        }

        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(local_solution, "u");
        data_out.add_data_vector(fe_degrees, "fe_degree");

        Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();

        data_out.add_data_vector(subdomain, "subdomain");
        data_out.build_patches();

        data_out.write_vtu_with_pvtu_record("./", "solution", cycle, mpi_communicator, 2, 8);
    }

    template <int dim>
    void PoissonHP<dim>::output_error()
    {
        if (this_mpi_process == 0)
        {
            convergence_table.set_precision("L2", 3);
            convergence_table.set_precision("Linfty", 3);
            convergence_table.set_precision("sTime", 3);
            convergence_table.set_precision("rTime", 3);
            convergence_table.set_precision("assemTime", 3);

            convergence_table.set_scientific("L2", true);
            convergence_table.set_scientific("Linfty", true);
            convergence_table.set_scientific("rTime", true);
            convergence_table.set_scientific("sTime", true);
            convergence_table.set_scientific("assemTime", true);

            convergence_table.set_tex_caption("cycle", "cycle");
            convergence_table.set_tex_caption("cells", "$n_{cells}$");
            convergence_table.set_tex_caption("dofs", "$n_{dof}$");
            convergence_table.set_tex_caption("L2", "$\\left\\|u_h - I_hu\\right\\| _{L_2}$");
            convergence_table.set_tex_caption("Linfty", "$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$");
            convergence_table.set_tex_caption("sTime", "$t_{solve}$");
            convergence_table.set_tex_caption("rTime", "$t_{refine}$");
            convergence_table.set_tex_caption("assemTime", "$t_{assembly}$");

            std::ofstream error_table_file("error_hp.tex");
            convergence_table.write_tex(error_table_file);

            std::ofstream output_customMax("error_max_dealii_hp.txt");

            output_customMax << "Deal.ii" << std::endl;
            output_customMax << "$n_\\text{dof}$" << std::endl;
            output_customMax << "$\\left\\|u_h - I_hu\\right\\|_{L_\\infty}$" << std::endl;
            output_customMax << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customMax << convergence_vector[i].n_dofs << " " << convergence_vector[i].max_error << std::endl;
            }
            output_customMax.close();

            std::ofstream output_customL2("error_l2_dealii_hp.txt");

            output_customL2 << "Deal.ii" << std::endl;
            output_customL2 << "$n_\\text{dof}$" << std::endl;
            output_customL2 << "$\\left\\|u_h - I_hu\\right\\|_{L_2}$" << std::endl;
            output_customL2 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customL2 << convergence_vector[i].n_dofs << " " << convergence_vector[i].l2_error << std::endl;
            }
            output_customL2.close();

            std::ofstream output_custom1("error_dealii_p1_hp.txt");

            output_custom1 << "$x_1$" << std::endl;
            output_custom1 << "$n_\\text{dof}$" << std::endl;
            output_custom1 << "$|u(x_i) - u_h(x_i)|$" << std::endl;
            output_custom1 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_custom1 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p1 << std::endl;
            }
            output_custom1.close();

            std::ofstream output_custom2("error_dealii_p2_hp.txt");

            output_custom2 << "$x_2$" << std::endl;
            output_custom2 << "$n_\\text{dof}$" << std::endl;
            output_custom2 << "$|u(x_i) - u_h(x_i)| $" << std::endl;
            output_custom2 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_custom2 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p2 << std::endl;
            }
            output_custom2.close();

            std::ofstream output_custom3("error_dealii_p3_hp.txt");

            output_custom3 << "$x_3$" << std::endl;
            output_custom3 << "$n_\\text{dof}$" << std::endl;
            output_custom3 << "$|u(x_i) - u_h(x_i)| $" << std::endl;
            output_custom3 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_custom3 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p3 << std::endl;
            }
            output_custom3.close();

            std::ofstream output_customTimeDOF("time_dof_dealii_hp.txt");

            output_customTimeDOF << "Deal.ii" << std::endl;
            output_customTimeDOF << "$n_\\text{dof}$" << std::endl;
            output_customTimeDOF << "$Time [s]$" << std::endl;
            output_customTimeDOF << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeDOF << convergence_vector[i].n_dofs << " " << convergence_vector[i].solution_time << std::endl;
            }
            output_customTimeDOF.close();

            std::ofstream output_customTimeL2("time_l2_dealii_hp.txt");

            output_customTimeL2 << "Deal.ii" << std::endl;
            output_customTimeL2 << "$Time [s]$" << std::endl;
            output_customTimeL2 << "$\\left\\|u_h - I_hu\\right\\|_{L_2}$" << std::endl;
            output_customTimeL2 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeL2 << convergence_vector[i].solution_time << " " << convergence_vector[i].l2_error << std::endl;
            }
            output_customTimeL2.close();

            std::ofstream output_customTimeMax("time_max_dealii_hp.txt");

            output_customTimeMax << "Deal.ii" << std::endl;
            output_customTimeMax << "$Time [s]$" << std::endl;
            output_customTimeMax << "$\\left\\|u_h - I_hu\\right\\|_{L_\\infty}$" << std::endl;
            output_customTimeMax << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeMax << convergence_vector[i].solution_time << " " << convergence_vector[i].max_error << std::endl;
            }
            output_customTimeMax.close();

            std::ofstream output_customTimeRef("refinement_time_dof_dealii_hp.txt");

            output_customTimeRef << "Deal.ii" << std::endl;
            output_customTimeRef << "$n_\\text{dof}$" << std::endl;
            output_customTimeRef << "$Time [s]$" << std::endl;
            output_customTimeRef << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeRef << convergence_vector[i].n_dofs << " " << convergence_vector[i].refinement_time << std::endl;
            }
            output_customTimeRef.close();

            std::ofstream output_customTimeRefL2("refinement_time_l2_dealii_hp.txt");

            output_customTimeRefL2 << "Deal.ii" << std::endl;
            output_customTimeRefL2 << "$Time [s]$" << std::endl;
            output_customTimeRefL2 << "$\\left\\|u_h - I_hu\\right\\|_{L_2}$" << std::endl;
            output_customTimeRefL2 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeRefL2 << convergence_vector[i].refinement_time << " " << convergence_vector[i].l2_error << std::endl;
            }
            output_customTimeRefL2.close();

            std::ofstream output_customTimeRefMax("refinement_time_max_dealii_hp.txt");

            output_customTimeRefMax << "Deal.ii" << std::endl;
            output_customTimeRefMax << "$Time [s]$" << std::endl;
            output_customTimeRefMax << "$\\left\\|u_h - I_hu\\right\\|_{L_\\infty}$" << std::endl;
            output_customTimeRefMax << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeRefMax << convergence_vector[i].refinement_time << " " << convergence_vector[i].max_error << std::endl;
            }
            output_customTimeRefMax.close();

            std::ofstream output_customTimeAssem("assembly_time_dof_dealii_hp.txt");

            output_customTimeAssem << "Deal.ii" << std::endl;
            output_customTimeAssem << "$n_\\text{dof}$" << std::endl;
            output_customTimeAssem << "$Time [s]$" << std::endl;
            output_customTimeAssem << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeAssem << convergence_vector[i].n_dofs << " " << convergence_vector[i].assembly_time << std::endl;
            }
            output_customTimeAssem.close();

            std::ofstream output_customTimeAssemL2("assembly_time_l2_dealii_hp.txt");

            output_customTimeAssemL2 << "Deal.ii" << std::endl;
            output_customTimeAssemL2 << "$Time [s]$" << std::endl;
            output_customTimeAssemL2 << "$\\left\\|u_h - I_hu\\right\\|_{L_2}$" << std::endl;
            output_customTimeAssemL2 << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeAssemL2 << convergence_vector[i].assembly_time << " " << convergence_vector[i].l2_error << std::endl;
            }
            output_customTimeAssemL2.close();

            std::ofstream output_customTimeAssemMax("assembly_time_max_dealii_hp.txt");

            output_customTimeAssemMax << "Deal.ii" << std::endl;
            output_customTimeAssemMax << "$Time [s]$" << std::endl;
            output_customTimeAssemMax << "$\\left\\|u_h - I_hu\\right\\|_{L_\\infty}$" << std::endl;
            output_customTimeAssemMax << convergence_vector.size() << std::endl;
            for (size_t i = 0; i < convergence_vector.size(); i++)
            {
                output_customTimeAssemMax << convergence_vector[i].assembly_time << " " << convergence_vector[i].max_error << std::endl;
            }
            output_customTimeAssemMax.close();
        }
    }

    //----------------------------------------------------------------
    //Calculate the exact error usign the solution class.
    //----------------------------------------------------------------
    template <int dim>
    void PoissonHP<dim>::calculate_exact_error(const unsigned int cycle, double solution_time, double refinement_time, double assembly_time)
    {
        Vector<float> difference_per_cell(triangulation.n_active_cells());
        VectorTools::integrate_difference(dof_handler,
                                          local_solution,
                                          Solution<dim>(),
                                          difference_per_cell,
                                          quadrature_collection,
                                          VectorTools::L2_norm);
        const double L2_error = VectorTools::compute_global_error(triangulation,
                                                                  difference_per_cell,
                                                                  VectorTools::L2_norm);

        VectorTools::integrate_difference(dof_handler,
                                          local_solution,
                                          Solution<dim>(),
                                          difference_per_cell,
                                          quadrature_collection,
                                          VectorTools::Linfty_norm);
        const double Linfty_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::Linfty_norm);

        const unsigned int n_active_cells = triangulation.n_global_active_cells();
        const unsigned int n_dofs = dof_handler.n_dofs();

        double local_error_p1 = abs(postprocessor1(dof_handler, local_solution) - Solution<dim>().value(Point<dim>(0.5, 0.125, 0.875)));
        double local_error_p2 = abs(postprocessor2(dof_handler, local_solution) - Solution<dim>().value(Point<dim>(0.5, 0.25, 0.875)));
        double local_error_p3 = abs(postprocessor3(dof_handler, local_solution) - Solution<dim>().value(Point<dim>(0.5, 0.5, 0.875)));
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
        convergence_table.add_value("assemTime", assembly_time);

        metrics values = {};
        values.max_error = Linfty_error;
        values.l2_error = L2_error;
        values.error_p1 = error_p1;
        values.error_p2 = error_p2;
        values.error_p3 = error_p3;
        values.n_dofs = n_dofs;
        values.cycle = cycle;
        values.cells = n_active_cells;
        values.solution_time = solution_time;
        values.refinement_time = refinement_time;
        values.assembly_time = assembly_time;

        convergence_vector.push_back(values);
    }

    //------------------------------
    //Execute the solving process with cylce refinement steps.
    //------------------------------
    template <int dim>
    void PoissonHP<dim>::run()
    {
        pcout << "Running on " << n_mpi_processes << " MPI rank(s)..." << std::endl;
        int cycle = 0;
        double solution_time = 0.0;
        double refinement_time = 0.0;
        double assembly_time = 0.0;
        make_grid();

        while (true)
        {
#ifdef USE_TIMING
            if (this_mpi_process == 0)
            {
                startTimer();
            }
#endif

            setup_system();
            assemble_system();

#ifdef USE_TIMING
            if (this_mpi_process == 0)
            {
                assembly_time = printTimer();
            }
#endif

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

            calculate_exact_error(cycle, solution_time, refinement_time, assembly_time);
            pcout << "Cycle " << cycle << std::endl;
            pcout << "DOFs: " << get_n_dof() << std::endl;

            //Stopping Condition when reached desired number of degrees of freedom
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