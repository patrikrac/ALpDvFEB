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

#include "evaluation.hpp"
#include "problem.hpp"

#ifdef USE_TIMING
#include "Timer.hpp"
#endif
#pragma once

namespace AspDEQuFEL
{

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
        void output_vtk(const unsigned int cycle);
        void output_results();

#ifdef USE_TIMING
        void startTimer();
        double printTimer();

        timing::Timer timer;
#endif

        int max_dofs;

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
    Poisson<dim>::Poisson(int degree, int max_dof) : max_dofs(max_dof), triangulation(Triangulation<dim>::limit_level_difference_at_vertices), fe(degree),
                                                     dof_handler(triangulation),
                                                     degree(degree),
                                                     postprocessor1(Point<dim>(0.125, 0.125, 0.125)), postprocessor2(Point<dim>(0.25, 0.25, 0.25)), postprocessor3(Point<dim>(0.5, 0.5, 0.5))
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
    void Poisson<dim>::setup_system()
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
    void Poisson<dim>::cell_worker(const Iterator &cell, ScratchData<dim> &scratch_data, CopyData &copy_data)
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
    void Poisson<dim>::assemble_multigrid()
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
    void Poisson<dim>::assemble_system()
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
    void Poisson<dim>::refine_grid()
    {
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

        KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(fe.degree + 1), {}, solution, estimated_error_per_cell);

        GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);

        triangulation.execute_coarsening_and_refinement();
    }

    //------------------------------
    //Print the mesh and the solution in a vtk file
    //------------------------------
    template <int dim>
    void Poisson<dim>::output_vtk(const unsigned int cycle)
    {
        std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
        DataOut<dim> data_out;

        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "u");
        data_out.build_patches();

        data_out.write_vtk(output);
    }

    //------------------------------
    //Output the result using a custiom file format
    //------------------------------
    template <int dim>
    void Poisson<dim>::output_results()
    {
        convergence_table.set_precision("Linfty", 3);
        convergence_table.set_precision("L2", 3);
        convergence_table.set_precision("error_p1", 3);
        convergence_table.set_precision("error_p2", 3);
        convergence_table.set_precision("error_p3", 3);
        convergence_table.set_scientific("Linfty", true);
        convergence_table.set_scientific("L2", true);
        convergence_table.set_scientific("error_p1", true);
        convergence_table.set_scientific("error_p2", true);
        convergence_table.set_scientific("error_p3", true);

        convergence_table.set_tex_caption("cells", "\\# cells");
        convergence_table.set_tex_caption("dofs", "\\# dofs");
        convergence_table.set_tex_caption("Linfty", "$\\left\\|u_h - I_hu\\right\\| _{L^\\infty}$");
        convergence_table.set_tex_caption("L2", "$\\left\\|u_h - I_hu\\right\\| _{L^2}}$");
        convergence_table.set_tex_caption("error_p1", "$\\left\\|u_h(x_1) - I_hu(x_1)\\right\\| $");
        convergence_table.set_tex_caption("error_p2", "$\\left\\|u_h(x_2) - I_hu(x_2)\\right\\| $");
        convergence_table.set_tex_caption("error_p3", "$\\left\\|u_h(x_3) - I_hu(x_3)\\right\\| $");

        std::ofstream error_table_file("error.tex");
        convergence_table.write_tex(error_table_file);

        std::ofstream output_customMax("error_max_dealii.txt");
        output_customMax << "$deal.ii$" << std::endl;
        output_customMax << "$n_\\text{dof}$" << std::endl;
        output_customMax << "$\\left\\|u_h - I_hu\\right\\| $" << std::endl;
        output_customMax << convergence_vector.size() << std::endl;
        for (size_t i = 0; i < convergence_vector.size(); i++)
        {
            output_customMax << convergence_vector[i].n_dofs << " " << convergence_vector[i].max_error << std::endl;
        }
        output_customMax.close();

        std::ofstream output_customL2("error_l2_dealii.txt");
        output_customL2 << "$deal.ii$" << std::endl;
        output_customL2 << "$n_\\text{dof}$" << std::endl;
        output_customL2 << "$\\left\\|u_h - I_hu\\right\\| $" << std::endl;
        output_customL2 << convergence_vector.size() << std::endl;
        for (size_t i = 0; i < convergence_vector.size(); i++)
        {
            output_customL2 << convergence_vector[i].n_dofs << " " << convergence_vector[i].max_error << std::endl;
        }
        output_customL2.close();

        std::ofstream output_custom1("error_dealii_p1.txt");

        output_custom1 << "$\\left\\|u_h(x_1) - I_hu(x_1)\\right\\| $" << std::endl;
        output_custom1 << "$n_\\text{dof}$" << std::endl;
        output_custom1 << "$\\left\\|u_h(x) - I_hu(x)\\right\\|$" << std::endl;
        output_custom1 << convergence_vector.size() << std::endl;
        for (size_t i = 0; i < convergence_vector.size(); i++)
        {
            output_custom1 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p1 << std::endl;
        }
        output_custom1.close();

        std::ofstream output_custom2("error_dealii_p2.txt");

        output_custom2 << "$\\left\\|u_h(x_2) - I_hu(x_2)\\right\\|$" << std::endl;
        output_custom2 << "$n_\\text{dof}$" << std::endl;
        output_custom2 << "$\\left\\|u_h(x) - I_hu(x)\\right\\| $" << std::endl;
        output_custom2 << convergence_vector.size() << std::endl;
        for (size_t i = 0; i < convergence_vector.size(); i++)
        {
            output_custom2 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p2 << std::endl;
        }
        output_custom2.close();

        std::ofstream output_custom3("error_dealii_p3.txt");

        output_custom3 << "$\\left\\|u_h(x_3) - I_hu(x_3)\\right\\|$" << std::endl;
        output_custom3 << "$n_\\text{dof}$" << std::endl;
        output_custom3 << "$\\left\\|u_h(x) - I_hu(x)\\right\\| $" << std::endl;
        output_custom3 << convergence_vector.size() << std::endl;
        for (size_t i = 0; i < convergence_vector.size(); i++)
        {
            output_custom3 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p3 << std::endl;
        }
        output_custom3.close();
    }

    //----------------------------------------------------------------
    //Calculate the exact error usign the solution class.
    //----------------------------------------------------------------
    template <int dim>
    void Poisson<dim>::calculate_exact_error(const unsigned int cycle)
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

        VectorTools::integrate_difference(dof_handler,
                                          solution,
                                          Solution<dim>(),
                                          difference_per_cell,
                                          QGauss<dim>(fe.degree + 1),
                                          VectorTools::L2_norm);
        const double L2_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);

        const double relative_Linfty_error = Linfty_error / Linfty_norm;

        const unsigned int n_active_cells = triangulation.n_active_cells();
        const unsigned int n_dofs = dof_handler.n_dofs();

        double error_p1 = abs(postprocessor1(dof_handler, solution) - Solution<dim>().value(Point<dim>(0.125, 0.125, 0.125)));
        double error_p2 = abs(postprocessor2(dof_handler, solution) - Solution<dim>().value(Point<dim>(0.25, 0.25, 0.25)));
        double error_p3 = abs(postprocessor3(dof_handler, solution) - Solution<dim>().value(Point<dim>(0.5, 0.5, 0.5)));

        std::cout << "Cycle " << cycle << ':' << std::endl
                  << "   Number of active cells:       " << n_active_cells
                  << std::endl
                  << "   Number of degrees of freedom: " << n_dofs << std::endl
                  << "Max error: " << Linfty_error << std::endl
                  << "L2 error: " << L2_error << std::endl;

        convergence_table.add_value("cycle", cycle);
        convergence_table.add_value("cells", n_active_cells);
        convergence_table.add_value("dofs", n_dofs);
        convergence_table.add_value("Linfty", Linfty_error);
        convergence_table.add_value("L2", L2_error);
        convergence_table.add_value("error_p1", error_p1);
        convergence_table.add_value("error_p2", error_p2);
        convergence_table.add_value("error_p3", error_p3);

        metrics values = {};
        values.max_error = Linfty_error;
        values.l2_error = L2_error;
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
    void Poisson<dim>::run()
    {

        int cycle = 0;
        make_grid();
        while (true)
        {
#ifdef USE_TIMING
            startTimer();
#endif

            setup_system();
            assemble_system();
            solve();

#ifdef USE_TIMING
            printTimer();
#endif

#ifdef USE_OUTPUT
            calculate_exact_error(cycle);
            output_vtk(cycle);
#endif

            //Netgen similar condition to reach desired number of degrees of freedom
            if (get_n_dof() > max_dofs)
            {
                break;
            }

            refine_grid();

            cycle++;
        }

#ifdef USE_OUTPUT
        output_results();
#endif
    }
}