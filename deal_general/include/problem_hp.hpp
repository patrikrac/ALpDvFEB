#include "data.hpp"
#include "evaluation.hpp"
#include "Timer.hpp"
#pragma once

namespace problem_hp
{

    /**********
 * Wrapper around the timer functions that are given.
 * */
    timing::Timer timer;
    // Starts or resets the current clock.
    void startTimer()
    {
        timer.reset();
    }
    // prints the current value of the clock
    double printTimer()
    {
        double time = timer.elapsed();
        std::cout << "Calculation took " << time << " seconds." << std::endl;
        return time;
    }
    /**
 * End of time wrapper functions
 *********/

    using namespace dealii;
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
        GridGenerator::hyper_rectangle(triangulation, Point<3>(0.0, 0.0, 0.0), Point<3>(1.0, 1.0, 1.0));

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
}