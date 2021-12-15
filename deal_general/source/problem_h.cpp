#include "../include/problem_h.hpp"

using namespace dealii;
//------------------------------
//Initialize the problem with first order finite elements
//The dof_handler manages enumeration and indexing of all degrees of freedom (relating to the given triangulation)
//------------------------------
template <int dim>
Problem<dim>::Problem() : mpi_communicator(MPI_COMM_WORLD),
                          n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
                          this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
                          pcout(std::cout, (this_mpi_process == 0)),
                          fe(2),
                          dof_handler(triangulation),
                          postprocessor1(Point<dim>(0.125, 0.125, 0.125)),
                          postprocessor2(Point<dim>(0.25, 0.25, 0.25)),
                          postprocessor3(Point<dim>(0.5, 0.5, 0.5))
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
    GridTools::partition_triangulation(n_mpi_processes, triangulation);

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::subdomain_wise(dof_handler);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    //VectorTools::interpolate_boundary_values(dof_handler, 0, BoundaryValues<dim>(), constraints);

    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    const std::vector<IndexSet> locally_owned_dofs_per_proc = DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
    const IndexSet locally_owned_dofs = locally_owned_dofs_per_proc[this_mpi_process];

    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

    solution.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}

//------------------------------
//Assemble the system by creating a quadrature rule for integeration, calculate local matrices using the appropriate weak formulations and assamble the global matrices.
//------------------------------
template <int dim>
void Problem<dim>::assemble_system()
{
    QGauss<dim> quadrature_formula(fe.degree + 1);
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
        if (cell->subdomain_id() == this_mpi_process)
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

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, false);
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
    SolverControl solver_control(1000, 1e-12);
    PETScWrappers::SolverCG cg(solver_control, mpi_communicator);

    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    Vector<double> localized_solution(solution);

    constraints.distribute(localized_solution);

    solution = localized_solution;
}

//------------------------------
//Refine the Grid using a built in error estimator
//------------------------------
template <int dim>
void Problem<dim>::refine_grid()
{
    const Vector<double> localized_solution(solution);

    Vector<float> local_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(fe.degree + 1), {},
                                       localized_solution,
                                       local_error_per_cell,
                                       ComponentMask(),
                                       nullptr,
                                       MultithreadInfo::n_threads(),
                                       this_mpi_process);

    const unsigned int n_local_cells = GridTools::count_cells_with_subdomain_association(triangulation, this_mpi_process);

    PETScWrappers::MPI::Vector distributed_all_errors(mpi_communicator, triangulation.n_active_cells(), n_local_cells);
    for (unsigned int i = 0; i < local_error_per_cell.size(); i++)
    {
        if (local_error_per_cell(i) != 0)
            distributed_all_errors(i) = local_error_per_cell(i);
    }
    distributed_all_errors.compress(VectorOperation::insert);
    const Vector<float> localized_all_errors(distributed_all_errors);
    GridRefinement::refine_and_coarsen_fixed_number(triangulation, localized_all_errors, 0.3, 0.03);

    triangulation.execute_coarsening_and_refinement();
}

//------------------------------
//Output the result using a vtk file format
//------------------------------
template <int dim>
void Problem<dim>::output_results(const unsigned int cycle)
{
    const Vector<double> localized_solution(solution);
    if (this_mpi_process == 0)
    {
        std::ofstream output("solution/solution-" + std::to_string(cycle) + ".vtk");

        DataOut<dim> data_out;

        data_out.attach_dof_handler(dof_handler);

        data_out.add_data_vector(localized_solution, "u");

        data_out.build_patches();
        data_out.write_vtk(output);
    }
}

template <int dim>
void Problem<dim>::output_error()
{
    if (this_mpi_process == 0)
    {
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

        std::ofstream output_custom1("error_dealii.txt");

        output_custom1 << "$deal.ii$" << std::endl;
        output_custom1 << "$n_\\text{dof}$" << std::endl;
        output_custom1 << "$\\left\\|u_h - I_hu\\right\\| $" << std::endl;
        output_custom1 << convergence_vector.size() << std::endl;
        for (size_t i = 0; i < convergence_vector.size(); i++)
        {
            output_custom1 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error << std::endl;
        }
        output_custom1.close();

        std::ofstream output_custom2("error_dealii_p1.txt");

        output_custom2 << "$\\left\\|u_h(x_1) - I_hu(x_1)\\right\\| $" << std::endl;
        output_custom2 << "$n_\\text{dof}$" << std::endl;
        output_custom2 << "$\\left\\|u_h(x) - I_hu(x)\\right\\|$" << std::endl;
        output_custom2 << convergence_vector.size() << std::endl;
        for (size_t i = 0; i < convergence_vector.size(); i++)
        {
            output_custom2 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p1 << std::endl;
        }
        output_custom2.close();

        std::ofstream output_custom3("error_dealii_p2.txt");

        output_custom3 << "$\\left\\|u_h(x_2) - I_hu(x_2)\\right\\|$" << std::endl;
        output_custom3 << "$n_\\text{dof}$" << std::endl;
        output_custom3 << "$\\left\\|u_h(x) - I_hu(x)\\right\\| $" << std::endl;
        output_custom3 << convergence_vector.size() << std::endl;
        for (size_t i = 0; i < convergence_vector.size(); i++)
        {
            output_custom3 << convergence_vector[i].n_dofs << " " << convergence_vector[i].error_p2 << std::endl;
        }
        output_custom3.close();

        std::ofstream output_custom4("error_dealii_p3.txt");

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

    pcout << "Cycle " << cycle << ':' << std::endl
              << "   Number of active cells:       " << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: " << n_dofs << std::endl
              << "Max error: " << Linfty_error << std::endl;

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
        solve();

        //calculate_exact_error(cycle);
        pcout << "Cycle " << cycle << std::endl;
        output_results(cycle);
        //Netgen similar condition to reach desired number of degrees of freedom
        if (get_n_dof() > 100000)
        {
            break;
        }

        cycle++;
    }

    
}