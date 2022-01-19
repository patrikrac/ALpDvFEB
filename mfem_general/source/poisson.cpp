//Created by Patrik Rác
//Source for the Poisson class
#include "poisson.hpp"

#ifdef USE_TIMING
#include "Timer.hpp"
#endif

namespace AspDEQuFEL
{
#ifdef USE_TIMING
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
#endif

    using namespace mfem;
    using namespace std;
    //----------------------------------------------------------------
<<<<<<< HEAD
    //Create the mesh the problem will be solved on
=======
    //Create the parallel mesh the problem will be solved on
>>>>>>> parallel
    //----------------------------------------------------------------
    void Poisson::make_mesh()
    {
        const char *mesh_file = "../data/unit_cube.mesh";
<<<<<<< HEAD
        mesh = Mesh::LoadFromFile(mesh_file);
=======
        Mesh mesh(mesh_file);

        //Turns the Quad mesh (which supports hanging nodes) to a tetrahedral mesh without hanging nodes
        if (!nc_simplices)
        {
            mesh = Mesh::MakeSimplicial(mesh);
        }

>>>>>>> parallel
        for (int i = 0; i < 3; i++)
        {
            mesh.UniformRefinement();
        }

<<<<<<< HEAD
        //Turns the Quad mesh (which supports hanging nodes) to a tetrahedral mesh without hanging nodes
        mesh = Mesh::MakeSimplicial(mesh);

        mesh.Finalize(true);

        cout << "Mesh generated." << endl;
=======
        mesh.Finalize(true);

        if (reorder_mesh)
        {
            Array<int> ordering;
            switch (reorder_mesh)
            {
            case 1:
                mesh.GetHilbertElementOrdering(ordering);
                break;
            case 2:
                mesh.GetGeckoElementOrdering(ordering);
                break;
            default:
                MFEM_ABORT("Unknown mesh reodering type " << reorder_mesh);
            }
            mesh.ReorderElements(ordering);
        }

        mesh.EnsureNCMesh(nc_simplices);

        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
>>>>>>> parallel
    }

    //----------------------------------------------------------------
    //Update all variables to adabt to the recently adapted mesh
    //----------------------------------------------------------------
<<<<<<< HEAD
    void Poisson::update(BilinearForm &a, LinearForm &f, FiniteElementSpace &fespace, GridFunction &x, GridFunction &error_zero)
=======
    void Poisson::update(ParBilinearForm &a, ParLinearForm &f, ParFiniteElementSpace &fespace, ParGridFunction &x)
>>>>>>> parallel
    {
        fespace.Update();

        x.Update();

<<<<<<< HEAD
        error_zero.Update();
        error_zero = 0.0;
=======
        if (pmesh->Nonconforming())
        {
            pmesh->Rebalance();

            fespace.Update();
            x.Update();
        }
>>>>>>> parallel

        fespace.UpdatesFinished();

        // Inform the linear and bilinear forms that the space has changed.
        a.Update();
        f.Update();
    }

<<<<<<< HEAD
    //----------------------------------------------------------------
    //Solve the Problem on the current mesh
    //----------------------------------------------------------------
    void Poisson::solve(BilinearForm &a, LinearForm &f, FiniteElementSpace &fespace, GridFunction &x, Array<int> &ess_bdr, FunctionCoefficient &bdr)
    {
        a.Assemble();
        f.Assemble();

        // Project the exact solution to the essential boundary DOFs.
        x.ProjectBdrCoefficient(bdr, ess_bdr);

        //Create and solve the linear system.
        Array<int> ess_tdof_list;
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        SparseMatrix A;
        Vector B, X;

        a.FormLinearSystem(ess_tdof_list, x, f, A, X, B);

        GSSmoother M(A);
        PCG(A, M, B, X, 0, 500, 1e-12, 0.0);
=======
    void Poisson::assemble(ParBilinearForm &a, ParLinearForm &f, ParFiniteElementSpace &fespace, ParGridFunction &x, Array<int> &ess_bdr, FunctionCoefficient &bdr)
    {
        f.Assemble();
        a.Assemble();

        x.ProjectBdrCoefficient(bdr, ess_bdr);
        Array<int> ess_tdof_list;
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        a.FormLinearSystem(ess_tdof_list, x, f, A, X, B);
    }

    //----------------------------------------------------------------
    //Solve the Problem on the current mesh
    //----------------------------------------------------------------
    void Poisson::solve(ParBilinearForm &a, ParLinearForm &f, ParGridFunction &x)
    {
        HypreBoomerAMG *amg = new HypreBoomerAMG(A);
        amg->SetPrintLevel(0);

        HyprePCG pcg(A);
        pcg.SetTol(1e-12);
        pcg.SetMaxIter(2000);
        pcg.SetPrintLevel(3);
        pcg.SetPreconditioner(*amg);
        pcg.Mult(B, X);
>>>>>>> parallel

        a.RecoverFEMSolution(X, f, x);
    }

    //----------------------------------------------------------------
    //Execute one refinement step and call the update funciton to adapt the other variables
    //----------------------------------------------------------------
<<<<<<< HEAD
    bool Poisson::refine(BilinearForm &a, LinearForm &f, FiniteElementSpace &fespace, GridFunction &x, GridFunction &error_zero, ThresholdRefiner &refiner)
    {
        refiner.Apply(mesh);
=======
    bool Poisson::refine(ThresholdRefiner &refiner)
    {
        refiner.Apply(*pmesh);
>>>>>>> parallel

        if (refiner.Stop())
        {
            return false;
        }

<<<<<<< HEAD
        //Update the space and interpolate the solution.
        update(a, f, fespace, x, error_zero);

=======
>>>>>>> parallel
        return true;
    }

    //----------------------------------------------------------------
    //This method initializes a lot of the functionality of the programm
    //Run the problem and terminate after a given condition
    //----------------------------------------------------------------
    void Poisson::run()
    {
        //Mesh Generation
        make_mesh();

        //Declaration
<<<<<<< HEAD
        int dim = mesh.Dimension();
        int sdim = mesh.SpaceDimension();

        //Setup the necessary spaces and solutions for the problem
        H1_FECollection fec(order, dim);
        FiniteElementSpace fespace(&mesh, &fec);

        BilinearForm a(&fespace);
        LinearForm f(&fespace);

        GridFunction x(&fespace);

        //Grid function for calculation of the relative norm
        GridFunction error_zero(&fespace);
        error_zero = 0.0;

        FunctionCoefficient u(bdr_func);

        Array<int> ess_bdr(mesh.bdr_attributes.Max());
=======
        int dim = pmesh->Dimension();
        int sdim = pmesh->SpaceDimension();

        MFEM_VERIFY(pmesh->bdr_attributes.Size() > 0,
                    "Boundary attributes required in the mesh.");

        //Setup the necessary spaces and solutions for the problem
        H1_FECollection fec(order, dim);
        ParFiniteElementSpace fespace(pmesh, &fec);

        ParBilinearForm a(&fespace);
        ParLinearForm f(&fespace);

        ParGridFunction x(&fespace);

        FunctionCoefficient u(bdr_func);

        Array<int> ess_bdr(pmesh->bdr_attributes.Max());
>>>>>>> parallel
        ess_bdr = 1;

        //Setup
        ConstantCoefficient one(1.0);
        FunctionCoefficient rhs(rhs_func);
        FunctionCoefficient bdr(bdr_func);

        //Specify the Problem
        BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
        a.AddDomainIntegrator(integ);
        f.AddDomainIntegrator(new DomainLFIntegrator(rhs));

<<<<<<< HEAD
        KellyErrorEstimator *estimator{nullptr};
        L2_FECollection flux_fec(order, dim);

        auto flux_fes = new FiniteElementSpace(&mesh, &flux_fec, sdim);
=======
        L2_FECollection flux_fec(order, dim);
        KellyErrorEstimator *estimator{nullptr};

        auto flux_fes = new ParFiniteElementSpace(pmesh, &flux_fec, sdim);
>>>>>>> parallel
        estimator = new KellyErrorEstimator(*integ, x, flux_fes);

        ThresholdRefiner refiner(*estimator);
        refiner.SetTotalErrorFraction(0.3);
<<<<<<< HEAD
        refiner.SetLocalErrorGoal(max_elem_error);
        refiner.SetNCLimit(0);
=======
>>>>>>> parallel

        x = 0.0;

        refiner.Reset();

<<<<<<< HEAD
        int step = 0;
        while (true)
        {

            cout << "Step: " << step << endl
                 << "DOF: " << fespace.GetNDofs() << endl;
=======
        double solve_timing = 0.0;
        double refine_timing = 0.0;
        double assembly_time = 0.0;
        int iter = 0;
        while (true)
        {
            HYPRE_BigInt global_dofs = fespace.GlobalTrueVSize();
>>>>>>> parallel

#ifdef USE_TIMING
            startTimer();
#endif

<<<<<<< HEAD
            solve(a, f, fespace, x, ess_bdr, bdr);

#ifdef USE_TIMING
            printTimer();
#endif

#ifdef USE_OUTPUT
            exact_error(step, fespace.GetNDofs(), x, error_zero, u);
            vtk_output(x, step);
#endif

            //Stop the loop if no more elements are marked for refinement or the desired number of DOFs is reached.
            if (fespace.GetNDofs() > max_dofs || !refine(a, f, fespace, x, error_zero, refiner))
            {
                break;
            }

            step++;
        }

        cout << "Final: " << step << endl
             << "DOF: " << fespace.GetNDofs() << endl;

        a.Update();
        f.Update();

        delete estimator;

#ifdef USE_OUTPUT
        output_table();
#endif
=======
            assemble(a, f, fespace, x, ess_bdr, bdr);

#ifdef USE_TIMING
            if (myid == 0)
            {
                assembly_time = printTimer();
            }
#endif

            if (myid == 0)
            {
                cout << "Iteration: " << iter << endl
                     << "DOFs: " << global_dofs << endl;

#ifdef USE_TIMING
                startTimer();
#endif
            }

            solve(a, f, x);

#ifdef USE_TIMING
            if (myid == 0)
            {
                solve_timing = printTimer();
            }
#endif

            exact_error(iter, global_dofs, solve_timing, refine_timing, assembly_time, x, u);

#ifdef USE_TIMING
            if (myid == 0)
            {
                startTimer();
            }
#endif
            //Stop the loop if no more elements are marked for refinement or the desired number of DOFs is reached.
            if (global_dofs >= max_dofs || !refine(refiner))
            {
                if (myid == 0)
                {
                    cout << "Stopping criterion satisfied. Stop." << endl;
                }
                break;
            }
#ifdef USE_TIMING
            if (myid == 0)
            {
                refine_timing = printTimer();
            }
#endif

            update(a, f, fespace, x);
            iter++;
        }
#ifdef USE_TIMING
        if (myid == 0)
        {
            printTimer();
        }
#endif
#ifdef USE_OUTPUT
        vtk_output(x);
#endif
        delete pmesh;

        output_table();
>>>>>>> parallel
    }

    //----------------------------------------------------------------
    //Calculate the exact error
    //----------------------------------------------------------------
<<<<<<< HEAD
    void Poisson::exact_error(int cycle, int dofs, GridFunction &x, GridFunction &error_zero, FunctionCoefficient &u)
=======
    void Poisson::exact_error(int cycle, int dofs, double solution_time, double refinement_time, double assembly_time, ParGridFunction &x, FunctionCoefficient &u)
>>>>>>> parallel
    {

        error_values values = {};
        values.cycle = cycle;
        values.dofs = dofs;
<<<<<<< HEAD
        values.cells = mesh.GetNE();
        values.max_error = x.ComputeMaxError(u);
        values.l2_error = x.ComputeL2Error(u);
        values.relative_error = values.l2_error / error_zero.ComputeL2Error(u);
=======
        values.solution_time = solution_time;
        values.refinement_time = refinement_time;
        values.assembly_time = assembly_time;
        int local_cells = pmesh->GetNE();
        MPI_Reduce(&local_cells, &values.cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        values.max_error = x.ComputeMaxError(u);
        values.l2_error = x.ComputeL2Error(u);
>>>>>>> parallel

        double p1[] = {0.125, 0.125, 0.125};
        double p2[] = {0.25, 0.25, 0.25};
        double p3[] = {0.5, 0.5, 0.5};

<<<<<<< HEAD
        values.error_p1 = abs(postprocessor1(x, mesh) - bdr_func(Vector(p1, 3)));
        values.error_p2 = abs(postprocessor2(x, mesh) - bdr_func(Vector(p2, 3)));
        values.error_p3 = abs(postprocessor3(x, mesh) - bdr_func(Vector(p3, 3)));
        table_vector.push_back(values);

        cout << "Max error for step " << cycle << ": " << setprecision(3) << scientific << values.max_error << endl;
        cout << "L2 error: " << setprecision(3) << scientific << values.l2_error << endl;
=======
        double local_error_p1 = abs(postprocessor1(x, *pmesh) - bdr_func(Vector(p1, 3)));
        MPI_Reduce(&local_error_p1, &values.error_p1, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        double local_error_p2 = abs(postprocessor2(x, *pmesh) - bdr_func(Vector(p2, 3)));
        MPI_Reduce(&local_error_p2, &values.error_p2, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        double local_error_p3 = abs(postprocessor3(x, *pmesh) - bdr_func(Vector(p3, 3)));
        MPI_Reduce(&local_error_p3, &values.error_p3, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        table_vector.push_back(values);
        if (myid == 0)
        {
            cout << "Max error for step " << cycle << ": " << setprecision(3) << scientific << values.max_error << endl;
            cout << "L2 error: " << setprecision(3) << scientific << values.l2_error << endl;
        }
>>>>>>> parallel
    }

    //----------------------------------------------------------------
    //Output the table containing the calcualted errors
    //----------------------------------------------------------------
    void Poisson::output_table()
    {
<<<<<<< HEAD
        std::ofstream output("table.tex");
        std::ofstream output_max("error_max_mfem.txt");
        std::ofstream output_l2("error_l2_mfem.txt");
        std::ofstream output_p1("error_p1.txt");
        std::ofstream output_p2("error_p2.txt");
        std::ofstream output_p3("error_p3.txt");

        output_l2 << "MFEM" << endl;
        output_l2 << "$n_\\text{dof}$" << endl;
        output_l2 << "$\\left\\|u - u_h\\right\\| _{L^2}$" << endl;
        output_l2 << table_vector.size() << endl;

        output_max << "MFEM" << endl;
        output_max << "$n_\\text{dof}$" << endl;
        output_max << "$\\left\\|u - u_h\\right\\| _{L^\\infty}$" << endl;
        output_max << table_vector.size() << endl;

        output_p1 << "MFEM" << endl;
        output_p1 << "$n_\\text{dof}$" << endl;
        output_p1 << "$\\left\\|u(x_1) - u_h(x_1)\\right\\| $" << endl;
        output_p1 << table_vector.size() << endl;

        output_p2 << "MFEM" << endl;
        output_p2 << "$n_\\text{dof}$" << endl;
        output_p2 << "$\\left\\|u(x_2) - u_h(x_2)\\right\\| $" << endl;
        output_p2 << table_vector.size() << endl;

        output_p3 << "MFEM" << endl;
        output_p3 << "$n_\\text{dof}$" << endl;
        output_p3 << "$\\left\\|u(x_3) - u_h(x_3)\\right\\| $" << endl;
        output_p3 << table_vector.size() << endl;

        output << "\\begin{table}[h]" << endl;
        output << "\t\\begin{center}" << endl;
        output << "\t\t\\begin{tabular}{|c|c|c|c|c|} \\hline" << endl;

        output << "\t\t\tcycle & \\# cells & \\# dofs & $\\norm{u - u_h}_{L^\\infty}$ & $\\norm{u - u_h}_{L^\\2}$ \\\\  \\hline" << endl;
        for (int i = 0; i < table_vector.size(); i++)
        {
            output << "\t\t\t" << table_vector[i].cycle << " & " << table_vector[i].cells << " & " << table_vector[i].dofs << " & " << setprecision(3) << scientific << table_vector[i].max_error << " & " << setprecision(3) << scientific << table_vector[i].l2_error << "\\\\ \\hline" << endl;
            output_max << table_vector[i].dofs << " " << table_vector[i].max_error << endl;
            output_l2 << table_vector[i].dofs << " " << table_vector[i].l2_error << endl;
            output_p1 << table_vector[i].dofs << " " << table_vector[i].error_p1 << endl;
            output_p2 << table_vector[i].dofs << " " << table_vector[i].error_p2 << endl;
            output_p3 << table_vector[i].dofs << " " << table_vector[i].error_p3 << endl;
        }

        output << "\t\t\\end{tabular}" << endl;
        output << "\t\\end{center}" << endl;
        output << "\\end{table}" << endl;
=======
        if (myid == 0)
        {
            std::ofstream output("table.tex");
            std::ofstream output_max("error_max_mfem.txt");
            std::ofstream output_l2("error_l2_mfem.txt");

            std::ofstream output_time_dof("time_dof_mfem.txt");
            std::ofstream output_time_l2("time_l2_mfem.txt");
            std::ofstream output_time_max("time_max_mfem.txt");

            std::ofstream output_refinement_time_dof("time_refinement_dof_mfem.txt");
            std::ofstream output_refinement_time_l2("time_refinement_l2_mfem.txt");
            std::ofstream output_refinement_time_max("time_refinement_max_mfem.txt");

            std::ofstream output_assembly_time_dof("time_assembly_dof_mfem.txt");
            std::ofstream output_assembly_time_l2("time_assembly_l2_mfem.txt");
            std::ofstream output_assembly_time_max("time_assembly_max_mfem.txt");

            std::ofstream output_p1("error_p1_mfem.txt");
            std::ofstream output_p2("error_p2_mfem.txt");
            std::ofstream output_p3("error_p3_mfem.txt");

            output_max << "MFEM" << endl;
            output_max << "$n_\\text{dof}$" << endl;
            output_max << "$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$" << endl;
            output_max << table_vector.size() << endl;

            output_l2 << "MFEM" << endl;
            output_l2 << "$n_\\text{dof}$" << endl;
            output_l2 << "$\\left\\|u_h - I_hu\\right\\| _{L_2}$" << endl;
            output_l2 << table_vector.size() << endl;

            output_time_dof << "MFEM" << endl;
            output_time_dof << "$n_\\text{dof}$" << endl;
            output_time_dof << "$Time [s]$" << endl;
            output_time_dof << table_vector.size() << endl;

            output_refinement_time_dof << "MFEM" << endl;
            output_refinement_time_dof << "$n_\\text{dof}$" << endl;
            output_refinement_time_dof << "$Time [s]$" << endl;
            output_refinement_time_dof << table_vector.size() << endl;

            output_assembly_time_dof << "MFEM" << endl;
            output_assembly_time_dof << "$n_\\text{dof}$" << endl;
            output_assembly_time_dof << "$Time [s]$" << endl;
            output_assembly_time_dof << table_vector.size() << endl;

            output_time_l2 << "MFEM" << endl;
            output_time_l2 << "$Time [s]$" << endl;
            output_time_l2 << "$\\left\\|u_h - I_hu\\right\\| _{L_2}$" << endl;
            output_time_l2 << table_vector.size() << endl;

            output_time_max << "MFEM" << endl;
            output_time_max << "$Time [s]$" << endl;
            output_time_max << "$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$" << endl;
            output_time_max << table_vector.size() << endl;

            output_refinement_time_l2 << "MFEM" << endl;
            output_refinement_time_l2 << "$Time [s]$" << endl;
            output_refinement_time_l2 << "$\\left\\|u_h - I_hu\\right\\| _{L_2}$" << endl;
            output_refinement_time_l2 << table_vector.size() << endl;

            output_refinement_time_max << "MFEM" << endl;
            output_refinement_time_max << "$Time [s]$" << endl;
            output_refinement_time_max << "$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$" << endl;
            output_refinement_time_max << table_vector.size() << endl;

            output_assembly_time_l2 << "MFEM" << endl;
            output_assembly_time_l2 << "$Time [s]$" << endl;
            output_assembly_time_l2 << "$\\left\\|u_h - I_hu\\right\\| _{L_2}$" << endl;
            output_assembly_time_l2 << table_vector.size() << endl;

            output_assembly_time_max << "MFEM" << endl;
            output_assembly_time_max << "$Time [s]$" << endl;
            output_assembly_time_max << "$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$" << endl;
            output_assembly_time_max << table_vector.size() << endl;

            output_p1 << "$x_1$" << endl;
            output_p1 << "$n_\\text{dof}$" << endl;
            output_p1 << "$|u(x_i) - u_h(x_i)|$" << endl;
            output_p1 << table_vector.size() << endl;

            output_p2 << "$x_2$" << endl;
            output_p2 << "$n_\\text{dof}$" << endl;
            output_p2 << "$|u(x_i) - u_h(x_i)|$" << endl;
            output_p2 << table_vector.size() << endl;

            output_p3 << "$x_3$" << endl;
            output_p3 << "$n_\\text{dof}$" << endl;
            output_p3 << "$|u(x_i) - u_h(x_i)|$" << endl;
            output_p3 << table_vector.size() << endl;

            output << "\\begin{table}[h]" << endl;
            output << "\t\\begin{center}" << endl;
            output << "\t\t\\begin{tabular}{|c|c|c|c|c|c|c|c|} \\hline" << endl;

            output << "\t\t\tcycle & $n_{cells} $ & $n_{dof}$ & $\\left\\|u_h - I_hu\\right\\| _{L_2}$ & $\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$ & $t_{solve}$ & $t_{refine}$ & $t_{assembly}$\\\\ \\hline" << endl;
            for (int i = 0; i < table_vector.size(); i++)
            {
                output << "\t\t\t" << table_vector[i].cycle << " & " << table_vector[i].cells << " & " << table_vector[i].dofs << " & " << setprecision(3) << scientific << table_vector[i].l2_error << " & " << setprecision(3) << scientific << table_vector[i].max_error << " & "
                       << setprecision(3) << scientific << table_vector[i].solution_time << " & "
                       << setprecision(3) << scientific << table_vector[i].refinement_time << " & "
                       << setprecision(3) << scientific << table_vector[i].assembly_time << "\\\\ \\hline" << endl;
                output_max << table_vector[i].dofs << " " << table_vector[i].max_error << endl;
                output_l2 << table_vector[i].dofs << " " << table_vector[i].l2_error << endl;

                output_time_dof << table_vector[i].dofs << " " << table_vector[i].solution_time << endl;
                output_refinement_time_dof << table_vector[i].dofs << " " << table_vector[i].refinement_time << endl;
                output_assembly_time_dof << table_vector[i].dofs << " " << table_vector[i].assembly_time << endl;

                output_time_l2 << table_vector[i].solution_time << " " << table_vector[i].l2_error << endl;
                output_time_max << table_vector[i].solution_time << " " << table_vector[i].max_error << endl;

                output_refinement_time_l2 << table_vector[i].refinement_time << " " << table_vector[i].l2_error << endl;
                output_refinement_time_max << table_vector[i].refinement_time << " " << table_vector[i].max_error << endl;

                output_assembly_time_l2 << table_vector[i].solution_time << " " << table_vector[i].l2_error << endl;
                output_assembly_time_max << table_vector[i].assembly_time << " " << table_vector[i].max_error << endl;

                output_p1 << table_vector[i].dofs << " " << table_vector[i].error_p1 << endl;
                output_p2 << table_vector[i].dofs << " " << table_vector[i].error_p2 << endl;
                output_p3 << table_vector[i].dofs << " " << table_vector[i].error_p3 << endl;
            }

            output << "\t\t\\end{tabular}" << endl;
            output << "\t\\end{center}" << endl;
            output << "\\end{table}" << endl;
        }
>>>>>>> parallel
    }

    //----------------------------------------------------------------
    //Create a vtk Output for the current solution
    //----------------------------------------------------------------
<<<<<<< HEAD
    void Poisson::vtk_output(GridFunction &x, int &cycle)
    {
        std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");

        mesh.PrintVTK(output, 0);
        x.SaveVTK(output, "u", 0);
        output.close();
    }

=======
    void Poisson::vtk_output(ParGridFunction &x)
    {
        ofstream output(MakeParFilename("solution", myid, ".vtk"));
        pmesh->PrintVTK(output, 0);
        x.SaveVTK(output, "u", 0);
        output.close();

        if (myid == 0)
        {
            ofstream output("solution.visit");
            output << "!NBLOCKS " << num_procs << endl;
            for (int proc = 0; proc < num_procs; proc++)
            {
                output << MakeParFilename("solution", proc, ".vtk") << endl;
            }
            output.close();
        }
    }
>>>>>>> parallel
}
