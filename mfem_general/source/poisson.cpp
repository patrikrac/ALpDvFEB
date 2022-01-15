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
    //Create the parallel mesh the problem will be solved on
    //----------------------------------------------------------------
    void Poisson::make_mesh()
    {
        const char *mesh_file = "../data/unit_cube.mesh";
        Mesh mesh(mesh_file);
        for (int i = 0; i < 3; i++)
        {
            mesh.UniformRefinement();
        }

        //Turns the Quad mesh (which supports hanging nodes) to a tetrahedral mesh without hanging nodes
        if (!nc_simplices)
        {
            mesh = Mesh::MakeSimplicial(mesh);
        }

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
    }

    //----------------------------------------------------------------
    //Update all variables to adabt to the recently adapted mesh
    //----------------------------------------------------------------
    void Poisson::update(ParBilinearForm &a, ParLinearForm &f, ParFiniteElementSpace &fespace, ParGridFunction &x, ParGridFunction &error_zero)
    {
        fespace.Update();

        x.Update();

        error_zero.Update();
        error_zero = 0.0;

        if (pmesh->Nonconforming())
        {
            pmesh->Rebalance();

            fespace.Update();
            x.Update();
        }

        fespace.UpdatesFinished();

        // Inform the linear and bilinear forms that the space has changed.
        a.Update();
        f.Update();
    }

    void Poisson::assemble(ParBilinearForm &a, ParLinearForm &f)
    {
        f.Assemble();
        a.Assemble();
    }

    //----------------------------------------------------------------
    //Solve the Problem on the current mesh
    //----------------------------------------------------------------
    void Poisson::solve(ParBilinearForm &a, ParLinearForm &f, ParFiniteElementSpace &fespace, ParGridFunction &x, Array<int> &ess_bdr, FunctionCoefficient &bdr)
    {
        x.ProjectBdrCoefficient(bdr, ess_bdr);
        Array<int> ess_tdof_list;
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        HypreParMatrix A;
        Vector B, X;

        a.FormLinearSystem(ess_tdof_list, x, f, A, X, B);

        HypreBoomerAMG *amg = new HypreBoomerAMG(A);
        amg->SetPrintLevel(0);

        HyprePCG pcg(A);
        pcg.SetTol(1e-12);
        pcg.SetMaxIter(2000);
        pcg.SetPrintLevel(3);
        pcg.SetPreconditioner(*amg);
        pcg.Mult(B, X);

        a.RecoverFEMSolution(X, f, x);
    }

    //----------------------------------------------------------------
    //Execute one refinement step and call the update funciton to adapt the other variables
    //----------------------------------------------------------------
    bool Poisson::refine(ParBilinearForm &a, ParLinearForm &f, ParFiniteElementSpace &fespace, ParGridFunction &x, ParGridFunction &error_zero, ThresholdRefiner &refiner)
    {
        refiner.Apply(*pmesh);

        if (refiner.Stop())
        {
            return false;
        }

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

        //Grid function for calculation of the relative norm
        ParGridFunction error_zero(&fespace);
        error_zero = 0.0;

        FunctionCoefficient u(bdr_func);

        Array<int> ess_bdr(pmesh->bdr_attributes.Max());
        ess_bdr = 1;

        //Setup
        ConstantCoefficient one(1.0);
        FunctionCoefficient rhs(rhs_func);
        FunctionCoefficient bdr(bdr_func);

        //Specify the Problem
        BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
        a.AddDomainIntegrator(integ);
        f.AddDomainIntegrator(new DomainLFIntegrator(rhs));

        L2_FECollection flux_fec(order, dim);
        ParFiniteElementSpace flux_fes(pmesh, &flux_fec, sdim);
        FiniteElementCollection *smooth_flux_fec = NULL;
        ParFiniteElementSpace *smooth_flux_fes = NULL;
        smooth_flux_fec = new RT_FECollection(order - 1, dim);
        smooth_flux_fes = new ParFiniteElementSpace(pmesh, smooth_flux_fec, 1);

        L2ZienkiewiczZhuEstimator estimator(*integ, x, flux_fes, *smooth_flux_fes);

        ThresholdRefiner refiner(estimator);
        refiner.SetTotalErrorFraction(0.3);

        x = 0.0;

        refiner.Reset();

        double timing = 0.0;
        int iter = 0;
        while (true)
        {
            HYPRE_BigInt global_dofs = fespace.GlobalTrueVSize();

            if (myid == 0)
            {
                cout << "Iteration: " << iter << endl
                     << "DOFs: " << global_dofs << endl;

                update(a, f, fespace, x, error_zero);
                assemble(a,f);
                
#ifdef USE_TIMING
                startTimer();
#endif
            }

            solve(a, f, fespace, x, ess_bdr, bdr);

#ifdef USE_TIMING
            if (myid == 0)
            {
                timing = printTimer();
            }
#endif

            exact_error(iter, timing, global_dofs, x, error_zero, u);

            //Stop the loop if no more elements are marked for refinement or the desired number of DOFs is reached.
            if (global_dofs >= max_dofs || !refine(a, f, fespace, x, error_zero, refiner))
            {
                if (myid == 0)
                {
                    cout << "Stopping criterion satisfied. Stop." << endl;
                }
                break;
            }
            iter++;
        }

#ifdef USE_OUTPUT
        vtk_output(x);
#endif
        delete pmesh;

        //output_table();
    }

    //----------------------------------------------------------------
    //Calculate the exact error
    //----------------------------------------------------------------
    void Poisson::exact_error(int cycle, int dofs, double time, ParGridFunction &x, ParGridFunction &error_zero, FunctionCoefficient &u)
    {

        error_values values = {};
        values.cycle = cycle;
        values.dofs = dofs;
        values.time = time;
        values.cells = pmesh->GetNE();
        values.max_error = x.ComputeMaxError(u);
        values.l2_error = x.ComputeL2Error(u);

        double p1[] = {0.125, 0.125, 0.125};
        double p2[] = {0.25, 0.25, 0.25};
        double p3[] = {0.5, 0.5, 0.5};

        //values.error_p1 = abs(postprocessor1(x, pmesh) - bdr_func(Vector(p1, 3)));
        //values.error_p2 = abs(postprocessor2(x, pmesh) - bdr_func(Vector(p2, 3)));
        //values.error_p3 = abs(postprocessor3(x, pmesh) - bdr_func(Vector(p3, 3)));
        //table_vector.push_back(values);
        if (myid == 0)
        {
            cout << "Max error for step " << cycle << ": " << setprecision(3) << scientific << values.max_error << endl;
            cout << "L2 error: " << setprecision(3) << scientific << values.l2_error << endl;
        }
    }

    //----------------------------------------------------------------
    //Output the table containing the calcualted errors
    //----------------------------------------------------------------
    void Poisson::output_table()
    {
        if (myid == 0)
        {
            std::ofstream output("table.tex");
            std::ofstream output_max("error_max_mfem.txt");
            std::ofstream output_l2("error_l2_mfem.txt");

            std::ofstream output_time_dof("time_dof.txt");
            std::ofstream output_time_l2("time_dof.txt");
            std::ofstream output_time_max("time_dof.txt");
            //std::ofstream output_p1("error_p1.txt");
            //std::ofstream output_p2("error_p2.txt");
            //std::ofstream output_p3("error_p3.txt");

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
            output_time_dof << "Time [s]" << endl;
            output_time_dof << table_vector.size() << endl;

            output_time_l2 << "MFEM" << endl;
            output_time_l2 << "$\\left\\|u_h - I_hu\\right\\| _{L_2}$" << endl;
            output_time_l2 << "Time [s]" << endl;
            output_time_l2 << table_vector.size() << endl;

            output_time_max << "MFEM" << endl;
            output_time_max << "$\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$" << endl;
            output_time_max << "Time [s]" << endl;
            output_time_max << table_vector.size() << endl;
            /*
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
            */
            output << "\\begin{table}[h]" << endl;
            output << "\t\\begin{center}" << endl;
            output << "\t\t\\begin{tabular}{|c|c|c|c|c|} \\hline" << endl;

            output << "\t\t\tcycle & \\# cells & \\# dofs & $\\left\\|u_h - I_hu\\right\\| _{L_2}$ & $\\left\\|u_h - I_hu\\right\\| _{L_\\infty}$\\\\ \\hline" << endl;
            for (int i = 0; i < table_vector.size(); i++)
            {
                output << "\t\t\t" << table_vector[i].cycle << " & " << table_vector[i].cells << " & " << table_vector[i].dofs << " & " << setprecision(3) << scientific << table_vector[i].l2_error << " & " << setprecision(3) << scientific << table_vector[i].max_error << "\\\\ \\hline" << endl;
                output_max << table_vector[i].dofs << " " << table_vector[i].max_error << endl;
                output_l2 << table_vector[i].dofs << " " << table_vector[i].l2_error << endl;
                output_time_dof << table_vector[i].time << " " << table_vector[i].dofs << endl;
                output_time_l2 << table_vector[i].time << " " << table_vector[i].l2_error << endl;
                output_time_max << table_vector[i].time << " " << table_vector[i].max_error << endl;
                //output_p1 << table_vector[i].dofs << " " << table_vector[i].error_p1 << endl;
                //output_p2 << table_vector[i].dofs << " " << table_vector[i].error_p2 << endl;
                //output_p3 << table_vector[i].dofs << " " << table_vector[i].error_p3 << endl;
            }

            output << "\t\t\\end{tabular}" << endl;
            output << "\t\\end{center}" << endl;
            output << "\\end{table}" << endl;
        }
    }

    //----------------------------------------------------------------
    //Create a vtk Output for the current solution
    //----------------------------------------------------------------
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
}
