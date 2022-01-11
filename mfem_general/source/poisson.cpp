//Created by Patrik Rác
//Source for the Poisson class
#include "poisson.hpp"
#include "multigrid.hpp"

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
    //Create the mesh the problem will be solved on
    //----------------------------------------------------------------
    Mesh *Poisson::make_mesh()
    {
        const char *mesh_file = "../data/unit_cube.mesh";
        Mesh mesh = Mesh::LoadFromFile(mesh_file);
        for (int i = 0; i < 3; i++)
        {
            mesh.UniformRefinement();
        }

        //Turns the Quad mesh (which supports hanging nodes) to a tetrahedral mesh without hanging nodes
        mesh = Mesh::MakeSimplicial(mesh);

        mesh.Finalize(true);

        cout << "Mesh generated." << endl;

        return new Mesh(mesh);
    }

    //----------------------------------------------------------------
    //Solve the Problem on the current mesh
    //----------------------------------------------------------------
    void Poisson::solve(LinearForm &f, FiniteElementSpaceHierarchy &fespaces, GridFunction &x, Array<int> &ess_bdr, FunctionCoefficient &bdr)
    {
        f.Assemble();

        // Project the exact solution to the essential boundary DOFs.
        x.ProjectBdrCoefficient(bdr, ess_bdr);

        //Create and solve the linear system.
        PoissonMultigrid *M = new PoissonMultigrid(fespaces, ess_bdr);
        M->SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);

        OperatorPtr A;
        Vector B, X;

        M->FormFineLinearSystem(x, f, A, X, B);

        PCG(*A, *M, B, X, 0, 2000, 1e-12, 0.0);

        M->RecoverFEMSolution(X, f, x);
    }

    //----------------------------------------------------------------
    //Execute one refinement step and call the update funciton to adapt the other variables
    //----------------------------------------------------------------
    bool Poisson::refine(LinearForm &f, FiniteElementSpaceHierarchy &fespaces, GridFunction &x, GridFunction &error_zero)
    {
        ConstantCoefficient one(1.0);
        BilinearFormIntegrator *integ = new DiffusionIntegrator(one);

        KellyErrorEstimator *estimator{nullptr};
        L2_FECollection flux_fec(order, fespaces.GetFinestFESpace().GetMesh()->Dimension());

        auto flux_fes = new FiniteElementSpace(fespaces.GetFinestFESpace().GetMesh(), &flux_fec, fespaces.GetFinestFESpace().GetMesh()->SpaceDimension());
        estimator = new KellyErrorEstimator(*integ, x, flux_fes);

        ThresholdRefiner refiner(*estimator);
        refiner.SetTotalErrorFraction(0.3);
        refiner.SetLocalErrorGoal(max_elem_error);
        refiner.SetNCLimit(0);

        refiner.Reset();

        Mesh *mesh = new Mesh(*fespaces.GetFinestFESpace().GetMesh());
        refiner.Apply(*mesh);
        if (refiner.Stop())
        {
            return false;
        }
        

        FiniteElementSpace &coarseFEspace = fespaces.GetFinestFESpace();
        FiniteElementSpace *fineFEspace = new FiniteElementSpace(mesh, coarseFEspace.FEColl());

        Operator *P = new TransferOperator(coarseFEspace, *fineFEspace);
        fespaces.AddLevel(mesh, fineFEspace, P, true, true, true);

        x.SetSpace(&fespaces.GetFinestFESpace());
        error_zero.SetSpace(&fespaces.GetFinestFESpace());
        x = 0.0;
        error_zero = 0.0;

        f.Update(&fespaces.GetFinestFESpace());

        return true;
    }

    //----------------------------------------------------------------
    //This method initializes a lot of the functionality of the programm
    //Run the problem and terminate after a given condition
    //----------------------------------------------------------------
    void Poisson::run()
    {
        //Mesh Generation
        Mesh *mesh = make_mesh();

        //Declaration
        int dim = mesh->Dimension();
        int sdim = mesh->SpaceDimension();

        //Setup the necessary spaces and solutions for the problem
        H1_FECollection *fec = new H1_FECollection(order, dim);
        FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
        FiniteElementSpaceHierarchy fespaces(mesh, fespace, true, true);

        LinearForm f(&fespaces.GetFinestFESpace());

        GridFunction x(&fespaces.GetFinestFESpace());
        x = 0.0;

        //Grid function for calculation of the relative norm
        GridFunction error_zero(&fespaces.GetFinestFESpace());
        error_zero = 0.0;

        FunctionCoefficient u(bdr_func);

        Array<int> ess_bdr(fespaces.GetFinestFESpace().GetMesh()->bdr_attributes.Max());
        ess_bdr = 1;

        //Setup
        FunctionCoefficient rhs(rhs_func);
        FunctionCoefficient bdr(bdr_func);

        //Specify the Problem
        f.AddDomainIntegrator(new DomainLFIntegrator(rhs));

        int step = 0;
        while (true)
        {

            cout << "Step: " << step << endl
                 << "DOF: " << fespaces.GetFinestFESpace().GetNDofs() << endl;

#ifdef USE_TIMING
            startTimer();
#endif

            solve(f, fespaces, x, ess_bdr, bdr);

#ifdef USE_TIMING
            printTimer();
#endif

#ifdef USE_OUTPUT
            exact_error(step, fespaces.GetFinestFESpace().GetNDofs(), fespaces, x, error_zero, u);
            vtk_output(fespaces, x, step);
#endif

            //Stop the loop if no more elements are marked for refinement or the desired number of DOFs is reached.
            if (fespaces.GetFinestFESpace().GetNDofs() > max_dofs || !refine(f, fespaces, x, error_zero))
            {
                break;
            }

            step++;
        }

        cout << "Final: " << step << endl
             << "DOF: " << fespaces.GetFinestFESpace().GetNDofs() << endl;

        f.Update();

#ifdef USE_OUTPUT
        output_table();
#endif
    }

    //----------------------------------------------------------------
    //Calculate the exact error
    //----------------------------------------------------------------
    void Poisson::exact_error(int cycle, int dofs, FiniteElementSpaceHierarchy &fespaces, GridFunction &x, GridFunction &error_zero, FunctionCoefficient &u)
    {

        error_values values = {};
        values.cycle = cycle;
        values.dofs = dofs;
        values.cells = 0;
        values.max_error = x.ComputeMaxError(u);
        values.l2_error = x.ComputeL2Error(u);
        values.relative_error = values.l2_error / error_zero.ComputeL2Error(u);

        double p1[] = {0.125, 0.125, 0.125};
        double p2[] = {0.25, 0.25, 0.25};
        double p3[] = {0.5, 0.5, 0.5};

        values.error_p1 = abs(postprocessor1(x, *fespaces.GetFinestFESpace().GetMesh()) - bdr_func(Vector(p1, 3)));
        values.error_p2 = abs(postprocessor2(x, *fespaces.GetFinestFESpace().GetMesh()) - bdr_func(Vector(p2, 3)));
        values.error_p3 = abs(postprocessor3(x, *fespaces.GetFinestFESpace().GetMesh()) - bdr_func(Vector(p3, 3)));
        table_vector.push_back(values);

        cout << "Max error for step " << cycle << ": " << setprecision(3) << scientific << values.max_error << endl;
        cout << "L2 error: " << setprecision(3) << scientific << values.l2_error << endl;
    }

    //----------------------------------------------------------------
    //Output the table containing the calcualted errors
    //----------------------------------------------------------------
    void Poisson::output_table()
    {
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
    }

    //----------------------------------------------------------------
    //Create a vtk Output for the current solution
    //----------------------------------------------------------------
    void Poisson::vtk_output(FiniteElementSpaceHierarchy &fespaces, GridFunction &x, int &cycle)
    {
        std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");

        fespaces.GetFinestFESpace().GetMesh()->PrintVTK(output, 0);
        x.SaveVTK(output, "u", 0);
        output.close();
    }

}
