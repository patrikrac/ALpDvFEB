//Created by Patrik Rác
//MFEM Multigrid solver class for the poissong problem

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#pragma once

using namespace std;
using namespace mfem;

class PoissonMultigrid : public GeometricMultigrid
{
public:
    PoissonMultigrid(FiniteElementSpaceHierarchy &fespaces, Array<int> &ess_bdr)
        : GeometricMultigrid(fespaces), one(1.0)
    {
        ConstructCoarseOperatorAndSolver(fespaces.GetFESpaceAtLevel(0), ess_bdr);

        for (int level = 1; level < fespaces.GetNumLevels(); ++level)
        {
            ConstructOperatorAndSmoother(fespaces.GetFESpaceAtLevel(level), ess_bdr);
        }
    }

private:
    ConstantCoefficient one;

    void ConstructBilinearForm(FiniteElementSpace &fespace, Array<int> &ess_bdr)
    {
        BilinearForm *form = new BilinearForm(&fespace);
        

        form->AddDomainIntegrator(new DiffusionIntegrator(one));

        form->Assemble();
        bfs.Append(form);

        essentialTrueDofs.Append(new Array<int>());
        fespace.GetEssentialTrueDofs(ess_bdr, *essentialTrueDofs.Last());
    }

    void ConstructCoarseOperatorAndSolver(FiniteElementSpace &coarse_fespace,
                                          Array<int> &ess_bdr)
    {
        ConstructBilinearForm(coarse_fespace, ess_bdr);

        OperatorPtr opr;
        opr.SetType(Operator::ANY_TYPE);
        bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
        opr.SetOperatorOwner(false);

        CGSolver *pcg = new CGSolver();
        pcg->SetPrintLevel(-1);
        pcg->SetMaxIter(200);
        pcg->SetRelTol(sqrt(1e-4));
        pcg->SetAbsTol(0.0);
        pcg->SetOperator(*opr.Ptr());

        AddLevel(opr.Ptr(), pcg, true, true);
    }

    void ConstructOperatorAndSmoother(FiniteElementSpace &fespace,
                                      Array<int> &ess_bdr)
    {
        ConstructBilinearForm(fespace, ess_bdr);

        OperatorPtr opr;
        opr.SetType(Operator::ANY_TYPE);
        bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
        opr.SetOperatorOwner(false);

        Vector diag(fespace.GetTrueVSize());
        bfs.Last()->AssembleDiagonal(diag);

        Solver *smoother = new OperatorChebyshevSmoother(*opr, diag,
                                                         *essentialTrueDofs.Last(), 2);
        AddLevel(opr.Ptr(), smoother, true, true);
    }
};