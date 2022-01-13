//Header file for the MFEM multigrid solver
#include "mfem.hpp"
#pragma once

using namespace std;
using namespace mfem;
class PoissonMultigrid : public GeometricMultigrid
{
private:
   ConstantCoefficient one;
   HypreBoomerAMG* amg;

public:
   // Constructs a diffusion multigrid for the ParFiniteElementSpaceHierarchy
   // and the array of essential boundaries
   PoissonMultigrid(ParFiniteElementSpaceHierarchy& fespaces,
                      Array<int>& ess_bdr)
      : GeometricMultigrid(fespaces), one(1.0)
   {
      ConstructCoarseOperatorAndSolver(fespaces.GetFESpaceAtLevel(0), ess_bdr);

      for (int level = 1; level < fespaces.GetNumLevels(); ++level)
      {
         ConstructOperatorAndSmoother(fespaces.GetFESpaceAtLevel(level), ess_bdr);
      }
   }

   virtual ~PoissonMultigrid()
   {
      delete amg;
   }

private:
   void ConstructBilinearForm(ParFiniteElementSpace& fespace, Array<int>& ess_bdr)
   {
      ParBilinearForm* form = new ParBilinearForm(&fespace);
      form->AddDomainIntegrator(new DiffusionIntegrator(one));
      form->Assemble();
      bfs.Append(form);

      essentialTrueDofs.Append(new Array<int>());
      fespace.GetEssentialTrueDofs(ess_bdr, *essentialTrueDofs.Last());
   }

   void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace& coarse_fespace,
                                         Array<int>& ess_bdr)
   {
      ConstructBilinearForm(coarse_fespace, ess_bdr);

      HypreParMatrix* hypreCoarseMat = new HypreParMatrix();
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), *hypreCoarseMat);

      amg = new HypreBoomerAMG(*hypreCoarseMat);
      amg->SetPrintLevel(-1);

      CGSolver* pcg = new CGSolver(MPI_COMM_WORLD);
      pcg->SetPrintLevel(-1);
      pcg->SetMaxIter(10);
      pcg->SetRelTol(sqrt(1e-4));
      pcg->SetAbsTol(0.0);
      pcg->SetOperator(*hypreCoarseMat);
      pcg->SetPreconditioner(*amg);

      AddLevel(hypreCoarseMat, pcg, true, true);
   }

   void ConstructOperatorAndSmoother(ParFiniteElementSpace& fespace,
                                     Array<int>& ess_bdr)
   {
      ConstructBilinearForm(fespace, ess_bdr);

      OperatorPtr opr;
      opr.SetType(Operator::ANY_TYPE);
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
      opr.SetOperatorOwner(false);

      Vector diag(fespace.GetTrueVSize());
      bfs.Last()->AssembleDiagonal(diag);

      Solver* smoother = new OperatorChebyshevSmoother(*opr, diag,
                                                       *essentialTrueDofs.Last(), 2, fespace.GetParMesh()->GetComm());

      AddLevel(opr.Ptr(), smoother, true, true);
   }
};