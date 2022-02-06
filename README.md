# ALpDvFEB
Collection of Programms using different Finite-Element Libraries to solve partial differential equations.
This is the test implementation for using geometric multigrid solvers in the adaptivity loop. The MEFM and Deal.ii implementations feature working (yet unoptimied) serial implementations.
The used libraries include 
- deal.ii
  - For more information and installation instructions visit <https://www.dealii.org>
- Netgen/NGSolve
  - For more information and installation instructions visit <https://ngsolve.org>
- MFEM
  - For more information and installation instructions visit <https://mfem.org>

The programms build the bases to solve PDEs using an adaptive mesh refinement loop.

The programms are written as the main work of my Bachelor Thesis at Friedrich-Alexander Universität Erlangen-Nürnberg.

All programms output teh soulution as vtk-files which can be visualised by any compatiable tool (I personally recommend [Visit](https://visit-dav.github.io/visit-website/index.html)).

