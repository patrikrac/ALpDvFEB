# ALpDvFEB
Collection of Programms using different Finite-Element Libraries to solve partial differential equations.
The used libraries include 
- deal.ii
  - For more information and installation instructions visit <https://www.dealii.org>
- Netgen/NGSolve
  - For more information and installation instructions visit <https://ngsolve.org>
- MFEM
  - For more information and installation instructions visit <https://mfem.org> and <https://mfem.org/howto/build-systems/> for instructions on how to build using cmake
  - The parallel build of MFEM will require additional libraries

The programms build the bases to solve PDEs using an adaptive mesh refinement loop (and in some cases hp-adaptive elements).

The programms are written as the main work of my Bachelor Thesis at Friedrich-Alexander University Erlangen-Nuernberg. The german title of the thesis is "Adaptive Lösung partieller Differentialgleichungen mit verschiedenen Finite-Elemente Bibliotheken".

All programms output teh soulution as vtk-files which can be visualised by any compatiable tool (I personally recommend [Visit](https://visit-dav.github.io/visit-website/index.html)).

