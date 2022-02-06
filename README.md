# ALpDvFEB
Collection of programs using different finite-element libraries to adaptively solve partial differential equations.
The used libraries include 
- Deal.ii
  - For more information and installation instructions visit <https://www.dealii.org>
- Netgen/NGSolve
  - For more information and installation instructions visit <https://ngsolve.org>
- MFEM
  - For more information and installation instructions visit <https://mfem.org> and <https://mfem.org/howto/build-systems/> for instructions on how to build using cmake
  - The parallel build of MFEM will require additional libraries

The programms build the bases to solve PDEs using an h-adaptive mesh refinement loop (and for Deal.ii also hp-adaptive).

The programms are written as the main work of my Bachelor Thesis at Friedrich-Alexander Universität Erlangen-Nürnberg titled `Adaptive solution of partial differential equations using different finite-element libraries`.

To visualize the output one needs a VTK visualization tool (I personally recommend [Visit](https://visit-dav.github.io/visit-website/index.html)).

