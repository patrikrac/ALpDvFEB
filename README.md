# ALpDvFEB/AspDEQuFEL
Collection of Programms using different Finite-Element Libraries to solve partial differential equations. 

The programms are implemented as part of my bachelor thesis at Friedrich-Alexander University Erlangen-Nuernberg with the title `"Adaptive solution of partial differential equations using different Finite-Element libraries"` (German: `"Adaptive Lösung partieller Differnzialgleichungen mit verschiedenen Finite-Elemente Bibliotheken"`)

The libraries used include:
- deal.ii
  - For more information and installation instructions visit <https://www.dealii.org> or view the programm specific README
- Netgen/NGSolve
  - For more information and installation instructions visit <https://ngsolve.org> or view the programm specific README
- MFEM
<<<<<<< HEAD
  - For more information and installation instructions visit <https://mfem.org> or view the programm specific README

The programms build the bases to solve PDEs using an adaptive mesh refinement loop (and in some cases optional hp-adaptive elements).
=======
  - For more information and installation instructions visit <https://mfem.org> and <https://mfem.org/howto/build-systems/> for instructions on how to build using cmake
  - The parallel build of MFEM will require additional libraries
>>>>>>> parallel

There is also a parallel branch of the project, where one can find MPI-parallel implementations of each version.  
Additionally, there is a multigrid branch of the project which tries to leverage the capabilities of multigrid methods for performance benefits.

<<<<<<< HEAD
All programms output their cycle-wise soulution as vtk-files which can be visualised by any compatiable tool (I personally recommend [Visit](https://visit-dav.github.io/visit-website/index.html)).
=======
The programms are written as the main work of my Bachelor Thesis at Friedrich-Alexander University Erlangen-Nuernberg. The german title of the thesis is "Adaptive Lösung partieller Differentialgleichungen mit verschiedenen Finite-Elemente Bibliotheken".
>>>>>>> parallel

Additionally the programms output multilple custom txt files containing the calculated exact errors of the run. Those are then visualised by a python-script using plotly and matplotlib.

