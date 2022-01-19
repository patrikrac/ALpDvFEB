//Created by Patrik Rác
<<<<<<< HEAD
//Program including the main method and executing the programm
=======
//This is a general MFEM program designed to solve Laplace-equations using the MFEM Library.
//The dimensionality of the programm can be changed by altering the input grid.

>>>>>>> parallel
#include "poisson.hpp"

int main(int argc, char *argv[])
{
<<<<<<< HEAD
    if (argc != 3)
    {
        std::cout << "Usage: ./mfem_general order max_dofs" << std::endl;
        return -1;
    }
    AspDEQuFEL::Poisson l(std::atoi(argv[1]), std::atoi(argv[2]));
    l.run();
=======

  if (argc != 3)
  {
    std::cout << "Usage: ./mfem_general order max_dofs" << std::endl;
    return -1;
  }
  
  // Initialize MPI
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  AspDEQuFEL::Poisson l(num_procs, myid, std::atoi(argv[1]), std::atoi(argv[2]));
  l.run();

  //Finalize MPI
  MPI_Finalize();
  return 0;
>>>>>>> parallel
}
