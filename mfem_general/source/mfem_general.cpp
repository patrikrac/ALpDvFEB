//Created by Patrik Rác
//This is a general MFEM program designed to solve Laplace-equations using the MFEM Library.
//The dimensionality of the programm can be changed by altering the input grid.

#include "problem.cpp"

int main(int argc, char *argv[])
{
   if (argc != 3)
  {
    std::cout << "Usage: ./mfem_general order max_dofs" << std::endl;
    return -1;
  }
   Problem l(std::atoi(argv[1]), std::atoi(argv[2]));
   l.run();
}


