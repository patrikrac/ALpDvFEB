/*  ----------------------------------------------------------------
Created by Patrik Rac
Programm solving a partial differential equation of arbitrary dimensionality using the functionality of the deal.ii FE-library.
The classes defined here can be modified in order to solve each specific Problem.
---------------------------------------------------------------- */
#include "../include/problem_h.hpp"

using namespace dealii;

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    if (argc != 3)
    {
        std::cout << "Usage: ./deal_general order max_dofs" << std::endl;
        return -1;
    }

    problem_h::Problem<3> l(std::atoi(argv[1]), std::atoi(argv[2]));
    l.run();
    return 0;
}