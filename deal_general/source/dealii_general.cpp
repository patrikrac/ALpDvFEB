/*  ----------------------------------------------------------------
Created by Patrik Rac
Programm solving a partial differential equation of arbitrary dimensionality using the functionality of the deal.ii FE-library.
The classes defined here can be modified in order to solve each specific Problem.
---------------------------------------------------------------- */
#include <iostream>
#include "poisson_h.hpp"
#include "poisson_hp.hpp"

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4)
    {
        std::cout << "Usage: ./deal_general order max_dofs" << std::endl;
        return -1;
    }

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    AspDEQuFEL::Poisson<3> l(std::atoi(argv[1]), std::atoi(argv[2]));
    l.run();
    return 0;
}