/*  ----------------------------------------------------------------
Created by Patrik Rac
Programm solving a partial differential equation of arbitrary dimensionality using the functionality of the deal.ii FE-library.
The classes defined here can be modified in order to solve each specific Problem.
---------------------------------------------------------------- */
#include <iostream>
#include "poisson_h.hpp"
#include "poisson_hp.hpp"

using namespace dealii;

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4)
    {
        std::cout << "Usage: ./dealii_general order max_dofs -hp" << std::endl;
        return -1;
    }
    try
    {
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        if(argc == 3)
        {
            AspDEQuFEL::Poisson<3> l(std::atoi(argv[1]), std::atoi(argv[2]));
        } else 
        {
            AspDEQuFEL::PoissonHP<3> l(std::atoi(argv[2]));
        }
        
        l.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}