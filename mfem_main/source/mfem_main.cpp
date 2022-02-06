//Created by Patrik Rác
//Program including the main method and executing the programm
#include "poisson.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage: ./mfem_main order max_dofs" << std::endl;
        return -1;
    }
    AspDEQuFEL::Poisson l(std::atoi(argv[1]), std::atoi(argv[2]));
    l.run();
}
