/*  ----------------------------------------------------------------
Created by Patrik Rac
Programm solving a partial differential equation of arbitrary dimensionality using the functionality of the deal.ii FE-library.
The classes defined here can be modified in order to solve each specific Problem.
---------------------------------------------------------------- */
#include "problem_h.cpp"
#include "problem_hp.cpp"

using namespace dealii;

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4)
    {
        std::cout << "Usage: ./deal_general order max_iterations [-hp]" << std::endl;
        return -1;
    }

    if (argc == 3)
    {
        Problem<3> l(std::atoi(argv[1]), std::atoi(argv[2]));
        l.run();
        return 0;
    }
    else 
    {
        ProblemHP<3> l(std::atoi(argv[2]));
        l.run();
        return 0;
    }
}