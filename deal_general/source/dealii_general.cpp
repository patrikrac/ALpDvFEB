/*  ----------------------------------------------------------------
Created by Patrik Rac
Programm solving a partial differential equation of arbitrary dimensionality using the functionality of the deal.ii FE-library.
The classes defined here can be modified in order to solve each specific Problem.
---------------------------------------------------------------- */
#include "../include/problem_h.hpp"
#include "../include/problem_hp.hpp"

using namespace dealii;

int main(void)
{
    Problem<3> l;
    l.run();
    return 0;
}