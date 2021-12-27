//Created by Patrik Rác
//Header for the problem definition of the poisson equation
#include "mfem.hpp"
#pragma once

using namespace mfem;

// Exact solution, used for the Dirichlet BC.
double bdr_func(const Vector &p);

// Right hand side function
double rhs_func(const Vector &p);