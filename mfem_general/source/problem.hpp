//Created by Patrik Rác
//This file contains the problem definitions
#include "mfem.hpp"
#pragma once

using namespace mfem;

// Exact solution, used for the Dirichlet BC.
double bdr_func(const Vector &p);

// Right hand side function
double rhs_func(const Vector &p);