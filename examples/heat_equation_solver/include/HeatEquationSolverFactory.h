#ifndef HEAT_EQUATION_SOLVER_FACTORY_H
#define HEAT_EQUATION_SOLVER_FACTORY_H

#include "HeatEquationSolverBase.h"
#include "HeatEquationSolverNoStreams.h"
#include "HeatEquationSolverWithStreams.h"

enum SolverType {
    NO_STREAMS,
    WITH_STREAMS
};

HeatEquationSolverBase* create_solver(SolverType type, int nx, int ny, float dx, float dy, float dt, float alpha, int num_threads) {
    if (type == NO_STREAMS) {
        return new HeatEquationSolverNoStreams(nx, ny, dx, dy, dt, alpha, num_threads);
    } else {
        return new HeatEquationSolverWithStreams(nx, ny, dx, dy, dt, alpha, num_threads);
    }
}

#endif // HEAT_EQUATION_SOLVER_FACTORY_H
