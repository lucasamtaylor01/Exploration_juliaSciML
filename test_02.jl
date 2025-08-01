# https://docs.sciml.ai/Overview/stable/getting_started/first_optimization/
# https://en.wikipedia.org/wiki/Rosenbrock_function


# Import the package
using Optimization, OptimizationNLopt, ForwardDiff

# Define the problem to optimize
L(u, p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2

u0 = zeros(2)
p = [1.0, 100.0]
optfun = OptimizationFunction(L, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optfun, u0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])

# Solve the optimization problem
sol = solve(prob, NLopt.LD_LBFGS())

typeof(sol)
    

# Analyze the solution
@show sol.u, L(sol.u, p)

