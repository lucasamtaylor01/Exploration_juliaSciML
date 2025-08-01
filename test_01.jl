# https://docs.sciml.ai/Overview/stable/getting_started/first_simulation/#first_sim

using ModelingToolkit, DifferentialEquations, Plots

# Define our state variables: state(t) = initial condition
@independent_variables t
@variables x(t)=1 y(t)=1 z(t)=2

# Define our parameters
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0

# Define our differential: takes the derivative with respect to `t`
D = Differential(t)

# Define the differential equations
eqs = [D(x) ~ α * x - β * x * y
       D(y) ~ -γ * y + δ * x * y
       z ~ x + y]

@mtkbuild sys = ODESystem(eqs, t)

tspan = (0.0, 10.0)
prob = ODEProblem(sys, [], tspan)

sol = solve(prob)

p1 = plot(sol, title = "Rabbits vs Wolves")

    