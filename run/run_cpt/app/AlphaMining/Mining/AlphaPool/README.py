# composite alpha can be synthesized using different ways:
#   linear, polynomial, exponential, logarithmic, SVN with non-linear kernels, NN, ensemble, ...

# Linear alpha composition means the final alpha is a linear combination (of weights) of other alphas,
# where the coefficients (weights) are typically independent of the alpha values.

# NOTE: if weights of a Linear alpha composition are dynamic(time-varying),
# it MAY or MAY not remain linear:
#   if every weights are uncorrelated to all alphas, even it is time-varying, it is still a linear system
#   else it is inherently a non-linear system
