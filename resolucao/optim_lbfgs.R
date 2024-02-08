# https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/network_1.html
#
# Chapter 10: Function minimization with L-BFGS
#
library(torch)

#> Revisiting the rosenbrock function
a <- 1
b <- 5

rosenbrock <- function(x) {
  x1 <- x[1]
  x2 <- x[2]
  (a - x1)^2 + b * (x2 - x1^2)^2
}

x <- torch_tensor(c(-1, 1), requires_grad = TRUE)
value <- rosenbrock(x)
value$backward()

opt <- optim_lbfgs(x)

num_iterations <- 2

x <- torch_tensor(c(-1, 1), requires_grad = TRUE)

optimizer <- optim_lbfgs(x)

calc_loss <- function() {
  optimizer$zero_grad()

  value <- rosenbrock(x)
  cat("Value is: ", as.numeric(value), "\n")

  value$backward()
  value
}

for (i in 1:num_iterations) {
  cat("\nIteration: ", i, "\n")
  optimizer$step(calc_loss)
}

# Using line search

num_iterations <- 2

x <- torch_tensor(c(-1, 1), requires_grad = TRUE)
#> Strong Wolfe
optimizer <- optim_lbfgs(x, line_search_fn = "strong_wolfe")

calc_loss <- function() {
  optimizer$zero_grad()

  value <- rosenbrock(x)
  cat("Value is: ", as.numeric(value), "\n")

  value$backward()
  value
}

for (i in 1:num_iterations) {
  cat("\nIteration: ", i, "\n")
  optimizer$step(calc_loss)
}
