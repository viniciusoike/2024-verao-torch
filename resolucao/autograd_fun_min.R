# https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/optim_1.html

# Chapter 5: Function minimization with autograd

library(torch)

a <- 1
b <- 5

rosenbrock <- function(x) {
  x1 <- x[1]
  x2 <- x[2]
  (a - x1)^2 + b * (x2 - x1^2)^2
}

#> Optimization parameters

# Learning rate
lr <- 0.01
# Number of iterations
num_iterations <- 1000

x <- torch_tensor(c(-1, 1), requires_grad = TRUE)

with_no_grad({
  #> Gradient descent: lr (learning-rate) * gradient
  x$sub_(lr * x$grad)
  #> Erase previous record of gradient (torch-specific)
  x$grad$zero_()
})

#> Complete code

num_iterations <- 1000

lr <- 0.01

x <- torch_tensor(c(-1, 1), requires_grad = TRUE)

for (i in 1:num_iterations) {
  if (i %% 100 == 0) cat("Iteration: ", i, "\n")

  value <- rosenbrock(x)
  if (i %% 100 == 0) {
    cat("Value is: ", as.numeric(value), "\n")
  }

  value$backward()
  if (i %% 100 == 0) {
    cat("Gradient is: ", as.matrix(x$grad), "\n")
  }

  with_no_grad({
    x$sub_(lr * x$grad)
    x$grad$zero_()
  })
}


#> Learning rate

minimize_fun <- function(n = 1000, lr = 0.01, x0 = c(-1, 1)) {

  x <- torch_tensor(x0, requires_grad = TRUE)

  for (i in seq_len(n)) {

    value <- rosenbrock(x)
    value$backward()

    with_no_grad({
      x$sub_(lr * x$grad)
      x$grad$zero_()
    })
  }

  return(list(x, value))

}

minimize_fun()

#> Converges to true minimum
minimize_fun(1000)

#> Pretty bad
minimize_fun(lr = 0.1)
#> Doesn't work
minimize_fun(lr = 1)
#> Not so bag
minimize_fun(lr = 0.05)
#> Barely changes
minimize_fun(lr = 0.0001)

#> A small learning-rate means the function changes slowly. So a larger number
#> of iterations is needed to achieve optimality

minimize_fun(n = 100, lr = 0.0001)
minimize_fun(n = 500, lr = 0.0001)
minimize_fun(n = 1000, lr = 0.001)
minimize_fun(n = 10000, lr = 0.0001)

minimize_fun(n = 100, lr = 0.001)
minimize_fun(n = 500, lr = 0.001)
minimize_fun(n = 1000, lr = 0.001)
minimize_fun(n = 10000, lr = 0.001)

minimize_fun(n = 100, lr = 0.01)
minimize_fun(n = 500, lr = 0.01)
minimize_fun(n = 1000, lr = 0.01)
minimize_fun(n = 10000, lr = 0.01)
