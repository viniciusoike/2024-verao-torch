# https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/network_1.html
#
# Chapter 6: A neural network from scratch
#

library(torch)

#> Input Data
x <- torch_randn(100, 3)
#> Weights matrix
w <- torch_randn(3, 1, requires_grad = TRUE)
#> Bias vector
b <- torch_zeros(1, 1, requires_grad = TRUE)

#> Initial "predicted" value
y <- x$matmul(w) + b

# A layer with eight units will need a weight matrix with eight columns.
w1 <- torch_randn(3, 8, requires_grad = TRUE)
b1 <- torch_zeros(1, 8, requires_grad = TRUE)
# A "back" layer
w2 <- torch_randn(8, 1, requires_grad = TRUE)
b2 <- torch_randn(1, 1, requires_grad = TRUE)


# Loop --------------------------------------------------------------------

library(torch)

# input dimensionality (number of input features)
d_in <- 3
# number of observations in training set
n <- 100
# Data
x <- torch_randn(n, d_in)
# Initial values for coefficients
coefs <- c(0.2, -1.3, -0.5)
# Compute the function f(x) = WX + b
y <- x$matmul(coefs)$unsqueeze(2) + torch_randn(n, 1)


# dimensionality of hidden layer
d_hidden <- 32
# output dimensionality (number of predicted features)
d_out <- 1

# weights connecting input to hidden layer
w1 <- torch_randn(d_in, d_hidden, requires_grad = TRUE)
# weights connecting hidden to output layer
w2 <- torch_randn(d_hidden, d_out, requires_grad = TRUE)

# hidden layer bias
b1 <- torch_zeros(1, d_hidden, requires_grad = TRUE)
# output layer bias
b2 <- torch_zeros(1, d_out, requires_grad = TRUE)

# 1. do a forward pass, yielding the network’s predictions
# (if you dislike the one-liner, feel free to split it up);
#
# 2. compute the loss
# (this, too, being a one-liner – we merely added some logging);
#
# 3. have autograd calculate the gradient of the loss
# with respect to the parameters; and
#
# 4. update the parameters accordingly
# (again, taking care to wrap the whole action in with_no_grad(),
# and zeroing the grad fields on every iteration).

learning_rate <- 1e-4
num_iterations <- 200


### training loop ----------------------------------------

for (t in seq_len(num_iterations)) {

  ### -------- Forward pass --------

  y_pred <- x$mm(w1)$add(b1)$relu()$mm(w2)$add(b2)

  ### -------- Compute loss --------
  loss <- (y_pred - y)$pow(2)$mean()
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")

  ### -------- Backpropagation --------

  # compute gradient of loss w.r.t. all tensors with
  # requires_grad = TRUE
  loss$backward()

  ### -------- Update weights --------

  # Wrap in with_no_grad() because this is a part we don't
  # want to record for automatic gradient computation
  with_no_grad({
    w1 <- w1$sub_(learning_rate * w1$grad)
    w2 <- w2$sub_(learning_rate * w2$grad)
    b1 <- b1$sub_(learning_rate * b1$grad)
    b2 <- b2$sub_(learning_rate * b2$grad)

    # Zero gradients after every pass, as they'd
    # accumulate otherwise
    w1$grad$zero_()
    w2$grad$zero_()
    b1$grad$zero_()
    b2$grad$zero_()
  })

}

# Refazer exemplo aula ----------------------------------------------------

cars_scale <- cars |>
  dplyr::mutate(
    speed = scale(speed),
    dist = scale(dist)
  )

y <- torch_tensor(cars_scale$dist)
x <- torch_tensor(cars_scale$speed)

b <- torch_zeros(1, 1, requires_grad = TRUE)
w <- torch_randn(1, 1, requires_grad = TRUE)


x$mm(w)$add(b)

# Montar uma rede com 32 hidden layers
w1 <- torch_randn(1, 32, requires_grad = TRUE)
w2 <- torch_randn(32, 1, requires_grad = TRUE)

b1 <- torch_zeros(1, 32, requires_grad = TRUE)
b2 <- torch_randn(1, 1, requires_grad = TRUE)

x$mm(w)$add(b)$relu()

num_iterations <- 5000
lr <- 0.001

for (i in seq_len(num_iterations)) {

  vl_linear <- x$mm(w1)$add(b1)
  act_fun <- vl_linear$relu()
  y_hat <- act_fun$mm(w2)$add(b2)

  loss <- (y_hat - y)$pow(2)$mean()

  if (i %% 50 == 0)
    cat("Iteração (Época): ", i, "   Perda: ", loss$item(), "\n")

  # Gradiente

  loss$backward()

  # Atualiza os parâmetros

  with_no_grad({
    # atualizar parâmetros
    w1$sub_(lr * w1$grad)
    w2$sub_(lr * w2$grad)
    b1$sub_(lr * b1$grad)
    b2$sub_(lr * b2$grad)

    # zerar gradientes
    w1$grad$zero_()
    w2$grad$zero_()
    b1$grad$zero_()
    b2$grad$zero_()
  })

}

ggplot2::ggplot(cars_scale) +
  ggplot2::aes(x = speed, y = dist) +
  ggplot2::geom_point() +
  ggplot2::geom_smooth(method = "lm", se = FALSE) +
  ggplot2::geom_line(
    colour = "red",
    data = data.frame(speed = cars_scale$speed, dist = as.numeric(y_hat))
  )
