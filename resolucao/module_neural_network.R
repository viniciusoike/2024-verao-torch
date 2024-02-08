# https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/network_2.html
#
# Chapter 11: Modularizing the Neural Network
#

# The forward pass, instead of calling functions on tensors, will call the model.
# In computing the loss, we now make use of torch’s nnf_mse_loss().
# Backpropagation of gradients is, in fact, the only operation that remains unchanged.
# Weight updating is taken care of by the optimizer.


## Data --------------------------------------------------------------------

# input dimensionality (number of input features)
d_in <- 3
# number of observations in training set
n <- 100
# Random matrix
x <- torch_randn(n, d_in)
# Initial values for parameters
coefs <- c(0.2, -1.3, -0.5)
# Initial value for yhat
y <- x$matmul(coefs)$unsqueeze(2) + torch_randn(n, 1)

## Network -----------------------------------------------------------------

# A neural network with two linear hidden layers and a reLU activation function

# dimensionality of hidden layer
d_hidden <- 32
# output dimensionality (number of predicted features)
d_out <- 1

net <- nn_sequential(
  nn_linear(d_in, d_hidden),
  nn_relu(),
  nn_linear(d_hidden, d_out)
)

## Training ----------------------------------------------------------------

#> Optimizer uses Adam
opt <- optim_adam(net$parameters)

### training loop --------------------------------------

for (t in 1:500) {

  ### -------- Forward pass --------
  y_pred <- net(x)

  ### -------- Compute loss --------
  loss <- nnf_mse_loss(y_pred, y)
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")

  ### -------- Backpropagation --------
  opt$zero_grad()
  loss$backward()

  ### -------- Update weights --------
  opt$step()

}

plot(y_pred, y)
abline(0, 1)
