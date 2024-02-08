# https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/training_with_luz.html

# Chapter 14: Training with luz

library(torch)
library(luz)


## Data --------------------------------------------------------------------

# input dimensionality (number of input features)
d_in <- 3
# number of observations in training set
n <- 1000

x <- torch_randn(n, d_in)
coefs <- c(0.2, -1.3, -0.5)
y <- x$matmul(coefs)$unsqueeze(2) + torch_randn(n, 1)

#> dataset() and dataloader()
ds <- tensor_dataset(x, y)
dl <- dataloader(ds, batch_size = 100, shuffle = TRUE)

## Model -------------------------------------------------------------------

# dimensionality of hidden layer
d_hidden <- 32
# output dimensionality (number of predicted features)
d_out <- 1

net <- nn_module(
  initialize = function(d_in, d_hidden, d_out) {
    self$net <- nn_sequential(
      nn_linear(d_in, d_hidden),
      nn_relu(),
      nn_linear(d_hidden, d_out)
    )
  },
  forward = function(x) {
    self$net(x)
  }
)

## Train the model ---------------------------------------------------------

fitted <- net %>%
  setup(loss = nn_mse_loss(), optimizer = optim_adam) %>%
  set_hparams(
    d_in = d_in,
    d_hidden = d_hidden, d_out = d_out
  ) %>%
  fit(dl, epochs = 200)

# Alternative: using the dataset()
fitted <- net %>%
  setup(loss = nn_mse_loss(), optimizer = optim_adam) %>%
  set_hparams(
    d_in = d_in,
    d_hidden = d_hidden, d_out = d_out
  ) %>%
  fit(ds, epochs = 200)

# Alternative: using the raw torch tensors
fitted <- net %>%
  setup(loss = nn_mse_loss(), optimizer = optim_adam) %>%
  set_hparams(
    d_in = d_in,
    d_hidden = d_hidden, d_out = d_out
  ) %>%
  fit(list(x, y), epochs = 200)

# Alternative: using raw matrix objects

fitted <- net %>%
  setup(loss = nn_mse_loss(), optimizer = optim_adam) %>%
  set_hparams(
    d_in = d_in,
    d_hidden = d_hidden, d_out = d_out
  ) %>%
  fit(list(as.matrix(x), as.matrix(y)), epochs = 200)


## Structure ---------------------------------------------------------------

#> Split 60% of the sample as training
train_ids <- sample(1:length(ds), size = 0.6 * length(ds))
#> Validation set: size is 20% of the sample and composed of a disjoint set between
#> training data and total data
valid_ids <- sample(
  setdiff(1:length(ds), train_ids),
  size = 0.2 * length(ds)
)
#> Test set: takes all the sample that is neither in the training nor the validation
#> sets. As the training and validations sets are disjoint, by definition, takes
#> whatever data "remains" from the full sample.
test_ids <- setdiff(
  1:length(ds),
  union(train_ids, valid_ids)
)

#> In this case: the split was a nice: 60 | 20 | 20.

train_ds <- dataset_subset(ds, indices = train_ids)
valid_ds <- dataset_subset(ds, indices = valid_ids)
test_ds <- dataset_subset(ds, indices = test_ids)

train_dl <- dataloader(train_ds,
                       batch_size = 100, shuffle = TRUE
)
valid_dl <- dataloader(valid_ds, batch_size = 100)
test_dl <- dataloader(test_ds, batch_size = 100)

#> Enhanced workflow with train + validation (using MAE as loss)
fitted <- net %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_mae())
  ) %>%
  set_hparams(
    d_in = d_in,
    d_hidden = d_hidden, d_out = d_out
  ) %>%
  fit(train_dl, epochs = 200, valid_data = valid_dl)

#> Predictions on the test set
fitted %>% predict(test_dl)

#> Evaluate predictions
fitted %>% evaluate(test_dl)

### Callbacks ---------------------------------------------------------------

#> Allows the user to recall useful objects generated during the training process
#>

# With this configuration, weights will be saved, but only if validation loss
# decreases. Training will halt if there is no improvement (again, in validation
# loss) for ten epochs. With both callbacks, you can pick any other metric to
# base the decision on, and the metric in question may also refer to the
# training set.

fitted <- net %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_mae())
  ) %>%
  set_hparams(d_in = d_in,
              d_hidden = d_hidden,
              d_out = d_out) %>%
  fit(
    train_dl,
    epochs = 200,
    valid_data = valid_dl,
    callbacks = list(
      luz_callback_model_checkpoint(path = "./models/",
                                    save_best_only = TRUE),
      luz_callback_early_stopping(patience = 10)
    )
  )

