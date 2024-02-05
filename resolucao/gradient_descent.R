library(dplyr)

mtcars_scale <- scale(mtcars)

summary(model_lm <- lm(mpg ~ wt + qsec + am, data = as.data.frame(mtcars_scale)))

y <- dat$mpg
X <- cbind(1, dat$wt)
N <- nrow(dat)

b0 <- runif(1)
b1 <- runif(1)

(yhat <- b0 + b1 * dat$wt)

error <- y - yhat
mse <- mean(error^2)

(gb0 <- sum(y - yhat) * (-2/N))
(gb1 <- sum((y - yhat) * dat$wt) * (-2/N))

alpha = 0.001

b0_new <- b0 - alpha * gb0
b1_new <- b1 - alpha * gb1


# Loop --------------------------------------------------------------------

num_iterations <- 10000
#> Learning-rate
alpha <- 0.001

y <- dat$mpg
x <- dat$wt
N <- nrow(dat)

#> Valores iniciais para estimativas dos parâmetros
b0 <- 0
b1 <- runif(1)

for (i in seq_len(num_iterations)) {
  #print(i)

  if (i %% 1000 == 0) cat("Iteração: ", i, "\n")

  #> Calcula o valor previsto
  yhat <- b0 + b1 * x

  #> Calcula a "função de perda"
  error <- y - yhat
  mse <- mean(error^2)

  if (i %% 1000 == 0) {
    cat("Valor da perda: ", as.numeric(mse), "\n")
  }

  #> Calcula o gradiente nos pontos atuais
  gb0 <- sum(y - yhat) * (-2/N)
  gb1 <- sum((y - yhat) * dat$wt) * (-2/N)
  #> Atualiza o valor dos parâmetros usando o gradiente
  b0_new <- b0 - alpha * gb0
  b1_new <- b1 - alpha * gb1

  b0 <- b0_new
  b1 <- b1_new

  if (i %% 1000 == 0) {
    cat("Betas: ", c(b0, b1), "\n\n")
  }

}

# Modelo ------------------------------------------------------------------

model_lm <- lm(mpg ~ wt, data = dat)
summary(model_lm)


# Regressão Múltipla ------------------------------------------------------

grad <- function(beta) {

  (2/N) * t(X) %*% (X %*% beta - y)

}

loss <- function(beta) {

  e = y - X %*% beta

  t(e) %*% e

}

dat <- mtcars |>
  select(c("mpg", "wt", "qsec", "am")) |>
  mutate(across(everything(), ~as.numeric(scale(.x))))

y <- dat$mpg
X <- as.matrix(dat[, c("wt", "qsec", "am")])
X <- cbind(1, X)
colnames(X)[1] <- c("coef")
N <- nrow(X)

beta <- runif(ncol(X))
yhat <- X %*% beta
loss(beta)
beta_new <- beta - alpha * grad(beta)


beta <- runif(ncol(X))
num_iterations <- 10000
alpha <- 0.001

for (i in seq_len(num_iterations)) {

  if (i %% 1000 == 0) cat("Iteração: ", i, "\n")

  #> Calcula o valor previsto
  yhat <- X %*% beta

  #> Calcula a "função de perda"
  vl_loss <- loss(beta)

  if (i %% 1000 == 0) {
    cat("Valor da perda: ", as.numeric(vl_loss), "\n")
  }

  #> Calcula o gradiente nos pontos atuais
  grad_current <- grad(beta)
  #> Atualiza o valor dos parâmetros usando o gradiente
  beta_current <- beta - alpha * grad_current

  beta <- beta_current

  if (i %% 1000 == 0) {
    cat("Betas: ", beta, "\n\n")
  }

}

summary(model_lm <- lm(mpg ~ wt + qsec + am, data = dat))
