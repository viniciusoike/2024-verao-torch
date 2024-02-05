# Exercício Teórico 1: Entendimento da Regressão Linear
# Explique, com suas próprias palavras, o que a equação
# y = β0 + β1 * x + ε representa no contexto da regressão linear.

# A equação define uma reta com intercepto β0 e coeficiente de inclinação β1.
# O termo ε é um erro ou ruído. Na regressão linear, tenta-se encontrar a reta
# que melhor se ajusta aos dados, no sentido de ter a menor distância aos pontos.
# Formalmente, minimiza-se a soma da distância quadrática (ou mean squared error).


# Exercício Teórico 2: Função de Perda e Verossimilhança
# Pergunta: Descreva a relação entre a função de perda do erro quadrático médio (MSE)
# e a verossimilhança na regressão linear.
# Por que minimizar o MSE é equivalente a maximizar a verossimilhança?

# Tablet


# Exercício Prático 3: Cálculo de Gradientes
library(torch)

t1 <- torch_tensor(10, requires_grad = TRUE)
# Realize algumas operações com t1, diferentes das vistas em aula. Exemplo:

t2 <- t1 + 2 # mude isso
t3 <- t2$square() # e isso

# Agora calcule o gradiente
t3$backward()
print(t1$grad)


#> Cobb-Douglas f(x, y) = x^(1/2) + y^(1/2)

x <- torch_tensor(1, requires_grad = TRUE)
y <- torch_tensor(1, requires_grad = TRUE)

t1 <- x$square()
t2 <- y$square()

t3 <- t1 + t2
t3$backward()

print(x$grad)
print(y$grad)

# Compare o resultado do gradiente calculado manualmente.

# Exercício Teórico 4: Derivação do Gradiente
# Obtenha o gradiente da função de perda MSE em relação a β0 e β1.
# Como isso se relaciona com a atualização dos parâmetros na descida de gradiente?
#
# MSE:


# O gradiente indica a direção de maior incremento de uma função. A descida de
# gradiente utiliza esta informação para encontrar o caminho de maior "decaimento"
# da função.

# Exercício Prático 5: Implementação da Regressão Linear com Autograd
# Use o dataset 'mtcars' para implementar uma regressão linear múltipla
# usando descida de gradiente com autograd.

dat <- mtcars |>
  select(mpg, wt, qsec, am) |>
  mutate(across(everything(), ~as.numeric(scale(.x))))

#> Resultado da regressão via função lm()
summary(model_lm <- lm(mpg ~ wt + qsec + am, data = dat))

# Implemente a regressão linear e compare com a solução analítica.

y <- dat$mpg
X <- dat[, c("wt", "qsec", "am")]
X <- as.matrix(cbind(1, X))

y <- torch_tensor(y)
X <- torch_tensor(X)

#> Modelo linear simples
model_lm <- function(X, beta) {
  X$matmul(beta)
}
#> Função de perda
mse <- function(beta) {
  resid <- X$matmul(beta) - y
  loss <- resid$square()$mean()
  return(loss)
}

## parâmetros da otimização
num_iterations <- 10000
lr <- 0.001 # learning rate, alpha

## parâmetros do modelo
beta <- torch_rand(ncol(dat), requires_grad = TRUE)

#> Calcula para um caso simples

vl_loss <- mse(beta)

vl_loss$backward()

beta$grad

beta$sub_(lr * beta$grad)
#> Substitui por zero
beta$grad$zero_()

beta$sub_(lr * beta$grad)
beta$grad$zero_()


for (i in seq_len(num_iterations)) {

  #> Calcula valor da função de perda
  vl_loss <- mse(beta)
  #> Atualiza os parâmetros via descida de gradiente
  vl_loss$backward()
}





#1:num_iterations
for (i in seq_len(num_iterations)) {

  if (i %% 100 == 0) cat("Iteração: ", i, "\n")

  perda <- mse(beta)

  if (i %% 100 == 0) {
    cat("Valor da perda: ", as.numeric(perda), "\n")
  }

  # calcula a derivada
  perda$backward()
  if (i %% 100 == 0) {
    cat("A derivada é: ", as.matrix(beta$grad), "\n\n")
  }

  with_no_grad({
    beta$sub_(lr * beta$grad)
    beta$grad$zero_()
  })
}

beta





# Exercício Prático 6: MLP com Descida de Gradiente
# Construa e treine uma MLP simples para o dataset 'mtcars'.
# Implemente a MLP e o processo de treinamento aqui.

# Exercício Teórico 7: Compreensão de Otimizadores
# Pergunta: Explique a diferença entre a descida de gradiente simples e
# métodos de otimização como L-BFGS ou Adam.
# Em que situações um pode ser preferível ao outro?
