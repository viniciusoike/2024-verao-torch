library(torch)

# Exercício 1: Explorando Dimensões de Tensores
# Considere um tensor criado com torch_tensor(1:12).
# Qual seria a dimensão desse tensor?

t0 <- torch_tensor(1:12)
#> Dimensão 1
t0
t0$dim()

# Exercício 2: Operações Básicas
# Dados dois tensores A e B, ambos com valores torch_tensor(1:4),
# qual seria o resultado de A * B?

A <- torch_tensor(1:4)
B <- torch_tensor(1:4)

A * B

A$multiply(B)

# Exercício 3: Desafio de Redimensionamento
# Se você tiver um tensor de 12 elementos,
# quais dimensões você poderia usar para redimensioná-lo em
# uma matriz 2D sem causar erro? Mostre como você faria isso.

t0 <- torch_tensor(1:12)

t0$view(c(3, 4))
t0$view(c(4, 3))
t0$view(c(6, 2))
t0$view(c(2, 6))
t0$view(c(1, 12))

# Também é possível adicionar um número arbitrário de dimensões "redundantes"
t0$unsqueeze(1)
t0$view(c(1, 1, 1, 1, 1, 1, 1, 1, 12, 1))

# Exercício 4: Slicing Avançado
# Considere um tensor 3D com dimensões (4, 4, 4).
# Como você acessaria apenas a segunda e terceira coluna da segunda "página"?
# Experimente várias formas de fazer isso.

t0 <- torch_randn(c(4, 4, 4))

t0[2, .., 2:3]
t0[2, .., c(2, 3)]


# Exercício 5: Broadcasting
# Dados tensor_a5 de dimensão (3, 2) e tensor_b5 de dimensão (2,),
# qual seria o resultado de tensor_a5 + tensor_b5? E se tentássemos somar
# tensor_a5 com um tensor de dimensão (3,), isso causaria um erro?

tensor_a5 <- torch_randn(c(3, 2))
tensor_b5 <- torch_ones(c(2, 1))

#> Não funciona pois a dimensão do tensor a (3) não bate com a dimensão do tensor b (2)
tensor_a5 + tensor_b5

#> Agora a soma funciona via broadcasting
tz <- torch_randn(3, 1)
tensor_a5 + tz

#> Assim também funciona
tz <- torch_randn(3, 2)
tensor_a5 + tz

#> Qualquer outro valor resulta em erro.

# Exercício 6: Interpretando Resultados de Operações Matriciais
# Se você multiplicar um tensor A (dimensão 2x3) por um tensor B (dimensão 3x2),
# qual será a dimensão do resultado? Qual seria a dimensão se você transpuser
# o resultado?

# O resultado da multiplicação de matrizes será 2x2
A <- torch_randn(c(2, 3))
B <- torch_randn(c(3, 2))

A$mm(B)

# Exercício 7: Manipulação de Datasets com Tensores
# Após converter o dataset 'mtcars' para um tensor, que passos você
# tomaria para calcular a média da primeira coluna (mpg)? Escreva o código.

# Calcula a média apenas da coluna mtcars
t0[, 1]$mean()
t0[, grep("mpg", names(mtcars))]$mean()
# Calcula a média de todas as colunas
t0$mean(dim = 1)

mean(mtcars$mpg)


# Exercício 8: Predição de Decomposição de Matrizes
# Ao realizar uma decomposição QR em um tensor quadrado 4x4,
# quantas matrizes e de quais dimensões você espera receber?

t0 <- torch_randn(c(4, 4))
#> Espera-se ter duas matrizes 4x4.
#> Uma matriz Q ortonormal e uma matriz R triangular superior.
linalg_qr(t0)

# Exercício 9: Desafio Prático - Regressão Linear
# Dado o dataset 'cars', com duas colunas (x e y), como você utilizaria
# tensores para calcular os coeficientes de uma regressão linear de y em x?
# Dica: Considere a fórmula da regressão linear (β = (X'X)^-1 X'y).
# Calcule todas as formas que você conhece para fazer isso e
# compare os resultados.

X <- cbind(1, cars$speed)
Y <- matrix(cars$dist, ncol = 1)

# Método usando matrizes
# solve(t(X) %*% X) %*% t(X) %*% Y

#> Converte para tensors a aplica as operações manualmente
x <- torch_tensor(X)
y <- torch_tensor(Y)

xtx <- x$t()$mm(x)
xty <- x$t()$mm(y)
#> Resultado final
linalg_inv(xtx)$mm(xty)

#> Usando a função lstsq
linalg_lstsq(x, y)

#> Decomposição QR

## Lembrando que, a decomposição QR é tal que X = QR
## Ax = b
## QRx = b
## Rx = Q^-1b
## Rx = Qtb

library(zeallot)

c(Q, R) %<-% linalg_qr(x)
Qtb <- Q$t()$mm(y)

torch_triangular_solve(Qtb, R)
