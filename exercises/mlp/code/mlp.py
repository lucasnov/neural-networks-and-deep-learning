"""MLP escrito do zero com NumPy: ativacoes, perdas, passagem direta,
passagem reversa (retropropagacao) e atualizacao por gradiente descendente.

Este arquivo e o nucleo usado, sem nenhuma alteracao, pelos Exercicios 2, 3 e 4
(e pela conferencia do Exercicio 1). O que muda entre os exercicios e passado
como argumento: os tamanhos das camadas, a ativacao da saida e a funcao de perda.

Convencoes (as mesmas do enunciado):
  - X tem formato (N, d): uma amostra por linha.
  - A camada l tem pesos W[l] com formato (n_saida, n_entrada) e bias b[l] com
    formato (n_saida,). A pre-ativacao de uma amostra e z = W x + b; para o
    lote inteiro isso vira Z = X @ W.T + b (cada linha e uma amostra).
  - Y (alvo) tem formato (N, n_classes_saida): (N, 1) no caso binario e
    one-hot (N, 3) no caso multiclasse.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Funcoes de ativacao
# ---------------------------------------------------------------------------

def tanh(z):
    return np.tanh(z)


def tanh_derivative_from_output(h):
    """d/dz tanh(z) = 1 - tanh(z)^2. Recebe h = tanh(z), ja calculado no forward."""
    return 1.0 - h ** 2


def sigmoid(z):
    # Formula estavel: evita exp de numero grande positivo, que daria overflow.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def softmax(z):
    # Subtrair o maximo de cada linha nao muda o resultado (o fator e^-max
    # cancela no numerador e no denominador) e impede overflow no exp.
    shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


OUTPUT_ACTIVATIONS = {"tanh": tanh, "sigmoid": sigmoid, "softmax": softmax}


# ---------------------------------------------------------------------------
# Funcoes de perda
#
# Cada perda tem dois metodos:
#   value(Y, P)        -> numero: a perda media sobre as N amostras
#   output_delta(Y, P) -> matriz (N, n_saida): dL/du, o gradiente da perda em
#                         relacao a pre-ativacao da camada de saida.
# Cada perda ja assume a ativacao de saida com que ela forma par (ver nome).
# ---------------------------------------------------------------------------

EPS = 1e-12  # protege o log(0)


class MSEWithTanh:
    """Erro quadratico medio com saida tanh (usado so na conferencia do Ex. 1)."""

    output_activation = "tanh"

    @staticmethod
    def value(Y, P):
        return float(np.mean(np.sum((Y - P) ** 2, axis=1)))

    @staticmethod
    def output_delta(Y, P):
        n = Y.shape[0]
        # dL/dy_hat = (2/N)(y_hat - y) ; dy_hat/du = 1 - y_hat^2
        return (2.0 / n) * (P - Y) * (1.0 - P ** 2)


class BinaryCrossEntropy:
    """Entropia cruzada binaria com saida sigmoide (Exercicio 2)."""

    output_activation = "sigmoid"

    @staticmethod
    def value(Y, P):
        P = np.clip(P, EPS, 1.0 - EPS)
        return float(-np.mean(Y * np.log(P) + (1.0 - Y) * np.log(1.0 - P)))

    @staticmethod
    def output_delta(Y, P):
        # Com sigmoide + BCE, dL/du simplifica para (y_hat - y) / N.
        return (P - Y) / Y.shape[0]


class CategoricalCrossEntropy:
    """Entropia cruzada categorica com saida softmax (Exercicios 3 e 4)."""

    output_activation = "softmax"

    @staticmethod
    def value(Y, P):
        P = np.clip(P, EPS, 1.0)
        return float(-np.mean(np.sum(Y * np.log(P), axis=1)))

    @staticmethod
    def output_delta(Y, P):
        # Com softmax + entropia cruzada, dL/du = (p - y) / N (derivado no relatorio).
        return (P - Y) / Y.shape[0]


# ---------------------------------------------------------------------------
# A rede
# ---------------------------------------------------------------------------

class MLP:
    """Perceptron de multiplas camadas com tanh nas camadas ocultas.

    layer_sizes: lista com o tamanho de cada camada, da entrada a saida.
        Ex.: [2, 8, 1] = 2 entradas, uma camada oculta de 8 neuronios, 1 saida.
             [4, 16, 16, 3] = duas camadas ocultas de 16, saida com 3 unidades.
    loss: uma das classes de perda acima; ela define tambem a ativacao de saida.
    rng: gerador np.random.default_rng usado para sortear os pesos iniciais.
    """

    def __init__(self, layer_sizes, loss, rng):
        self.layer_sizes = list(layer_sizes)
        self.loss = loss
        self.output_activation = OUTPUT_ACTIVATIONS[loss.output_activation]

        # Inicializacao de Xavier/Glorot: desvio 1/sqrt(n_entrada) mantem as
        # pre-ativacoes perto de 0, onde a tanh nao esta saturada. Biases em zero.
        self.W, self.b = [], []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.W.append(rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_out, n_in)))
            self.b.append(np.zeros(n_out))

    @property
    def n_parameters(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    def forward(self, X):
        """Passagem direta. Guarda as ativacoes de cada camada para o backward."""
        self.activations = [X]          # activations[0] = entrada
        h = X
        last = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b             # pre-ativacao: z = W h + b, para cada amostra
            if l < last:
                h = tanh(z)             # camadas ocultas: tanh
            else:
                h = self.output_activation(z)   # saida: depende da perda
            self.activations.append(h)
        return h                        # previsao: probabilidades (ou y_hat)

    def backward(self, Y):
        """Passagem reversa: calcula dL/dW e dL/db de todas as camadas.

        Usa as ativacoes guardadas no ultimo forward. O delta de cada camada e
        dL/dz daquela camada; ele vem da camada seguinte pela regra da cadeia.
        """
        P = self.activations[-1]
        delta = self.loss.output_delta(Y, P)        # dL/du da camada de saida
        self.grad_W = [None] * len(self.W)
        self.grad_b = [None] * len(self.b)
        for l in reversed(range(len(self.W))):
            h_prev = self.activations[l]            # entrada desta camada
            # Gradientes dos parametros: soma sobre as amostras do lote
            # (o 1/N ja esta dentro do delta).
            self.grad_W[l] = delta.T @ h_prev       # (n_saida, n_entrada)
            self.grad_b[l] = delta.sum(axis=0)      # (n_saida,)
            if l > 0:
                # Volta pela matriz de pesos: dL/dh_prev = delta @ W ...
                grad_h_prev = delta @ self.W[l]
                # ... e pela derivada da tanh da camada anterior: dL/dz_prev.
                delta = grad_h_prev * tanh_derivative_from_output(h_prev)

    def update(self, learning_rate):
        """Gradiente descendente: parametro <- parametro - eta * gradiente."""
        for l in range(len(self.W)):
            self.W[l] -= learning_rate * self.grad_W[l]
            self.b[l] -= learning_rate * self.grad_b[l]

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        P = self.forward(X)
        if P.shape[1] == 1:                         # binario: limiar 0.5
            return (P[:, 0] >= 0.5).astype(int)
        return P.argmax(axis=1)                     # multiclasse: maior probabilidade


def train(model, X, Y, epochs, learning_rate):
    """Treino por gradiente descendente em lote completo (full batch).

    Em cada epoca: forward com todo o treino, perda, backward, atualizacao.
    Devolve a lista com a perda de treino de cada epoca.
    """
    losses = []
    for _ in range(epochs):
        P = model.forward(X)
        losses.append(model.loss.value(Y, P))
        model.backward(Y)
        model.update(learning_rate)
    return losses


def one_hot(y, n_classes):
    """Converte rotulos inteiros (N,) em matriz one-hot (N, n_classes)."""
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))
