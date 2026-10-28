"""Exercicio 1 - conferencia da conta feita na mao.

A conta do Exercicio 1 esta escrita passo a passo no relatorio. Este script
nao substitui a conta: ele so confirma os numeros de duas formas independentes:

  1. Python puro (math.tanh), seguindo as mesmas formulas do relatorio;
  2. a propria classe MLP de mlp.py (a mesma dos Exercicios 2-4), com os
     pesos do enunciado colocados a mao e a perda MSE com saida tanh.

Uso (a partir da raiz do repositorio):

    python docs/exercises/mlp/code/exercise1_check.py
"""

import math

import numpy as np

from mlp import MLP, MSEWithTanh

# Valores do enunciado
x = [0.5, -0.2]
y = 1.0
W1 = [[0.3, -0.1], [0.2, 0.4]]
b1 = [0.1, -0.2]
W2 = [0.5, -0.3]
b2 = 0.2
eta = 0.3


def by_hand():
    """Mesma sequencia de contas do relatorio, em Python puro."""
    # A - passagem direta
    z1 = [W1[i][0] * x[0] + W1[i][1] * x[1] + b1[i] for i in range(2)]
    h1 = [math.tanh(z) for z in z1]
    u2 = W2[0] * h1[0] + W2[1] * h1[1] + b2
    y_hat = math.tanh(u2)
    loss = (y - y_hat) ** 2

    # B - passagem reversa
    dL_dyhat = 2 * (y_hat - y)
    dL_du2 = dL_dyhat * (1 - y_hat ** 2)
    dL_dW2 = [dL_du2 * h1[0], dL_du2 * h1[1]]
    dL_db2 = dL_du2
    dL_dh1 = [dL_du2 * W2[0], dL_du2 * W2[1]]
    dL_dz1 = [dL_dh1[i] * (1 - h1[i] ** 2) for i in range(2)]
    dL_dW1 = [[dL_dz1[i] * x[j] for j in range(2)] for i in range(2)]
    dL_db1 = dL_dz1

    # C - atualizacao
    new_W2 = [W2[i] - eta * dL_dW2[i] for i in range(2)]
    new_b2 = b2 - eta * dL_db2
    new_W1 = [[W1[i][j] - eta * dL_dW1[i][j] for j in range(2)] for i in range(2)]
    new_b1 = [b1[i] - eta * dL_db1[i] for i in range(2)]

    # Segunda passagem direta, com os parametros novos
    z1n = [new_W1[i][0] * x[0] + new_W1[i][1] * x[1] + new_b1[i] for i in range(2)]
    h1n = [math.tanh(z) for z in z1n]
    y_hat_n = math.tanh(new_W2[0] * h1n[0] + new_W2[1] * h1n[1] + new_b2)
    loss_n = (y - y_hat_n) ** 2

    return {
        "z1": z1, "h1": h1, "u2": u2, "y_hat": y_hat, "loss": loss,
        "dL_dyhat": dL_dyhat, "dL_du2": dL_du2, "dL_dW2": dL_dW2, "dL_db2": dL_db2,
        "dL_dh1": dL_dh1, "dL_dz1": dL_dz1, "dL_dW1": dL_dW1, "dL_db1": dL_db1,
        "new_W2": new_W2, "new_b2": new_b2, "new_W1": new_W1, "new_b1": new_b1,
        "y_hat_second": y_hat_n, "loss_second": loss_n,
    }


def with_mlp_class():
    """Os mesmos numeros, agora pela classe MLP usada no resto da atividade."""
    net = MLP([2, 2, 1], MSEWithTanh, rng=np.random.default_rng(0))
    net.W = [np.array(W1, dtype=float), np.array([W2], dtype=float)]
    net.b = [np.array(b1, dtype=float), np.array([b2], dtype=float)]
    X = np.array([x])
    Y = np.array([[y]])

    y_hat = net.forward(X)[0, 0]
    loss = net.loss.value(Y, net.activations[-1])
    net.backward(Y)
    net.update(eta)
    return y_hat, loss, net


def main():
    r = by_hand()
    f = lambda v: f"{v:.4f}"
    print("Exercicio 1 - conferencia (4 casas)")
    print("z1      =", [f(v) for v in r["z1"]])
    print("h1      =", [f(v) for v in r["h1"]])
    print("u2      =", f(r["u2"]), "| y_hat =", f(r["y_hat"]), "| L =", f(r["loss"]))
    print("dL/dyhat =", f(r["dL_dyhat"]), "| dL/du2 =", f(r["dL_du2"]))
    print("dL/dW2  =", [f(v) for v in r["dL_dW2"]], "| dL/db2 =", f(r["dL_db2"]))
    print("dL/dh1  =", [f(v) for v in r["dL_dh1"]], "| dL/dz1 =", [f(v) for v in r["dL_dz1"]])
    print("dL/dW1  =", [[f(v) for v in row] for row in r["dL_dW1"]])
    print("dL/db1  =", [f(v) for v in r["dL_db1"]])
    print("W2 novo =", [f(v) for v in r["new_W2"]], "| b2 novo =", f(r["new_b2"]))
    print("W1 novo =", [[f(v) for v in row] for row in r["new_W1"]])
    print("b1 novo =", [f(v) for v in r["new_b1"]])
    print("2a passagem: y_hat =", f(r["y_hat_second"]), "| L =", f(r["loss_second"]))

    y_hat, loss, net = with_mlp_class()
    assert abs(y_hat - r["y_hat"]) < 1e-12 and abs(loss - r["loss"]) < 1e-12
    assert np.allclose(net.W[0], r["new_W1"]) and np.allclose(net.W[1][0], r["new_W2"])
    assert np.allclose(net.b[0], r["new_b1"]) and np.isclose(net.b[1][0], r["new_b2"])
    print("classe MLP: mesmos valores (asserts ok)")
    return r


if __name__ == "__main__":
    main()
