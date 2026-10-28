"""Perceptron de camada unica escrito do zero com NumPy.

Esta e a unica implementacao do perceptron da atividade: o Exercicio 1 e o
Exercicio 2 chamam a mesma funcao train(), sem alteracoes. O rastreio do
"pocket" (melhores pesos ja vistos) ja faz parte do laco desde o Exercicio 1;
la ele nao muda nada, porque o treino termina com 100% de acerto.

Rotulos y em {0, 1}. Pesos w com formato (2,) e bias b escalar.
"""

import numpy as np


def step(z):
    """Funcao degrau: 1 se z >= 0, 0 caso contrario (funciona com escalar ou vetor)."""
    return np.where(z >= 0, 1, 0)


def predict(X, w, b):
    """y_hat = degrau(w . x + b) para todas as linhas de X de uma vez."""
    return step(X @ w + b)


def accuracy(X, y, w, b):
    """Fracao de pontos do dataset classificados corretamente por (w, b)."""
    return float(np.mean(predict(X, w, b) == y))


def train(X, y, w_init, b_init, learning_rate, max_epochs=100):
    """Treina o perceptron com a regra dirigida pelo erro.

    Para cada amostra (x, y), na ordem de X:
        y_hat = degrau(w . x + b)
        erro  = y - y_hat                 (0 se acertou, +1 ou -1 se errou)
        w <- w + eta * erro * x
        b <- b + eta * erro

    Parada: uma epoca inteira sem nenhuma atualizacao (convergiu) ou max_epochs.

    Pocket: depois de cada atualizacao, mede a acuracia no dataset inteiro; se for
    maior que a melhor ja vista, copia (w, b) para o "bolso".
    """
    w = np.array(w_init, dtype=float)       # copia: nao altera o w_init de quem chamou
    b = float(b_init)

    # O pocket comeca com os pesos iniciais (a primeira acuracia "ja vista").
    pocket = {"w": w.copy(), "b": b, "accuracy": accuracy(X, y, w, b), "epoch": 0}

    accuracy_per_epoch = []        # acuracia dos pesos atuais ao fim de cada epoca
    best_so_far_per_epoch = []     # acuracia do pocket ao fim de cada epoca
    updates_per_epoch = []         # quantas amostras geraram atualizacao na epoca
    update_positions = []          # (epoca, indice da amostra) de cada atualizacao
    w_per_epoch, b_per_epoch = [], []   # pesos ao fim de cada epoca
    converged = False

    for epoch in range(1, max_epochs + 1):
        n_updates = 0
        for i, (x_i, y_i) in enumerate(zip(X, y)):
            y_hat = step(x_i @ w + b)
            error = y_i - y_hat
            if error != 0:                  # so atualiza quando erra
                w = w + learning_rate * error * x_i
                b = b + learning_rate * error
                n_updates += 1
                update_positions.append((epoch, i))

                # Pocket: a unica coisa a mais no laco.
                current = accuracy(X, y, w, b)
                if current > pocket["accuracy"]:
                    pocket = {"w": w.copy(), "b": b, "accuracy": current, "epoch": epoch}

        accuracy_per_epoch.append(accuracy(X, y, w, b))
        best_so_far_per_epoch.append(pocket["accuracy"])
        updates_per_epoch.append(n_updates)
        w_per_epoch.append(w.copy())
        b_per_epoch.append(b)

        if n_updates == 0:                  # passagem completa sem atualizar: parou
            converged = True
            break

    return {
        "w": w, "b": b,
        "epochs": len(accuracy_per_epoch),
        "converged": converged,
        "final_accuracy": accuracy_per_epoch[-1],
        "accuracy_per_epoch": accuracy_per_epoch,
        "best_so_far_per_epoch": best_so_far_per_epoch,
        "updates_per_epoch": updates_per_epoch,
        "update_positions": update_positions,
        "w_per_epoch": w_per_epoch,
        "b_per_epoch": b_per_epoch,
        "pocket": pocket,
    }
