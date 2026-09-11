"""Exercicio 1 - Nuvens de pontos: geometria e espalhamento em 2D.

Gera as 4 classes gaussianas do enunciado em quatro escalas de espalhamento,
mede quao separadas elas ficam e salva as Figuras 1, 2 e 3.

Uso (a partir da raiz do repositorio):

    python docs/exercises/data/code/exercise1_point_clouds.py
"""

import numpy as np
import matplotlib.pyplot as plt

from common import SEED, savefig

# Parametros do enunciado: media e desvio padrao (x, y) de cada classe.
MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100
SCALES = [0.5, 1.0, 2.0, 4.0]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def generate_clouds(rng, scale=1.0):
    """Amostra 100 pontos por classe; os desvios sao multiplicados por ``scale``."""
    points, labels = [], []
    for k in range(4):
        cloud = rng.normal(MEANS[k], STDS[k] * scale, size=(N_PER_CLASS, 2))
        points.append(cloud)
        labels.append(np.full(N_PER_CLASS, k))
    return np.vstack(points), np.concatenate(labels)


def nearest_center(points):
    """Rotula cada ponto pela media teorica (MEANS) mais proxima."""
    dists = np.linalg.norm(points[:, None, :] - MEANS[None, :, :], axis=2)
    return dists.argmin(axis=1)


def mixing_rate(points, labels):
    """Fracao de pontos cujo centro de classe mais proximo nao e o da sua classe."""
    return float(np.mean(nearest_center(points) != labels))


def separation_ratios():
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j) para os 6 pares, em s = 1."""
    sigma_bar = STDS.mean(axis=1)  # (sigma_x + sigma_y) / 2 por classe
    ratios = {}
    for i in range(4):
        for j in range(i + 1, 4):
            num = np.linalg.norm(MEANS[i] - MEANS[j])
            ratios[(i, j)] = float(num / (sigma_bar[i] + sigma_bar[j]))
    return ratios


def figure1(points, labels):
    """Figura 1: dispersao em s = 1, centros marcados e regioes do centro mais proximo."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Regiao de decisao geometrica: cor de fundo = classe do centro teorico mais proximo.
    pad = 1.5
    xs = np.linspace(points[:, 0].min() - pad, points[:, 0].max() + pad, 400)
    ys = np.linspace(points[:, 1].min() - pad, points[:, 1].max() + pad, 400)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    region = nearest_center(grid).reshape(gx.shape)
    ax.contourf(gx, gy, region, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                colors=CLASS_COLORS, alpha=0.12)

    for k in range(4):
        cloud = points[labels == k]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=16, alpha=0.75,
                   color=CLASS_COLORS[k], label=f"Classe {k}")
        ax.scatter(*MEANS[k], marker="X", s=180, color=CLASS_COLORS[k],
                   edgecolor="black", linewidth=1.2, zorder=5)

    ax.set_title("Figura 1 - Nuvens gaussianas (s = 1) com centros e regioes do centro mais proximo")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left", title="marca X = media teorica")
    return fig


def figure2(datasets):
    """Figura 2: as 4 escalas em subplots com os mesmos limites de eixo."""
    all_points = np.vstack([X for X, _ in datasets.values()])
    pad = 1.0
    xlim = (all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
    ylim = (all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), SCALES):
        X, y = datasets[s]
        for k in range(4):
            cloud = X[y == k]
            ax.scatter(cloud[:, 0], cloud[:, 1], s=12, alpha=0.65,
                       color=CLASS_COLORS[k], label=f"Classe {k}")
        ax.scatter(MEANS[:, 0], MEANS[:, 1], marker="X", s=120,
                   color="black", zorder=5)
        ax.set_title(f"s = {s}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Figura 2 - Mesmas 4 classes, espalhamento multiplicado por s (eixos compartilhados)")
    fig.tight_layout()
    return fig


def figure3(mixing):
    """Figura 3: taxa de mistura em funcao da escala s."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scales = list(mixing)
    values = [mixing[s] for s in scales]
    ax.plot(scales, values, marker="o", color="#1f77b4")
    for s, v in zip(scales, values):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.set_title("Figura 3 - Taxa de mistura x fator de escala s")
    ax.set_xlabel("fator de escala s")
    ax.set_ylabel("taxa de mistura (fracao de pontos)")
    ax.set_xticks(scales)
    ax.grid(alpha=0.3)
    return fig


def main():
    rng = np.random.default_rng(SEED)

    # Gera os 4 datasets de uma vez, em ordem crescente de s, reaproveitando o rng.
    datasets = {s: generate_clouds(rng, s) for s in SCALES}

    fig1 = figure1(*datasets[1.0])
    p1 = savefig(fig1, "figure1_gaussian_clouds.png")
    fig2 = figure2(datasets)
    p2 = savefig(fig2, "figure2_spread_scales.png")

    mixing = {s: mixing_rate(*datasets[s]) for s in SCALES}
    fig3 = figure3(mixing)
    p3 = savefig(fig3, "figure3_mixing_rate.png")

    ratios = separation_ratios()
    min_pair = min(ratios, key=ratios.get)
    min_value = ratios[min_pair]

    print("== Exercicio 1 ==")
    print(f"figuras salvas: {p1.name}, {p2.name}, {p3.name}")
    print("razao de separacao r_ij (s = 1):")
    for (i, j), r in ratios.items():
        print(f"  r_{i}{j} = {r:.3f}")
    print(f"menor r_ij: r_{min_pair[0]}{min_pair[1]} = {min_value:.3f} "
          f"-> em s = 2 vale {min_value / 2:.3f}")
    print("taxa de mistura por escala:")
    for s, v in mixing.items():
        print(f"  s = {s}: {v:.3f}")

    return {
        "separation_ratios": ratios,
        "min_ratio_pair": min_pair,
        "min_ratio_s1": min_value,
        "min_ratio_s2": min_value / 2,
        "mixing_rate": mixing,
    }


if __name__ == "__main__":
    main()
