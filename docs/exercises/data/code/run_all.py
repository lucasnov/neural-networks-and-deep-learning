"""Roda os tres exercicios, regenera as seis figuras e imprime o resumo final.

Uso (a partir da raiz do repositorio):

    python docs/exercises/data/code/run_all.py
"""

import sys

import exercise1_point_clouds as ex1
import exercise2_geometry_5d as ex2
import exercise3_preprocessing as ex3
from common import FIGURES_DIR


def main():
    try:
        r1 = ex1.main()
        print()
        r2 = ex2.main()
        print()
        r3 = ex3.main()
    except FileNotFoundError as err:
        print(f"\nERRO: {err}", file=sys.stderr)
        sys.exit(1)

    pair = r1["min_ratio_pair"]
    tr = r3["train_range"]
    te = r3["test_range"]

    print("\n" + "=" * 60)
    print("RESUMO DOS RESULTADOS")
    print("=" * 60)
    rows = [
        ("1", "Taxa de mistura em s = 0.5", f"{r1['mixing_rate'][0.5]:.3f}"),
        ("2", "Taxa de mistura em s = 1.0", f"{r1['mixing_rate'][1.0]:.3f}"),
        ("3", "Taxa de mistura em s = 2.0", f"{r1['mixing_rate'][2.0]:.3f}"),
        ("4", "Taxa de mistura em s = 4.0", f"{r1['mixing_rate'][4.0]:.3f}"),
        ("5", "Menor r_ij em s = 1.0 (e o par)",
         f"{r1['min_ratio_s1']:.3f} - par (Classe {pair[0]}, Classe {pair[1]})"),
        ("6", "Distancia entre centros - Dataset I", f"{r2['dist_centers_i']:.3f}"),
        ("7", "Distancia entre centros - Dataset II", f"{r2['dist_centers_ii']:.3f}"),
        ("8", "Variancia explicada PC1+PC2 - Dataset I", f"{r2['evr_i'][2]:.3f}"),
        ("9", "Variancia explicada PC1+PC2 - Dataset II", f"{r2['evr_ii'][2]:.3f}"),
        ("10", "Proporcao da classe positiva em Transported", f"{r3['positive_ratio']:.4f}"),
        ("11", "Media e mediana de FoodCourt no treino (bruto)",
         f"media {r3['food_mean_train']:.2f} | mediana {r3['food_median_train']:.2f}"),
        ("12", "Shape final da matriz de features de treino", f"{tuple(r3['train_shape'])}"),
        ("13", "Min/max apos escalonamento",
         f"treino [{tr[0]:.3f}, {tr[1]:.3f}] | teste [{te[0]:.3f}, {te[1]:.3f}]"),
    ]
    for num, item, value in rows:
        print(f"{num:>2} | {item:<48} | {value}")

    print("\nfiguras em:", FIGURES_DIR)


if __name__ == "__main__":
    main()
