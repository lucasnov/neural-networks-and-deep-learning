"""Roda os dois exercicios com um unico rng, regenera as seis figuras e
imprime o resumo final.

Uso (a partir da raiz do repositorio):

    python docs/exercises/perceptron/code/run_all.py
"""

import numpy as np

import exercise1_separable as ex1
import exercise2_overlap as ex2
from common import FIGURES_DIR, SEED


def main():
    rng = np.random.default_rng(SEED)   # o mesmo rng para toda a atividade
    r1 = ex1.main(rng)
    print()
    r2 = ex2.main(rng)

    run, alt, run2 = r1["run"], r1["run_alt"], r2["run"]
    pocket = run2["pocket"]
    fw = lambda w: f"[{w[0]:.4f}, {w[1]:.4f}]"

    print("\n" + "=" * 60)
    print("RESUMO DOS RESULTADOS")
    print("=" * 60)
    rows = [
        ("1", "Ex.1 w e b finais", f"w = {fw(run['w'])}, b = {run['b']:.4f}"),
        ("2", "Ex.1 epocas ate convergir", f"{run['epochs']}"),
        ("3", "Ex.1 acuracia final", f"{run['final_accuracy']:.4f}"),
        ("4", "Ex.1 epocas e acuracia, eta = 1.0",
         f"{alt['epochs']} epocas, {alt['final_accuracy']:.4f}"),
        ("5", "Ex.2 w e b finais", f"w = {fw(run2['w'])}, b = {run2['b']:.4f}"),
        ("6", "Ex.2 acuracia dos pesos finais", f"{run2['final_accuracy']:.4f}"),
        ("7", "Ex.2 acuracia do pocket",
         f"{pocket['accuracy']:.4f} (w = {fw(pocket['w'])}, b = {pocket['b']:.4f})"),
        ("8", "Ex.2 epoca do melhor do pocket", f"{pocket['epoch']}"),
    ]
    for num, item, value in rows:
        print(f"{num:>2} | {item:<36} | {value}")

    print("\nfiguras em:", FIGURES_DIR)


if __name__ == "__main__":
    main()
