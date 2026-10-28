"""Roda os quatro exercicios, regenera as seis figuras e imprime o resumo final.

Uso (a partir da raiz do repositorio):

    python docs/exercises/mlp/code/run_all.py
"""

import exercise1_check as ex1
import exercise2_binary as ex2
import exercise3_multiclass as ex3
import exercise4_depth as ex4
from common import FIGURES_DIR


def main():
    r1 = ex1.main()
    print()
    r2 = ex2.main()
    print()
    r3 = ex3.main()
    print()
    r4 = ex4.main()

    f = lambda v: f"{v:.4f}"
    print("\n" + "=" * 60)
    print("RESUMO DOS RESULTADOS")
    print("=" * 60)
    rows = [
        ("1", "Ex.1 z1 e h1", f"z1 = {[f(v) for v in r1['z1']]} | h1 = {[f(v) for v in r1['h1']]}"),
        ("2", "Ex.1 u2, y_hat e L", f"{f(r1['u2'])} | {f(r1['y_hat'])} | {f(r1['loss'])}"),
        ("3", "Ex.1 parametros atualizados",
         f"W1 = {[[f(v) for v in row] for row in r1['new_W1']]} b1 = {[f(v) for v in r1['new_b1']]} "
         f"W2 = {[f(v) for v in r1['new_W2']]} b2 = {f(r1['new_b2'])}"),
        ("4", "Ex.2 linha de base linear", f(r2["baseline_acc"])),
        ("5", "Ex.2 arquitetura e acuracia", f"{ex2.LAYER_SIZES} | {f(r2['test_acc'])}"),
        ("6", "Ex.3 arquitetura e acuracia", f"{ex3.LAYER_SIZES} | {f(r3['test_acc'])}"),
        ("7", "Ex.3 par mais confundido", f"classes {r3['pair']} ({r3['pair_errors']} erros)"),
        ("8", "Ex.4 arquitetura e acuracia (semente 42)",
         f"{ex4.DEEP_SIZES} | {f(r4['deep_seed42_test_acc'])}"),
        ("9", "Ex.4 - Ex.3, media das sementes", f"{r4['diff']:+.4f}"),
    ]
    for num, item, value in rows:
        print(f"{num:>2} | {item:<42} | {value}")

    print("\nfiguras em:", FIGURES_DIR)


if __name__ == "__main__":
    main()
