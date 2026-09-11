"""Exercicio 3 - Preparando dados reais (Spaceship Titanic) para uma rede com tanh.

Faz a analise descritiva, separa treino/teste antes de qualquer transformacao,
imputa, codifica categorias, cria TotalSpend, aplica log1p nos gastos e escala
para [-1, 1]. Salva a Figura 6 e imprime as checagens finais.

Uso (a partir da raiz do repositorio):

    python docs/exercises/data/code/exercise3_preprocessing.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from common import SEED, savefig, load_spaceship_titanic

DROP_COLS = ["PassengerId", "Cabin", "Name"]
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
# Colunas numericas apos a engenharia de features (Age + 5 gastos + TotalSpend).
NUMERIC_COLS = ["Age"] + SPEND_COLS + ["TotalSpend"]


def _make_onehot():
    """OneHotEncoder denso, tolerante a versao do scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def describe_data(df):
    """Item A: balanceamento, tipos de feature, faltantes e estatisticas de gasto."""
    n_true = int((df["Transported"]).sum())
    n_false = int((~df["Transported"]).sum())
    total = len(df)

    missing = pd.DataFrame({
        "faltantes": df.isna().sum(),
        "percentual": (df.isna().mean() * 100).round(2),
    }).sort_values("percentual", ascending=False)

    spend_stats = df[SPEND_COLS].agg(["mean", "median", "max"]).T

    print("== Exercicio 3 - A ==")
    print(f"linhas: {total} | colunas: {df.shape[1]}")
    print(f"Transported=True : {n_true} ({n_true / total:.4f})")
    print(f"Transported=False: {n_false} ({n_false / total:.4f})")
    print("\nvalores faltantes por coluna:")
    print(missing.to_string())
    print("\ngastos (dataset bruto completo):")
    print(spend_stats.round(2).to_string())

    return {
        "n_rows": total,
        "n_cols": df.shape[1],
        "positive_ratio": n_true / total,
        "missing": missing,
        "spend_stats": spend_stats,
    }


def split(df):
    """Item B: split 80/20 estratificado, ANTES de qualquer estatistica."""
    X = df.drop(columns=["Transported"])
    y = df["Transported"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


def preprocess(X_train, X_test):
    """Item C: imputacao, TotalSpend, log1p e escalonamento - tudo ajustado no treino."""
    X_train = X_train.drop(columns=DROP_COLS).copy()
    X_test = X_test.drop(columns=DROP_COLS).copy()

    base_numeric = ["Age"] + SPEND_COLS

    # 1) Imputacao numerica (mediana do treino).
    num_imputer = SimpleImputer(strategy="median")
    X_train[base_numeric] = num_imputer.fit_transform(X_train[base_numeric])
    X_test[base_numeric] = num_imputer.transform(X_test[base_numeric])

    # 2) Imputacao categorica (categoria mais frequente no treino).
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[CATEGORICAL_COLS] = cat_imputer.fit_transform(X_train[CATEGORICAL_COLS])
    X_test[CATEGORICAL_COLS] = cat_imputer.transform(X_test[CATEGORICAL_COLS])

    # 3) TotalSpend a partir dos gastos JA imputados (evita depender do skipna do pandas).
    X_train["TotalSpend"] = X_train[SPEND_COLS].sum(axis=1)
    X_test["TotalSpend"] = X_test[SPEND_COLS].sum(axis=1)

    # 4) Cauda pesada: log1p nos 5 gastos e no TotalSpend (mesma natureza, mesma escala).
    log_cols = SPEND_COLS + ["TotalSpend"]
    X_train[log_cols] = np.log1p(X_train[log_cols])
    X_test[log_cols] = np.log1p(X_test[log_cols])

    # 5) Escalonamento para [-1, 1] (faixa de saida da tanh), so nas colunas numericas.
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    # 6) One-hot nas categorias (categoria nao vista no teste -> tudo zero).
    encoder = _make_onehot()
    ohe_train = encoder.fit_transform(X_train[CATEGORICAL_COLS])
    ohe_test = encoder.transform(X_test[CATEGORICAL_COLS])
    ohe_names = list(encoder.get_feature_names_out(CATEGORICAL_COLS))

    feature_names = NUMERIC_COLS + ohe_names
    train_matrix = np.hstack([X_train[NUMERIC_COLS].to_numpy(), ohe_train])
    test_matrix = np.hstack([X_test[NUMERIC_COLS].to_numpy(), ohe_test])

    return train_matrix, test_matrix, feature_names


def figure6(X_train_raw):
    """Figura 6: FoodCourt no treino, antes e depois de log1p."""
    food_raw = X_train_raw["FoodCourt"].fillna(X_train_raw["FoodCourt"].median())
    food_log = np.log1p(food_raw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(food_raw, bins=50, color="#1f77b4")
    axes[0].set_title("FoodCourt - original")
    axes[0].set_xlabel("gasto em FoodCourt")
    axes[0].set_ylabel("contagem (treino)")

    axes[1].hist(food_log, bins=50, color="#2ca02c")
    axes[1].set_title("FoodCourt - apos log1p (antes do escalonamento)")
    axes[1].set_xlabel("log(1 + gasto em FoodCourt)")
    axes[1].set_ylabel("contagem (treino)")

    fig.suptitle("Figura 6 - Efeito de log1p sobre uma feature de cauda pesada")
    fig.tight_layout()
    return fig


def main():
    df = load_spaceship_titanic()
    desc = describe_data(df)

    X_train, X_test, y_train, y_test = split(df)
    food_mean = float(X_train["FoodCourt"].mean())
    food_median = float(X_train["FoodCourt"].median())

    p6 = savefig(figure6(X_train), "figure6_foodcourt_transform.png")

    train_matrix, test_matrix, feature_names = preprocess(X_train, X_test)

    nan_train = int(np.isnan(train_matrix).sum())
    nan_test = int(np.isnan(test_matrix).sum())
    assert nan_train == 0 and nan_test == 0, "sobrou NaN apos o pre-processamento"
    assert train_matrix.shape[1] == test_matrix.shape[1], "treino e teste com nº de colunas diferente"
    assert train_matrix.shape[0] == len(y_train) and test_matrix.shape[0] == len(y_test)

    tr_min, tr_max = float(train_matrix.min()), float(train_matrix.max())
    te_min, te_max = float(test_matrix.min()), float(test_matrix.max())

    print("\n== Exercicio 3 - B/C/D ==")
    print(f"figura salva: {p6.name}")
    print(f"FoodCourt no treino (bruto): media = {food_mean:.2f} | mediana = {food_median:.2f}")
    print(f"split: treino {train_matrix.shape[0]} linhas | teste {test_matrix.shape[0]} linhas")
    print(f"NaN restante: treino = {nan_train} | teste = {nan_test}")
    print(f"shape final treino: {train_matrix.shape} | teste: {test_matrix.shape}")
    print(f"n features: {len(feature_names)}")
    print(f"faixa treino: [{tr_min:.3f}, {tr_max:.3f}] | faixa teste: [{te_min:.3f}, {te_max:.3f}]")
    print(f"features: {feature_names}")

    return {
        "n_rows": desc["n_rows"],
        "positive_ratio": desc["positive_ratio"],
        "missing": desc["missing"],
        "spend_stats": desc["spend_stats"],
        "food_mean_train": food_mean,
        "food_median_train": food_median,
        "train_shape": train_matrix.shape,
        "test_shape": test_matrix.shape,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "nan_train": nan_train,
        "nan_test": nan_test,
        "train_range": (tr_min, tr_max),
        "test_range": (te_min, te_max),
    }


if __name__ == "__main__":
    main()
