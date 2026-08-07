"""
Modelo de Credit Score — Regressão Logística
Dataset: German Credit Data

Prevê se um proponente de crédito é bom ou mau pagador,
usando o dataset clássico German Credit. O foco do projeto
não é só acurácia, mas robustez da métrica (via cross-validation)
e interpretabilidade — critério relevante no mercado de crédito real,
onde modelos precisam ser explicáveis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)
from scipy.stats import ks_2samp

# ──────────────────────────────────────────────
#  CONFIGURAÇÕES GERAIS
# ──────────────────────────────────────────────
SEED = 42  # fixa em TODO lugar que envolve aleatoriedade — split, modelo, cross-validation
N_FOLDS = 5

# ──────────────────────────────────────────────
#  1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────
df = pd.read_csv("german_credit_data.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.dropna()
df["Risk"] = df["Risk"].map({"good": 1, "bad": 0})
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Risk", axis=1)
y = df_encoded["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)

# Regressão Logística é sensível à escala das variáveis — sem isso,
# colunas como "Credit amount" (milhares) dominam colunas binárias (0/1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ──────────────────────────────────────────────
#  2. CROSS-VALIDATION — a métrica que realmente importa
# ──────────────────────────────────────────────
# Um único split (train/test) dá um KS que varia por sorte da amostra.
# Cross-validation estratificada mostra a faixa real de desempenho do modelo.
print("=== VALIDAÇÃO CRUZADA (5 folds) ===")
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
modelo_cv = LogisticRegression(max_iter=2000, random_state=SEED)

auc_scores = cross_val_score(modelo_cv, scaler.fit_transform(X), y, cv=cv, scoring="roc_auc")
print(f"AUC por fold: {np.round(auc_scores, 4)}")
print(f"AUC médio: {auc_scores.mean():.4f}  (desvio padrão: {auc_scores.std():.4f})\n")

# ──────────────────────────────────────────────
#  3. TREINO FINAL (no split de teste, para os gráficos e métricas de negócio)
# ──────────────────────────────────────────────
print("=== TREINAMENTO FINAL ===")
print(f"O modelo está aprendendo com {len(X_train)} clientes...\n")
modelo = LogisticRegression(max_iter=2000, random_state=SEED)
modelo.fit(X_train_scaled, y_train)
print("Treinamento concluído!\n")

print("=== TESTE ===")
previsoes = modelo.predict(X_test_scaled)
scores = modelo.predict_proba(X_test_scaled)[:, 1]

print("Acurácia:", accuracy_score(y_test, previsoes))
print("\nRelatório:")
print(classification_report(y_test, previsoes))

# ──────────────────────────────────────────────
#  4. KS — métrica principal em modelos de crédito
# ──────────────────────────────────────────────
prob_bons = scores[y_test == 1]
prob_maus = scores[y_test == 0]
ks_estatistica, _ = ks_2samp(prob_bons, prob_maus)

print("=== KS DO MODELO (split de teste) ===")
print(f"KS: {ks_estatistica:.4f} ({ks_estatistica*100:.1f}%)\n")

# ──────────────────────────────────────────────
#  5. INTERPRETABILIDADE — coeficientes do modelo
# ──────────────────────────────────────────────
# Diferente de Random Forest, aqui dá pra explicar CADA decisão:
# coeficiente positivo → aumenta chance de "bom pagador"
# coeficiente negativo → aumenta chance de "mau pagador"
coeficientes = pd.DataFrame({
    "variavel": X.columns,
    "coeficiente": modelo.coef_[0]
}).sort_values("coeficiente", ascending=False)

print("=== COEFICIENTES DO MODELO (interpretabilidade) ===")
print(coeficientes.to_string(index=False))

# ──────────────────────────────────────────────
#  6. GRÁFICOS ESSENCIAIS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Regressão Logística — Credit Score (German Credit)", fontsize=13, fontweight="bold")

# Matriz de confusão
cm = confusion_matrix(y_test, previsoes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Mau Pagador", "Bom Pagador"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Matriz de Confusão")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, scores)
auc = roc_auc_score(y_test, scores)
axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Modelo aleatório")
axes[1].set_xlabel("Taxa de Falsos Positivos")
axes[1].set_ylabel("Taxa de Verdadeiros Positivos")
axes[1].set_title("Curva ROC")
axes[1].legend()

plt.tight_layout()
plt.savefig("resultados_modelo.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────
#  7. GRÁFICO DE APOIO — distribuição de risco por faixa de valor de crédito
# ──────────────────────────────────────────────
# Este é o único gráfico "de estrato" mantido no projeto — mostra que o
# modelo foi validado não só por métrica agregada, mas por comportamento
# em subgrupos, prática comum em times de risco de crédito.
df_test = df.loc[X_test.index].copy()
df_test["Score"] = scores

df_test["Faixa_Credito"] = pd.qcut(df_test["Credit amount"], q=5, duplicates="drop")
resumo = df_test.groupby("Faixa_Credito", observed=True)["Risk"].mean().reset_index()

plt.figure(figsize=(9, 5))
plt.bar(resumo["Faixa_Credito"].astype(str), resumo["Risk"], color="steelblue")
plt.xlabel("Faixa de Valor de Crédito")
plt.ylabel("Proporção de Bons Pagadores")
plt.title("Taxa de Bons Pagadores por Faixa de Valor de Crédito")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("distribuicao_por_faixa_credito.png", dpi=150)
plt.show()

print("\n✅ Concluído! Gráficos salvos: resultados_modelo.png e distribuicao_por_faixa_credito.png")
