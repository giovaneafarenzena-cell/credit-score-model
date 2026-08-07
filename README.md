# Credit Score Model

Modelo de classificação de risco de crédito (bom pagador vs. mau pagador), treinado sobre o dataset clássico **German Credit Data**. O foco do projeto não é só métrica de desempenho, mas **interpretabilidade** e **robustez estatística** — dois critérios centrais em modelagem de crédito real.

## Por que Regressão Logística

Modelos baseados em árvore (Random Forest, XGBoost) costumam ter desempenho bruto melhor em dados tabulares, mas funcionam como caixa-preta. Em crédito, times de risco (e reguladores) precisam justificar **por que** um crédito foi negado — e Regressão Logística permite isso diretamente, através dos coeficientes de cada variável.

## Metodologia

- **Pré-processamento:** variável alvo convertida com `map`, variáveis categóricas com `get_dummies(drop_first=True)`, variáveis numéricas padronizadas com `StandardScaler` (fit apenas no treino, para evitar vazamento de dados)
- **Split:** 85% treino / 15% teste, estratificado pela variável alvo, seed fixa (`random_state=42`) para reprodutibilidade
- **Validação:** 5-fold cross-validation estratificada, além do split de teste isolado

## Resultados

**Validação cruzada (5 folds), AUC:**

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Média | Desvio padrão |
|--------|--------|--------|--------|--------|-------|----------------|
| 0.732  | 0.684  | 0.658  | 0.557  | 0.781  | **0.682** | 0.075 |

**Split de teste isolado:**

| Métrica | Valor |
|---|---|
| Acurácia | 64,6% |
| AUC | 0,651 |
| KS | 29,9% |
| Recall (bom pagador) | 0,77 |
| Recall (mau pagador) | 0,49 |

## Leitura honesta dos resultados

O modelo é um **baseline razoável, não um modelo forte**. Dois pontos relevantes:

1. **Instabilidade entre folds** — o AUC varia de 0,56 a 0,78 dependendo da amostra, um desvio padrão de 0,075. Isso indica que o modelo, na forma atual, é sensível à composição dos dados de treino — não é um resultado robusto o suficiente pra produção.
2. **Recall fraco para maus pagadores (0,49)** — o modelo erra quase metade dos clientes que de fato não pagariam, classificando-os como bons pagadores. Em um cenário real de crédito, esse é o erro mais custoso (aprovar quem não paga), e é a principal limitação a atacar.

## Interpretabilidade — coeficientes do modelo

Coeficiente positivo aumenta a chance prevista de "bom pagador"; negativo, de "mau pagador".

| Variável | Coeficiente | Leitura |
|---|---|---|
| `Duration` | -0.647 | Prazos mais longos de empréstimo → maior risco |
| `Purpose_car` | -0.213 | Crédito para carro associado a maior risco |
| `Purpose_education` | -0.203 | Crédito para educação associado a maior risco |
| `Saving accounts_rich` | +0.307 | Poupança alta → menor risco |
| `Checking account_rich` | +0.300 | Conta corrente alta → menor risco |

A relação entre `Duration` e risco é a mais forte do modelo e é consistente com a intuição de crédito: quanto mais longo o compromisso financeiro, maior a incerteza sobre a capacidade de pagamento.

## Próximos passos

- Testar `class_weight="balanced"` para melhorar o recall de maus pagadores
- Ajustar o hiperparâmetro de regularização (`C`) via grid search
- Investigar engenharia de features (ex: transformação de `Duration` e `Credit amount`)
- Comparar com modelos baseados em árvore como referência de teto de desempenho, mantendo a Regressão Logística como modelo de produção pela interpretabilidade

## Como rodar

```bash
pip install pandas numpy matplotlib scikit-learn scipy
python modelo.py
```

O arquivo `german_credit_data.csv` já está incluído neste repositório (dataset público, originalmente disponibilizado pela [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)).
