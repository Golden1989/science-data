# Regressão linear: preço de imóveis

Pipeline completo de regressão linear para prever o preço de imóveis a partir de área, quartos, banheiros, andares,
ano de construção, localização, estado de conservação e garagem.

## Arquivos

- `train_model.py` -> script que roda tudo (carga, limpeza, features, modelos, avaliação e gráficos)
- `House Price Prediction Dataset.csv` -> dados de entrada (2.000 imóveis)
- `metrics.csv` -> R², MAE, MSE e RMSE de cada modelo no conjunto de teste
- `coefficients.csv` -> coeficientes do modelo escolhido (escala padronizada)
- `pred_vs_real.png` e `residuals.png` -> gráficos de previsto × real e dos resíduos
- `best_model.joblib` -> modelo treinado

## Como rodar

Na pasta `regressao_linear`:

```
pip install pandas numpy scikit-learn matplotlib joblib
python train_model.py
```

As saídas são gravadas na própria pasta.

## O que o script faz

1. **Limpeza:** remove duplicatas e linhas sem preço, padroniza o texto das categorias, imputa ausentes (mediana e
   moda) e descarta valores impossíveis (área ≤ 0, ano de construção fora de 1800–2026 etc.).
2. **Novas variáveis:** idade do imóvel (no lugar do ano de construção), razão banheiros/quartos e área por cômodo.
3. **Pré-processamento:** padronização das numéricas, `OrdinalEncoder` no estado de conservação
   (Poor < Fair < Good < Excellent) e One-Hot Encoding em localização e garagem.
4. **Modelos:** `LinearRegression`, `Ridge`, `Lasso` e `ElasticNet`, com hiperparâmetros escolhidos por
   `GridSearchCV` (5 folds) só no treino (80/20).
5. **Avaliação:** o melhor modelo pela validação cruzada é avaliado no teste e comparado com um baseline que sempre
   prevê a média.

## Resultados

| Modelo (teste) | R² | MAE | RMSE |
|---|---|---|---|
| Baseline (média) | −0,001 | 242.480 | 279.024 |
| LinearRegression | −0,012 | 244.010 | 280.531 |
| Ridge | −0,007 | 243.597 | 279.853 |
| Lasso (escolhido) | −0,005 | 243.481 | 279.665 |
| ElasticNet | −0,006 | 243.551 | 279.791 |

**Nenhum modelo supera o baseline.** Isso não é um erro do código: neste dataset o preço não tem relação com as
variáveis disponíveis. As correlações do preço com área, quartos, banheiros, andares e ano ficam entre −0,02 e
0,06, e o preço médio é praticamente o mesmo em todas as localizações. O Lasso escolhido pela validação cruzada
percebeu isso e zerou quase todos os coeficientes, ficando próximo de prever a média.

A conclusão do projeto é justamente essa: **sem sinal nos dados, nenhum modelo consegue prever**. Comparar com um
baseline é o que permite chegar a essa conclusão em vez de confiar num R² sem referência. O dataset parece ter
sido gerado artificialmente, com preços sorteados independentemente das características.

Para um exemplo com dados que têm relação real com o preço, veja `modelos_preditivos/` (aluguel) e
`previsao_precos_imoveis_bh/`.
