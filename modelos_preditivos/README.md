# Modelos preditivos: aluguel e câncer de mama

Um único script com dois exercícios de aprendizado de máquina com dados tabulares:

1. **Regressão linear**: prever o valor do aluguel (`houses_to_rent.csv`).
2. **Regressão logística**: classificar tumores como benignos ou malignos (`BreastCancer.csv`).

## Arquivos

| Arquivo | O que é |
|---|---|
| `models_pipeline.py` | Script que roda as duas tarefas |
| `houses_to_rent.csv`, `BreastCancer.csv` | Dados de entrada |
| `best_rent_model.joblib`, `metrics_aluguel.csv`, `coefficients_aluguel.csv` | Saídas da tarefa 1 |
| `best_cancer_model.joblib`, `metrics_cancer.csv` | Saídas da tarefa 2 |
| `requirements.txt` | Bibliotecas e versões testadas |

## Como rodar

Na pasta `modelos_preditivos`:

```
pip install -r requirements.txt
python models_pipeline.py                 # rodada normal (gera os modelos e as métricas)
python models_pipeline.py onehot          # rodada normal, com a alternativa do tratamento de 'floor'
python models_pipeline.py estabilidade    # repete os experimentos com 10 sementes (não grava arquivos)
```

A rodada `estabilidade` leva alguns minutos, porque treina os modelos 10 vezes.

O script procura os CSVs na própria pasta, e se não encontrar, na pasta acima. Todas as saídas são salvas na pasta do script.

## Resultados

### Tarefa 1: aluguel
O modelo escolhido é a **regressão polinomial de grau 2 com Ridge**, com log das variáveis de área, condomínio e IPTU. A escolha é feita pela validação cruzada de 5 folds, calculada só com os dados de treino.

| Métrica (teste) | Valor |
|---|---|
| R² | 0,570 |
| MAE | R$ 1.512 |
| RMSE | R$ 2.244 |

Para comparação, a regressão linear simples (sem log e sem polinômio) chega a R² 0,501 e RMSE R$ 2.416.

### Tarefa 2: câncer de mama
Regressão logística com o parâmetro C escolhido por validação cruzada estratificada.

| Métrica (teste) | Valor |
|---|---|
| Acurácia | 0,957 |
| Precisão | 0,938 |
| Recall | 0,938 |
| F1 | 0,938 |
| AUC-ROC | 0,995 |

Matriz de confusão no teste: 89 benignos corretos, 45 malignos corretos, 3 falsos positivos e 3 falsos negativos.

## Decisões importantes

- **Vazamento de dados removido:** `total` (soma do aluguel com as taxas) e `fire insurance` (cerca de 1,3% do aluguel, correlação de 0,986) são calculadas a partir do próprio alvo. Usá-las dava um R² de 0,96, que não é real.
- **Tratamento de `floor`:** o valor `-` aparece em 1.555 casas, provavelmente as que não têm andar. Não tratamos `-` como um andar qualquer. Veja a opção abaixo.
- **Log só nas explicativas:** o log do aluguel piorou o modelo, então o alvo continua em reais.
- **Códigos de cidade:** `city` é um código numérico sem significado documentado, então é tratado como categoria.
- **Rótulos do câncer:** o arquivo não diz o que é 0 e 1. Assumimos 0 = benigno e 1 = maligno pela contagem (458 e 241).
- **Imputação dentro do Pipeline:** as medianas são calculadas só com os dados de treino de cada rodada, para evitar vazamento.

## Opção de `floor`

O comentário no topo de `models_pipeline.py` explica as duas opções. Para trocar, sem editar o código:

```
python models_pipeline.py          # opção padrão: flag
python models_pipeline.py onehot   # alternativa: One-Hot por andar
```

- `flag` (padrão): `floor` continua numérico, e um indicador `floor_missing` marca as casas com `-`.
- `onehot`: cada andar vira uma coluna, e `-` é uma categoria própria (a referência do modelo).

Os dois resultados ficaram praticamente iguais no teste (R² 0,570 nos dois). A `flag` foi escolhida por ter menos colunas e por ser melhor na validação cruzada.

## Limitações

- O modelo de aluguel explica cerca de 57% da variação do preço. O bairro não está no dataset, e é um dos fatores mais importantes.
- Os números da tabela acima são da semente 42. Repetindo com 10 sementes diferentes:
  - **Aluguel:** o polinômio com Ridge foi escolhido em todas e teve R² 0,589 ± 0,028. A regressão linear base é instável (em duas sementes o RMSE passou de R$ 9.000) e não deve ser comparada só pela semente 42.
  - **Câncer:** acurácia 0,957 ± 0,014, recall 0,938 ± 0,017 e AUC 0,993 ± 0,004.
- O conjunto de teste do câncer tem 140 casos, então cada erro pesa bastante.
- Os coeficientes do aluguel estão em escala padronizada e com termos polinomiais, então não se leem como "R$ por unidade".
