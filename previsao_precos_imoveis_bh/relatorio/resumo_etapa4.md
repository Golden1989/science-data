# Etapa 4 – Modelagem: resumo do modelo final

_Gerado automaticamente por `src/gerar_resumo.py` (chamado por `src/modelo_final.py`). Números do treino (n = 224) e do teste (n = 57), separados 80/20 com seed 42._

## 1. Especificação

**Modelo principal (linear, OLS):**

```
preco ~ area_m2 + area_m2×casa + casa + quartos + banheiros + vagas + regional + lancamento
```

**Modelo complementar (log-log, OLS + smearing de Duan):** mesmas variáveis, com `log(preco)` e `log(area_m2)`; usado para ler os efeitos em percentual.

- `casa`: 1 para casa/sobrado; **cobertura conta como apartamento** (a elasticidade da área da cobertura não diferiu da do apartamento, p = 0,50).
- `regional`: regional administrativa de BH (9 categorias, referência = **Oeste**), a partir de `regionais_bh.csv`.
- `lancamento`: 1 quando o link do anúncio é de `/imoveis-lancamentos/` (39 imóveis, todos apartamentos; 30 no treino). **Variável identificada depois da primeira avaliação no teste**; entrou porque melhorou R², RMSE e MAE na CV do treino (seção 2).
- Pontos influentes (Cook > 4/n) **mantidos**; ver robustez na seção 5.
- Erros-padrão robustos **HC3**, porque o teste de Breusch-Pagan indica heterocedasticidade (seção 4).
- Escolhas feitas por validação cruzada 5-fold × 3 repetições **só no treino**. Critério definido antes do teste: desempenho em R$ (R², RMSE, MAE). O teste foi avaliado duas vezes: na especificação original e, depois, com `lancamento`; as duas avaliações estão na seção 2.

Caminho até a especificação (scripts `analise_casas_cook.py` e `etapa4_itens3_4.py`): a interação área × casa entrou porque nas casas a área informada se comporta como área de terreno; a regional venceu o bairro com limiar de 5 ou 3 anúncios e o target encoding em R² e MAE na CV, com coeficientes mais estáveis.

## 2. Métricas

### Validação cruzada no treino (5-fold × 3)

| Modelo | Estimador | R² | RMSE | MAE | Erro mediano |
|---|---|---|---|---|---|
| linear | OLS | 0,674 ± 0,009 | R$ 358.607 | R$ 265.451 | R$ 190.457 |
| linear | Ridge | 0,672 ± 0,004 | R$ 359.633 | R$ 264.370 | R$ 189.889 |
| linear | Lasso | 0,666 ± 0,007 | R$ 363.095 | R$ 267.071 | R$ 196.395 |
| log-log | OLS | 0,589 ± 0,027 | R$ 402.288 | R$ 245.018 | R$ 150.388 |
| log-log | Ridge | 0,582 ± 0,027 | R$ 405.915 | R$ 245.644 | R$ 150.893 |
| log-log | Lasso | 0,533 ± 0,078 | R$ 427.841 | R$ 250.002 | R$ 151.165 |
| *referência* | Random Forest | 0,682 ± 0,007 | R$ 354.248 | R$ 244.192 | R$ 165.638 |

**OLS mantido.** No linear, a diferença de R² para o OLS foi de -0,002 (Ridge) e -0,008 (Lasso), com MAE de R$ 264.370 e R$ 267.071 contra R$ 265.451. Diferenças dessa ordem não justificam a troca: o desvio entre repetições (0,009) só mede a variação de embaralhar os mesmos dados e subestima a incerteza real. Mantido o OLS por parcimônia e interpretabilidade. No log-log, Ridge e Lasso foram piores que o OLS.

**Referência não linear:** um Random Forest com as mesmas variáveis e a mesma CV chega a R² 0,682 e MAE R$ 244.192, contra 0,674 e R$ 265.451 do linear: o modelo linear perde 0,008 de R² e R$ 21.259 de MAE, em troca de coeficientes interpretáveis. Ele é só referência e não substitui o modelo principal.

### Variável `lancamento` (identificada depois da primeira avaliação no teste)

Depois da primeira avaliação no teste, notamos que os links com `/imoveis-lancamentos/` têm preço por m² muito maior (mediana de R$ 17.628/m², contra R$ 7.365/m² nos demais). A variável foi testada com o mesmo protocolo (CV só no treino, OLS) e a regra fixada antes de rodar: entra se o linear melhorar R², RMSE e MAE.

| Modelo | Variáveis | R² (CV) | RMSE (CV) | MAE (CV) | Erro mediano (CV) |
|---|---|---|---|---|---|
| linear | atual | 0,584 | R$ 405.155 | R$ 284.986 | R$ 181.699 |
| linear | + lancamento | 0,674 | R$ 358.607 | R$ 265.451 | R$ 190.457 |
| log-log | atual | 0,357 | R$ 503.546 | R$ 294.193 | R$ 169.970 |
| log-log | + lancamento | 0,589 | R$ 402.288 | R$ 245.018 | R$ 150.388 |

O linear melhorou nas três métricas da regra (o erro mediano piorou de R$ 181.699 para R$ 190.457), e a variável foi adotada nos dois modelos.

### Teste

| Avaliação | Modelo | R² | RMSE | MAE | Erro mediano |
|---|---|---|---|---|---|
| 1ª (sem `lancamento`) | linear (principal) | 0,599 | R$ 349.131 | R$ 243.669 | R$ 184.067 |
| 1ª (sem `lancamento`) | log-log + Duan (complementar) | 0,671 | R$ 316.113 | R$ 219.193 | R$ 148.706 |
| 2ª (com `lancamento`) | linear (principal) | 0,658 | R$ 322.444 | R$ 238.265 | R$ 194.657 |
| 2ª (com `lancamento`) | log-log + Duan (complementar) | 0,782 | R$ 257.115 | R$ 197.789 | R$ 154.786 |

A 2ª avaliação **não é uma estimativa independente**: a variável foi encontrada depois de o teste já ter sido usado uma vez, e a melhora no teste deve ser lida com essa ressalva. A melhora na CV do treino, onde a decisão foi tomada, é a evidência principal.

**Linear × log-log.** Na CV do treino o resultado se divide: o linear vence em R² e RMSE (R² 0,674 vs. 0,589; RMSE R$ 358.607 vs. R$ 402.288), e o log-log vence em MAE (MAE R$ 245.018 vs. R$ 265.451). No teste, o log-log fica à frente nas três métricas (R² 0,782 vs. 0,658; MAE R$ 197.789 vs. R$ 238.265). O critério fixado antes do teste (R², RMSE e MAE em R$) não previa um resultado dividido; **mantivemos o linear como principal** porque ele vence nas duas métricas que penalizam erros grandes (R² e RMSE), e essa decisão não foi revista depois de ver o teste.

A desvantagem do log-log nessas duas métricas continua vindo de 1–2 casas grandes: uma casa de 405 m² no Comiteco é o maior erro em 3 de 3 repetições. Tirando os 2 maiores erros do log-log de cada repetição e comparando os dois modelos **nos mesmos imóveis**, o log-log passa à frente também em R² (0,728 vs. 0,665) e RMSE (R$ 317.430 vs. R$ 352.515). O teste (57 imóveis, maior área 365 m²) não tem casos desse tipo, o que explica a inversão.

Fator de Duan (treino): 1,040, ou seja, sem a correção o log-log subestimaria os preços em cerca de 4,0%.

## 3. Coeficientes interpretados

Sempre **mantidas as demais variáveis constantes**. Significância a 5% com erros-padrão HC3.

### Modelo principal (linear, em R$)

| Efeito | Estimativa | p-valor | Significativo? |
|---|---|---|---|
| Cada m² adicional – **apartamento/cobertura** | R$ 2.013 | 0,038 | sim |
| Cada m² adicional – **casa** | R$ 77 | 0,889 | **não** |
| Casa × apartamento, ambos com 255 m² (área mediana das casas) | R$ -248.336 | 0,096 | **não** |
| +1 banheiro | R$ 131.165 | 0,002 | sim |
| +1 vaga | R$ 164.282 | < 0,001 | sim |
| +1 quarto | R$ 70.406 | 0,159 | **não** |
| Barreiro vs. Oeste (n treino = 7) | R$ -130.250 | 0,313 | **não** |
| Centro-Sul vs. Oeste (n treino = 53) | R$ 399.119 | < 0,001 | sim |
| Leste vs. Oeste (n treino = 16) | R$ 115.530 | 0,241 | **não** |
| Nordeste vs. Oeste (n treino = 21) | R$ 8.471 | 0,912 | **não** |
| Noroeste vs. Oeste (n treino = 9) | R$ -186.524 | 0,150 | **não** |
| Norte vs. Oeste (n treino = 7) | R$ -248.526 | 0,038 | sim |
| Pampulha vs. Oeste (n treino = 28) | R$ -17.740 | 0,819 | **não** |
| Venda Nova vs. Oeste (n treino = 19) | R$ -60.384 | 0,460 | **não** |
| Lançamento vs. demais | R$ 597.901 | < 0,001 | sim |

- **Área:** no linear, o m² sai R$ 2.013 no apartamento (p = 0,038) e R$ 77 na casa (p = 0,889). A estimativa em R$/m² usada na interpretação vem do log-log (abaixo), que lida melhor com a heterocedasticidade.
- Um banheiro a mais: R$ 131.165; uma vaga a mais: R$ 164.282.
- Regional: o imóvel no **Centro-Sul** custa R$ 399.119 a mais que um idêntico na Oeste; no **Norte**, R$ 248.526 a menos.
- O coeficiente isolado de `casa` (R$ 245.476) é a diferença com área zero e não tem leitura própria; a comparação útil é a da linha "Casa × apartamento" acima.

### Modelo complementar (log-log, em %)

| Efeito | Estimativa | p-valor | Significativo? |
|---|---|---|---|
| +1% de área – **apartamento/cobertura** | +0,49% | < 0,001 | sim |
| +1% de área – **casa** | +0,06% | 0,646 | **não** |
| Casa × apartamento com 255 m² | -29,5% | < 0,001 | sim |
| +1 banheiro | 10,0% | 0,002 | sim |
| +1 vaga | 14,0% | < 0,001 | sim |
| +1 quarto | 2,0% | 0,680 | **não** |
| Barreiro vs. Oeste | -25,8% | 0,047 | sim |
| Centro-Sul vs. Oeste | 31,8% | < 0,001 | sim |
| Leste vs. Oeste | 13,1% | 0,170 | **não** |
| Nordeste vs. Oeste | -4,4% | 0,581 | **não** |
| Noroeste vs. Oeste | -26,5% | 0,006 | sim |
| Norte vs. Oeste | -30,4% | < 0,001 | sim |
| Pampulha vs. Oeste | -8,4% | 0,202 | **não** |
| Venda Nova vs. Oeste | -20,4% | 0,001 | sim |
| Lançamento vs. demais | 72,0% | < 0,001 | sim |

- **+1% de área → +0,49% no preço do apartamento e +0,06% no da casa.**
- **Cada m² adicional eleva o preço do apartamento em cerca de R$ 4.475** (IC 95%: R$ 2.934 a R$ 6.016), mantidas as demais constantes. Cálculo: elasticidade × preço mediano / área mediana dos apartamentos do treino (0,486 × R$ 810.000 / 88 m²), com IC pelo método delta tratando as medianas como fixas. Na casa, o mesmo cálculo dá R$ 252/m² (IC 95%: R$ -825 a R$ 1.330), não significativo. Para comparação, o coeficiente do modelo linear é R$ 2.013/m² (p = 0,04).
- **+1 banheiro → 10,0%; +1 vaga → 14,0%.**
- **Centro-Sul ≈ 32% vs. Oeste** para um imóvel com as mesmas características; sem `lancamento`, o prêmio estimado era de 58%, porque parte dos lançamentos (26 de 39) fica no Centro-Sul e o efeito deles era atribuído à regional.
- **Lançamento:** com as demais variáveis iguais, o anúncio de lançamento custa **72% a mais** no log-log (R$ 597.901 no linear). **Ressalva:** o coeficiente mistura o ágio do imóvel novo com um possível erro de medida. Nos lançamentos, o cartão mostra o preço "a partir de" da construtora (só 8% são múltiplos de R$ 10 mil, contra 73% nos demais) junto com a área e os quartos de uma unidade específica, que nem sempre é a mais barata; um lançamento chegou a ter área em faixa nos dados brutos (107 a 218 m²). Não dá para separar as duas coisas com estes dados.

### Coeficientes não significativos e por quê

- **Quartos** (p = 0,159 no linear e 0,680 no log-log): colinear com a área (correlação 0,69 no treino). Com a área fixa, um quarto a mais só divide o mesmo espaço em cômodos menores; o efeito próprio é pequeno e impreciso.
- **Casa × apartamento no linear** (R$ -248.336 com 255 m², p = 0,096; o nível isolado de `casa` tem p = 0,054 e só tem leitura junto da área): poucas casas no treino (n = 44) e muita dispersão de preço em R$. No log-log a diferença é significativa (-29,5%, p < 0,001).
- **Área da casa no log-log** (elasticidade 0,06, p = 0,646): o preço da casa quase não varia com a área informada, coerente com a área de terreno (seção 6).
- **Regionais** sem diferença significativa da Oeste: Barreiro, Leste, Nordeste, Noroeste, Pampulha, Venda Nova no linear; Leste, Nordeste, Pampulha no log-log. São regionais com poucos imóveis no treino (Barreiro n = 7, Noroeste n = 9, Leste n = 16, Venda Nova n = 19, Nordeste n = 21, Pampulha n = 28) ou com preço próximo ao da referência.

## 4. Diagnósticos

| Modelo | Breusch-Pagan LM | p-valor | Conclusão |
|---|---|---|---|
| linear | 45,6 | < 0,001 | heterocedástico (rejeita H0 a 5%) |
| log-log | 25,1 | 0,049 | heterocedástico (rejeita H0 a 5%) |

VIF (treino), com e sem centrar a área:

| Variável | VIF linear | linear, área centrada | VIF log-log | log-log, área centrada |
|---|---|---|---|---|
| area_m2 | 9,98 | 9,98 | 4,77 | 4,77 |
| area_x_casa | 14,13 | 6,30 | 156,08 | 4,70 |
| casa | 5,21 | 2,16 | 144,97 | 4,11 |
| quartos | 2,62 | 2,62 | 2,88 | 2,88 |
| banheiros | 2,30 | 2,30 | 2,23 | 2,23 |
| vagas | 2,14 | 2,14 | 2,15 | 2,15 |
| reg_Barreiro | 1,12 | 1,12 | 1,12 | 1,12 |
| reg_Centro-Sul | 1,67 | 1,67 | 1,66 | 1,66 |
| reg_Leste | 1,21 | 1,21 | 1,21 | 1,21 |
| reg_Nordeste | 1,25 | 1,25 | 1,25 | 1,25 |
| reg_Noroeste | 1,16 | 1,16 | 1,16 | 1,16 |
| reg_Norte | 1,16 | 1,16 | 1,16 | 1,16 |
| reg_Pampulha | 1,34 | 1,34 | 1,34 | 1,34 |
| reg_Venda Nova | 1,26 | 1,26 | 1,26 | 1,26 |
| lancamento | 1,33 | 1,33 | 1,34 | 1,34 |

O VIF de `area_m2` no linear (9,98) não cai ao centrar a área, mas também **não vem de quartos, banheiros e vagas**: sem a interação ele é 2,55, e na forma equivalente com a área separada por tipo (`area_apto`, `area_casa`, mesmo ajuste) o VIF da área do apartamento é 3,33. O valor alto é estrutural: as casas concentram as maiores áreas, então `area_m2` e `area_x_casa` andam juntas. Entre os apartamentos, a área tem colinearidade moderada com quartos, banheiros e vagas (correlação de 0,54 a 0,72). Regionais, banheiros, vagas, quartos e `lancamento` ficam abaixo de 3.

Gráficos: `graficos/5_teste_real_x_previsto.png`, `graficos/6_residuos_x_previsto.png`, `graficos/7_qqplot_residuos.png`.

## 5. Robustez: pontos influentes

23 imóveis do treino têm distância de Cook > 4/n (0,0179) no modelo linear. Eles foram **mantidos**. Reajustando sem eles:

| Coeficiente | Com todos | Sem influentes |
|---|---|---|
| area_m2 (linear) | R$ 2.013 | R$ 3.096 |
| area_x_casa (linear) | R$ -1.937 | R$ -3.246 |
| banheiros (linear) | R$ 131.165 | R$ 39.086 |
| vagas (linear) | R$ 164.282 | R$ 172.580 |
| reg_Centro-Sul (linear) | R$ 399.119 | R$ 355.075 |
| log_area (log-log) | 0,486 | 0,427 |
| log_area_x_casa (log-log) | -0,426 | -0,399 |
| banheiros (log-log) | 0,095 | 0,075 |
| vagas (log-log) | 0,131 | 0,180 |
| reg_Centro-Sul (log-log) | 0,276 | 0,332 |

Na CV do modelo linear (avaliada em todos os imóveis do treino), treinar sem os influentes leva o R² de 0,674 para 0,678 e o MAE de R$ 265.451 para R$ 252.065. O coeficiente mais sensível a esses pontos é `banheiros` no linear (-70%); os sinais dos coeficientes da tabela não mudam. Os influentes ficam na base (decisão registrada) e a análise entra como robustez.

**Banheiros no linear depende de poucos imóveis caros.** Os 23 influentes têm preço mediano de R$ 1.678.000 (contra R$ 795.000 nos demais; 12 deles acima do P75 do treino) e 3,7 banheiros em média (contra 2,2). Sem eles, o efeito de +1 banheiro no linear cai 70% (R$ 131.165 → R$ 39.086, p = 0,171), enquanto no log-log a variação é menor (21% no coeficiente: 10,0% → 7,8% por banheiro, p = 0,013). Em R$, os poucos imóveis caros com muitos banheiros pesam muito mais na reta do que na escala log; o efeito em % do log-log é a leitura mais estável.

## 6. Limitações

- **Área de terreno nas casas:** casas com área ≥ 300 m² e R$/m² abaixo do P10 das casas provavelmente informam a área do lote (padrão de BH = 360 m²), não a área construída. A interação área × casa absorve parte disso (o m² da casa vale R$ 77 contra R$ 2.013 no apartamento), mas a variável continua medindo coisas diferentes em cada tipo.
- **Viés do Buritis:** 16,7% da amostra é de um único bairro (Oeste), o que pesa no efeito da regional de referência e reflete a ordenação do site, não o mercado.
- **Mapeamento de regionais aproximado:** feito manualmente; 5 bairros de divisa ou com nome repetido (19 imóveis) estão marcados como duvidosos. Trocá-los para a regional alternativa quase não muda as métricas (R² log 0,637 → 0,651 na CV, teste feito antes da inclusão de `lancamento`).
- **`lancamento` incompleta e posterior ao teste:** 92 imóveis vêm de cartões agrupados ("Ver os N anúncios deste imóvel"), sem link, e ficaram como não lançamento por falta de informação (mediana de R$ 7.969/m², perto dos não lançamentos). A variável foi encontrada depois da primeira avaliação no teste, então a 2ª métrica de teste não é independente.
- **Previsões abaixo do mínimo:** o modelo linear não respeita a positividade do preço. No teste, 4 de 57 previsões ficaram abaixo do menor preço do treino (R$ 240.000); na CV, 9,3 por repetição. O log-log não tem esse problema.
- **1–2 casas grandes derrubam o log-log em R$:** o `exp()` amplifica erros nos imóveis grandes, sobretudo uma casa de 405 m² no Comiteco (maior erro em 3 de 3 repetições). Na CV, sem os 2 maiores erros de cada repetição, o R² em R$ do log-log passaria de 0,59 para 0,73. Por isso o log-log é usado para interpretar efeitos em %, e o linear para prever em R$.
- **Amostra pequena e de anúncios:** 281 imóveis, preço pedido (não de venda), coletados das 10 primeiras páginas do VivaReal em uma data; não é uma amostra aleatória do mercado de BH.
