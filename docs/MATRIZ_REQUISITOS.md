# Matriz de rastreabilidade dos requisitos (FIAP — Tech Challenge Fase 4)

Fonte de verdade: o PDF oficial "MLET – Tech Challenge – Fase 4". Status: ✅ atendido · 🟡 parcialmente atendido · ⏳ pendente (externo).

## 1. Requisitos obrigatórios do repositório

| # | Requisito oficial | Implementação | Arquivo responsável | Validação | Evidência | Status |
|---|---|---|---|---|---|---|
| R1 | Pipeline de validação de dados (*data contracts*) configurado | 3 contratos Pandera (treino, inferência, outcome) e gate com quarentena | `src/validation/contracts.py`, `src/validation/gate.py` | `tests/unit/test_contracts.py` (uma regra por teste), `test_gate_and_batches.py` | `reports/validation/*.json`, `reports/figures/invalid_batch_violations.png` | ✅ |
| R2 | Scripts para simulação e detecção estatística de drift | 4 cenários + PSI/KS/qui² + calibração | `src/drift/simulation.py`, `statistics.py`, `concept.py`, `monitor.py` | `tests/unit/test_drift_statistics.py`, `test_concept_and_simulation.py` | `reports/drift/*_feature_drift.csv` | ✅ |
| R3 | Dashboards ou relatórios HTML comprovando o monitoramento | 4 relatórios Evidently + dashboard consolidado + MLflow | `src/drift/evidently_reports.py`, `src/monitoring/dashboard.py` | `tests/e2e/test_pipeline_e2e.py::test_reports_are_generated` | `reports/evidently/*.html`, `reports/monitoring_dashboard.html`, `reports/screenshots/` | ✅ |
| R4 | Documentação consolidada de Governança no README.md | Seções 11–14 do README | `README.md` | Revisão (seção 4 abaixo) | README §11–14 | ✅ |
| R5 | Histórico de commits semântico e organizado | Commits no padrão Conventional Commits, organizados por marcos e validados marco a marco | histórico Git | Simulação dos commits com testes em checkout limpo (todos aprovados) | `docs/VALIDACAO_FINAL.md` §4 | ⏳ execução dos commits é sua |

## 2. Bibliotecas e boas práticas obrigatórias

| Requisito | Atendimento | Evidência | Status |
|---|---|---|---|
| Scikit-Learn / XGBoost para o modelo base | scikit-learn 1.9.1 (regressão logística) | `src/model/train.py` | ✅ |
| Great Expectations **ou** Pandera | Pandera 0.36.0 | `src/validation/contracts.py` | ✅ |
| Evidently AI (ou Alibi Detect) | Evidently 0.7.23 (`DataDriftPreset` + `ClassificationPreset`) | `reports/evidently/` | ✅ |
| Separação Referência (treino) × Produção (drift) | 4 partições disjuntas, verificação de 0 ids em comum | `artifacts/data_split_manifest.json`, `test_prepare.py::test_no_leakage_between_partitions` | ✅ |
| ≥ 3 regras rígidas de qualidade (ex.: sem nulos, faixa de idade, limites de renda) | 9 regras (R01–R09), incluindo nulos, idade 18–100 e limite de crédito (proxy de renda, já que o dataset não tem renda) | README §7.1 | ✅ |
| Relatório estatístico (PSI ou KS) | PSI + KS + qui² por feature; PSI/KS do score | `reports/drift/*.csv`, README §8.3 | ✅ |
| Documentação de compliance com plano de privacidade (LGPD) | README §11 | README | ✅ |
| Opcional: Prometheus-client | Não usado (MLflow foi a escolha de rastreamento, permitida no PDF) | — | n/a |

## 3. Etapas, entregáveis e critérios de avaliação

### Validação de Dados — 25%

| Requisito oficial | Implementação | Arquivo | Validação | Evidência | Status | Melhorias possíveis |
|---|---|---|---|---|---|---|
| Treinar classificador base em dataset limpo (Referência) | LogReg com seleção na validação e avaliação holdout | `src/model/train.py` | `tests/integration/test_training_and_inference.py` | AUC 0,7607; MLflow `credit-scoring-training`; registry v1 | ✅ | Comparar com um GBM como desafiante |
| Contrato com regras estritas (idade > 18, renda não nula, sem duplicatas) | R05 (idade 18–100), R04 (sem nulos, incluindo `LIMIT_BAL`), R07 (sem `client_id` duplicado) + outras 6 regras | `contracts.py` | 1 teste por regra | `tests/unit/test_contracts.py` | ✅ | Regras entre colunas (ex.: fatura × limite) |
| Forçar lote com erros e demonstrar o contrato falhando e bloqueando a ingestão | `batch_invalid.csv` com 10 violações injetadas → 12 falhas em 7 regras → exceção, quarentena, alerta CRITICAL, run FAILED, 0 predições | `src/data/batches.py`, `gate.py`, `predict.py` | `test_invalid_batch_blocks_inference_and_produces_no_predictions`, e2e | `reports/validation/batch_invalid_*`, `reports/screenshots/mlflow_failed_run.png` | ✅ | — |
| Entregável: script de validação operante e baseline treinado | `python -m src.cli validate` / `train` | `src/cli.py` | `test_cli_step_by_step_flow` | exit code 0 / 2 | ✅ | — |

### Detecção de Drift — 25%

| Requisito oficial | Implementação | Arquivo | Validação | Evidência | Status | Melhorias possíveis |
|---|---|---|---|---|---|---|
| Dataset de Produção simulando passagem do tempo, alterando ≥ 2 variáveis importantes | Mudança de mix (LIMIT_BAL −57% PSI 1,20; AGE PSI 0,19) e mudança de P(Y\|X) | `simulation.py` | `test_data_drift_shifts_limit_and_age_but_keeps_real_labels` | `reports/figures/shift_limit_bal.png`, `shift_age.png` | ✅ | Série temporal com várias janelas |
| Gerar predições para o novo dataset | Inferência com gate por cenário | `monitor.py` → `predict.py` | e2e | `data/production/predictions/` | ✅ | — |
| Evidently comparando Referência × Produção | 4 relatórios | `evidently_reports.py` | `test_reports_are_generated` | `reports/evidently/*.html` | ✅ | — |
| Testes estatísticos (PSI, KS) identificando features com drift | PSI (quantis) + KS/qui², severidade | `statistics.py` | `test_feature_drift_table_identifies_only_shifted_features`, `test_no_drift_scenario_flags_nothing` | `reports/drift/data_drift_feature_drift.csv` | ✅ | — |
| Entregável: relatório apontando claramente as variáveis degradadas | Evidently + tabela do README §8.3 + heatmap de PSI | — | — | `reports/figures/psi_heatmap.png` | ✅ | — |
| (Contexto) Concept Drift | Mecanismo com X idêntico, teste de calibração, regras suspeito/confirmado | `concept.py` | `test_calibration_*`, `test_concept_drift_keeps_features_and_changes_labels` | `calibration_by_score_decile.png`, AUC 0,658 | ✅ | Monitorar janelas de maturação de outcome |

### Observabilidade e Logs — 20%

| Requisito oficial | Implementação | Arquivo | Validação | Evidência | Status | Melhorias possíveis |
|---|---|---|---|---|---|---|
| Centralizar logs (sucesso/falha de ingestão e transformação) e resultados de drift | JSONL com `run_id`, etapas, status, alertas | `logger.py`, `observability.py` | `test_logs_are_correlated_by_run_id`, `test_structured_logs_are_json_lines_with_run_id` | `logs/pipeline.jsonl` (local) | ✅ | Envio para um coletor (ELK/Loki) |
| Rastreamento (MLflow) | Experimentos de treino e monitoramento, run pai/filhos, registry, runs FAILED | `tracker.py`, `run_pipeline.py`, `monitor.py` | `test_mlflow_has_successful_and_failed_runs` | `reports/screenshots/mlflow_*.png` | ✅ | Servidor MLflow compartilhado |
| Definir e documentar métricas de saúde (status, latência, volume, alertas) | Tabela README §9.1 + regras §9.2 | README | — | `monitoring_summary.json` | ✅ | — |
| Entregável: dashboard/ambiente de rastreamento com saúde do pipeline, estabilidade e alertas | Dashboard HTML + MLflow | `dashboard.py` | `test_reports_are_generated` | `reports/monitoring_dashboard.html` | 🟡 → ✅ | A "estabilidade da infraestrutura" é medida apenas no processo local (RSS/CPU); não há servidores. Isso está documentado como limitação, sem afirmações falsas |

### Governança (LGPD) — 15%

| Requisito oficial | Implementação | Evidência | Status | Melhorias possíveis |
|---|---|---|---|---|
| Plano de proteção de dados: como PII foi tratada | Pseudonimização HMAC, minimização (R02), supressão de grupos pequenos, logs sem PII | README §11.2–11.3 | ✅ | Gestão de chave em cofre |
| Base legal para a decisão de crédito | Art. 7º, X (proteção do crédito) + V; limites do art. 11 e da Lei 12.414 | README §11.1 | ✅ | Parecer jurídico |
| Plano de retenção | Tabela de temporalidade proposta | README §11.3 | ✅ | — |
| Mitigação de vieses | Auditoria por grupo com supressão, achados e ações | README §12, `reports/fairness/` | ✅ | Testar remoção de `EDUCATION` e *proxies* |
| Documentação de causalidade | Diagrama causal + demonstrado × hipótese | README §13 | ✅ | Dados macroeconômicos reais |

### Vídeo STAR — 15%

| Requisito oficial | Implementação | Evidência | Status |
|---|---|---|---|
| Vídeo ≤ 5 min, método STAR (S, T, A, R) | Apresentação da situação, tarefa, ações e resultados reais deste repositório | Link no README | ⏳ gravação e link pendentes (externo) |

## 4. Auditoria crítica (papel de avaliador)

Revisão feita sobre código, arquitetura, documentação, testes e resultados. Problemas encontrados **durante o desenvolvimento** e corrigidos (com nova execução das verificações afetadas):

| Problema encontrado | Correção | Verificação |
|---|---|---|
| Evidently 0.7.23 incompatível com pandas 3.1.1 (`ImportError`) | pandas fixado em 2.3.3 | instalação limpa + `pip check` |
| MLflow 3.18 recusava salvar o modelo (skops: tipo não confiável) | `skops_trusted_types` declarado | treino + teste de integração |
| MLflow abria na visão GenAI (sem tabela de runs) | tag `mlflow.experimentKind = custom_model_development` | screenshot `mlflow_runs.png` |
| Mensagem de alerta dizia "1 feature" e listava 14 | mensagem separa severas e moderadas | log e dashboard revisados |
| Gráfico de AUC em barras com eixo truncado (distorce) | trocado por pontos com IC | figura `scenario_auc.png` |
| Relatório Evidently dizia "Dataset Drift NOT detected" com 39% das colunas em drift (limiar padrão 0,5) | `drift_share = 0,25` com justificativa documentada no `config.yaml` | relatório `data_drift_report.html` |
| PSI do Evidently menor que o próprio em PAY_AMT | investigado: o Evidently usa bins de largura igual (Sturges); diferença documentada | README §8.3 |
| Dados sintéticos de teste com sinal fraco de concept drift (p = 0,04) | gerador ajustado; o efeito de modelo mal especificado foi documentado como cautela | testes e2e |
| `validate` levava 6,6 s pela importação de MLflow/Evidently | importação preguiçosa no CLI → 0,75 s | medição local |
| Coluna "column" do relatório mostrava o nome do schema para R02 | rótulo corrigido para a coluna real (`SEX`) | `batch_invalid_validation.json` |

Verificações de integridade:

* **Vazamento de dados:** a produção não participa do treino; o EDA é só no treino; o limiar é escolhido na validação; a referência é usada só para avaliação e monitoramento. ✅
* **Resultados inventados:** todas as métricas do README vêm de `reports/*.json` da execução `20261116T194926Z-8cb19635`, e uma segunda execução independente em clone limpo reproduziu os mesmos valores. ✅
* **Reprodutibilidade:** seeds fixas, hash do dataset, versão do modelo derivada do hash do treino. ✅
* **Instalação:** versões fixadas e validadas juntas em Python 3.12 (Linux). Windows: comandos preparados, mas a execução nativa não foi possível no meu ambiente. 🟡

## 5. Estimativa fundamentada de aderência

| Critério | Peso | Aderência estimada | Justificativa |
|---|---:|---|---|
| Validação de dados | 25% | Alta | 9 regras, 3 contratos, bloqueio comprovado de ponta a ponta (log, alerta, quarentena, MLflow FAILED, 0 predições) |
| Detecção de drift | 25% | Alta | Simulação controlada que separa P(X) de P(Y\|X), PSI + KS + Evidently, relatórios claros |
| Observabilidade e logs | 20% | Alta | MLflow (sucesso e falha), logs correlacionados, alertas com severidade, dashboard. Limite: sem infraestrutura real (documentado) |
| Governança (LGPD) | 15% | Alta | Base legal, PII, retenção, vieses e causalidade, separando implementado de recomendado |
| Vídeo STAR | 15% | **Pendente** | Depende da gravação e da publicação do link |

Não é possível garantir nota máxima: a avaliação do vídeo e a leitura da banca são externas. **Pendências externas:** (1) executar os commits e publicar no GitHub; (2) confirmar a execução no Windows com os comandos da seção 15 do README; (3) gravar o vídeo e inserir o link no README.
