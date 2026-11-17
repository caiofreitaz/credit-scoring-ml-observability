# Validação final — resultados reais

Este documento registra **o que foi efetivamente executado** e com qual resultado. Nada aqui é estimado.

## 1. Ambiente de validação

| Item | Valor |
|---|---|
| Sistema | Linux (container de desenvolvimento) |
| Python | 3.12.3 |
| Dependências | exatamente as de `requirements.txt` (`pip check`: *No broken requirements found*) |
| Data | 16/11/2026 |
| Caminho de teste | pastas **com espaço** (`/tmp/val fase 4/…`, `/tmp/final fase 4…/…`), simulando `C:\Users\Caio\Documents\fase 4` |

> **Não validado nativamente:** Windows/PowerShell. Os comandos foram preparados para Windows (aspas em caminhos com espaço, `Activate.ps1`, `$LASTEXITCODE`, `ConvertFrom-Json`, scripts `.ps1` com BOM e CRLF), e verifiquei pontos de risco conhecidos: comprimento máximo de caminho no `mlruns` (~185 caracteres no Windows, abaixo de 260), URI SQLite com barras normais, URIs de artefato com espaço, ausência de caracteres fora do cp1252 no console. A confirmação final deve ser feita com os comandos da seção 15 do README.

## 2. Execução de referência (fonte das evidências do README)

| Item | Resultado |
|---|---|
| `run_id` | `20261116T194926Z-8cb19635` |
| Status | SUCCESS (14 etapas SUCCESS + 1 BLOCKED intencional) |
| Duração total das etapas | 22,0 s (dados já baixados) |
| Dataset | 30.000 × 25 · SHA-256 `84ba892a…0248` · fonte usada: espelho do `.xls` original (UCI bloqueada pelo proxy do ambiente) |
| Modelo | `1.0.0+3f888c17b195`, registry v1, AUC referência 0,7607 |
| Lotes | `batch_valid` aceito (500 predições, 11,3 ms) · `batch_invalid` rejeitado (12 falhas, 7 regras, 0 predições) |
| Alertas | 9 CRITICAL · 4 WARNING · 2 INFO |
| Pico de memória do processo | 482,8 MB |

## 3. Clone limpo (simulação do avaliador)

Procedimento: cópia **somente dos arquivos versionáveis** (lista gerada pelo próprio Git respeitando o `.gitignore`, sem `reports/`), **novo** ambiente virtual e instalação a partir do `requirements.txt`.

| Etapa | Comando | Resultado |
|---|---|---|
| Instalação | `pip install -r requirements.txt` + `pip check` | OK, sem conflitos (~20 s com cache) |
| Importação | `python -c "import src, evidently, mlflow, pandera, sklearn"` | `Ambiente OK` |
| Pipeline | `python -m src.cli run-all` | exit 0 em **33 s** (com download), `status: SUCCESS`, `run_id 20261116T200948Z-b4a97b59` |
| Reprodutibilidade | comparação com a execução de referência | **métricas, features com drift, calibração, versão do modelo e alertas idênticos** |
| Lint | `ruff check .` / `ruff format --check .` | `All checks passed!` / nenhum arquivo a reformatar |
| Testes | `python -m pytest --cov` | **109 passed** em 131,8 s · cobertura **97,79%** |

Uma terceira execução, comando a comando (`download-data` → `monitor`), também foi feita em clone limpo. Exit codes observados: 0, 0, 0, **2** (validação do lote inválido, intencional), 0, 0, **2** (inferência do lote inválido, intencional), 0, 0.

## 4. Validação dos marcos de commit

Os 12 marcos de código do plano de commits foram aplicados em sequência num repositório temporário. Após cada commit, os testes do marco rodaram num **checkout limpo contendo apenas os arquivos commitados até ali**:

| Commit | Teste do marco | Resultado |
|---:|---|---|
| 1 | importação de `src.config` | `21` features |
| 2 | importação de `src.data.*` | ok |
| 3 | importação de `src.validation.gate` | ok |
| 4 | testes de ingestão, preparação, contratos e gate | **49 passed** |
| 5 | importação de `src.model.*` | ok |
| 6 | testes de métricas e integração | **12 passed** |
| 7 | testes de drift e simulação | **23 passed** |
| 8 | importação de `src.drift.monitor` e `src.data.eda` | ok |
| 9 | testes de monitoramento | **14 passed** |
| 10 | `python -m src.cli --help` | ok |
| 11 | suíte completa com cobertura | **109 passed**, 98% |
| 12 | `ruff check` + `ruff format --check` | All checks passed |

Os commits seguintes contêm somente documentação e relatórios.

## 5. Cobertura por módulo (execução final)

| Módulo | Cobertura | Módulo | Cobertura |
|---|---:|---|---:|
| `cli.py` | 100% | `monitoring/alerts.py` | 100% |
| `config.py` | 100% | `monitoring/dashboard.py` | 100% |
| `data/batches.py` | 100% | `monitoring/logger.py` | 95% |
| `data/eda.py` | 100% | `monitoring/observability.py` | 100% |
| `data/ingest.py` | 97% | `monitoring/plots.py` | 100% |
| `data/prepare.py` | 96% | `monitoring/tracker.py` | 91% |
| `drift/concept.py` | 97% | `pipeline/run_pipeline.py` | 94% |
| `drift/evidently_reports.py` | 95% | `validation/contracts.py` | 100% |
| `drift/monitor.py` | 97% | `validation/gate.py` | 96% |
| `drift/simulation.py` | 100% | `model/train.py` | 100% |
| `drift/statistics.py` | 100% | `model/predict.py` | 96% |
| `model/features.py` | 100% | `model/metrics.py` | 95% |
| `model/fairness.py` | 98% | **TOTAL** | **97,8%** |

As linhas não cobertas são ramos defensivos: erro de leitura de rede real (`http_fetch`), exceção não lazy do Pandera e caminho de falha genérica do pipeline. Não criei testes artificiais só para cobri-las.

## 6. O que a suíte cobre (mapa requisito → teste)

| Item exigido | Teste(s) |
|---|---|
| Carregamento dos dados | `test_ingest.py` (normalização, hash, fallback, *zip* sem `.xls`, arquivo ausente) |
| Preparação e transformação | `test_prepare.py::test_clean_*`, `test_features_metrics.py::test_feature_engineer_adds_columns` |
| Separação dos conjuntos | `test_split_sizes_and_stratification`, `test_split_is_reproducible` |
| Ausência de vazamento | `test_no_leakage_between_partitions`, `test_leakage_is_detected` |
| Contratos (cada regra) | `test_contracts.py::test_r01…test_r09` |
| Aceitação de dados válidos | `test_valid_batches_pass_both_contracts`, `test_valid_demo_batch_is_accepted` |
| Rejeição de dados inválidos | `test_invalid_demo_batch_violates_every_injected_rule` |
| Bloqueio da inferência | `test_invalid_batch_blocks_inference_and_produces_no_predictions`, `test_enforce_contract_blocks_quarantines_logs_and_alerts` |
| Treinamento do modelo | `test_training_persists_model_metadata_and_reference_scores` |
| Persistência e carregamento | `test_persisted_model_loads_and_predicts_identically` |
| Geração de predições | `test_valid_batch_generates_predictions` |
| Reprodutibilidade | `test_training_is_reproducible`, `test_scenarios_are_reproducible` |
| Simulação de Data Drift | `test_data_drift_shifts_limit_and_age_but_keeps_real_labels`, `test_mix_weights_*` |
| Simulação de Concept Drift | `test_concept_drift_keeps_features_and_changes_labels`, `test_concept_mechanism_targets_expected_clients` |
| Identificação de variáveis com drift | `test_feature_drift_table_identifies_only_shifted_features`, `test_scenarios_are_diagnosed` |
| Cenário sem drift | `test_no_drift_scenario_flags_nothing`, `test_calibration_stable_when_relationship_unchanged` |
| Geração de alertas | `test_data_drift_rule_severity_levels`, `test_performance_rule`, `test_alerts_have_expected_severities` |
| Registro de logs | `test_structured_logs_are_json_lines_with_run_id`, `test_logs_are_correlated_by_run_id` |
| Registro de métricas | `test_training_is_logged_and_registered_in_mlflow`, `test_mlflow_has_successful_and_failed_runs` |
| Relatórios HTML | `test_reports_are_generated` |
| Pipeline completo | `test_pipeline_succeeds_and_blocks_invalid_batch`, `test_cli_step_by_step_flow`, `test_cli_run_all_reuses_existing_model` |

## 7. Pendências externas

1. Executar a sequência de commits e publicar no GitHub, confirmando o CI verde.
2. Executar o pipeline no Windows e confirmar os mesmos números.
3. Gravar o vídeo STAR e inserir o link no README.
