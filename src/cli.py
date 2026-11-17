"""Command-line interface. Run from the repository root: ``python -m src.cli <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import load_config, resolve_paths
from src.drift.simulation import SCENARIOS
from src.monitoring.logger import configure_logging, new_run_id, set_run_id
from src.validation.gate import read_batch, validate_batch

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONTRACT_VIOLATION = 2


def _pipeline():
    """Lazy import: heavy dependencies (MLflow, Evidently) load only when a command needs them."""
    from src.pipeline import run_pipeline

    return run_pipeline


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def cmd_download(args: argparse.Namespace) -> int:
    rp = _pipeline()
    ctx = rp.create_context(args.config, args.root)
    manifest = rp.stage_download(ctx, force=args.force)
    _print_json({k: manifest[k] for k in ("source_name", "rows", "columns", "canonical_sha256")})
    return EXIT_OK


def cmd_prepare(args: argparse.Namespace) -> int:
    rp = _pipeline()
    ctx = rp.create_context(args.config, args.root)
    manifest = rp.stage_prepare(ctx)
    _print_json(manifest)
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    configure_logging(resolve_paths(config, args.root).logs_dir)
    set_run_id(new_run_id())
    name = args.name or Path(args.input).stem
    result = validate_batch(read_batch(args.input), args.mode, name)
    payload = result.to_dict()
    _print_json(payload)
    if result.is_valid:
        print(f"\n[OK] Lote '{name}' ACEITO pelo contrato '{args.mode}'.")
        return EXIT_OK
    print(f"\n[BLOQUEADO] Lote '{name}' REJEITADO: {len(result.rules_violated)} regras violadas.")
    return EXIT_CONTRACT_VIOLATION


def cmd_train(args: argparse.Namespace) -> int:
    rp = _pipeline()
    ctx = rp.create_context(args.config, args.root)
    metadata = rp.stage_train(ctx)
    ref = metadata["reference_metrics"]
    _print_json(
        {
            "model_version": metadata["model_version"],
            "registered_model_version": metadata["registered_model_version"],
            "C": metadata["params"]["C"],
            "threshold": metadata["threshold"],
            "reference_metrics": {k: round(v, 4) for k, v in ref.items() if isinstance(v, float)},
            "confusion_matrix": {k: ref[k] for k in ("tn", "fp", "fn", "tp")},
        }
    )
    return EXIT_OK


def cmd_predict(args: argparse.Namespace) -> int:
    rp = _pipeline()
    ctx = rp.create_context(args.config, args.root)
    name = args.name or Path(args.input).stem
    result = rp.run_batch_inference(ctx, args.input, name)
    ctx.alerts.save(ctx.paths.reports_dir / "validation" / f"{name}_alerts.json")
    _print_json(
        {
            k: result[k]
            for k in (
                "batch",
                "status",
                "n_rows",
                "n_failures",
                "rules_violated",
                "predictions_generated",
                "mlflow_run_id",
            )
        }
    )
    if result["status"] == "rejected":
        print(f"\n[BLOQUEADO] Inferência bloqueada para '{name}'. Nenhuma predição foi gerada.")
        return EXIT_CONTRACT_VIOLATION
    print(
        f"\n[OK] {result['predictions_generated']} predições salvas em "
        f"data/production/predictions/{name}_predictions.csv"
    )
    return EXIT_OK


def cmd_simulate(args: argparse.Namespace) -> int:
    rp = _pipeline()
    ctx = rp.create_context(args.config, args.root)
    _print_json(rp.stage_simulate(ctx))
    return EXIT_OK


def cmd_monitor(args: argparse.Namespace) -> int:
    rp = _pipeline()
    ctx = rp.create_context(args.config, args.root)
    scenarios = tuple(args.scenario) if args.scenario else SCENARIOS
    summaries = rp.stage_monitor(ctx, scenarios)
    rp.finalize(ctx, rp.load_batch_results(ctx), summaries, "SUCCESS")
    for s in summaries:
        perf = s["performance"].get("current", {})
        print(
            f"{s['scenario']:<24} drift={s['n_drifted_features']:>2} severas={s['n_severe_drift']} "
            f"psi_score={s['prediction_drift']['prediction_psi']:.3f} auc={perf.get('roc_auc', float('nan')):.4f} "
            f"calib_p={s['calibration']['calibration_pvalue']:.2e} -> {s['diagnosis']['diagnosis']}"
        )
    print("\nDashboard: reports/monitoring_dashboard.html | Evidently: reports/evidently/")
    return EXIT_OK


def cmd_run_all(args: argparse.Namespace) -> int:
    rp = _pipeline()
    ctx = rp.create_context(args.config, args.root)
    summary = rp.run_full_pipeline(ctx, skip_download=args.skip_download, skip_train=args.skip_train)
    print("\n================ RESUMO DA EXECUÇÃO ================")
    print(f"run_id: {summary['run_id']} | status: {summary['status']}")
    for b in summary["batches"]:
        print(f"lote {b['batch']:<14} -> {b['status']:<8} predições={b['predictions_generated']}")
    for s in summary["scenarios"]:
        print(f"cenário {s['scenario']:<24} -> {s['diagnosis']['diagnosis']}")
    print(f"alertas: {summary['alerts_total']}")
    print("dashboard: reports/monitoring_dashboard.html")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Credit scoring reliability pipeline")
    parser.add_argument("--config", default=None, help="caminho alternativo para o config.yaml")
    parser.add_argument("--root", default=None, help="raiz alternativa para dados/artefatos (usado nos testes)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download-data", help="baixa e verifica o dataset UCI")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    sub.add_parser("prepare-data", help="limpa, pseudonimiza, separa e gera lotes de demonstração").set_defaults(
        func=cmd_prepare
    )

    p = sub.add_parser("validate", help="valida um CSV contra um contrato")
    p.add_argument("--input", required=True)
    p.add_argument("--mode", choices=["training", "inference", "outcome"], default="inference")
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_validate)

    sub.add_parser("train", help="treina, avalia e registra o baseline").set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="inferência com gate de contrato")
    p.add_argument("--input", required=True)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_predict)

    sub.add_parser("simulate-drift", help="gera os 4 cenários de produção").set_defaults(func=cmd_simulate)

    p = sub.add_parser("monitor", help="monitora cenários e gera relatórios")
    p.add_argument("--scenario", nargs="*", choices=list(SCENARIOS))
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("run-all", help="executa o pipeline completo")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.set_defaults(func=cmd_run_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
