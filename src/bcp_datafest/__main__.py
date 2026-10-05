import argparse
from pathlib import Path

from .io import ROOT


def main():
    parser = argparse.ArgumentParser(description="DataFest BCP: validación temporal y CSV verificable")
    parser.add_argument("command", choices=["audit", "baseline", "experiment", "tune", "blend", "evaluate-final", "fit-final", "predict", "validate-submission", "boost-audit", "boost-experiment", "boost-diagnose"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model", choices=["catboost", "lightgbm"])
    parser.add_argument("--trials", type=int)
    parser.add_argument("--competitive", action="store_true", help="Run the bounded Candidate B competitive comparison")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/submission.csv")
    parser.add_argument("--file", type=Path, default=ROOT / "outputs/submission.csv")
    args = parser.parse_args()
    if args.command == "boost-audit":
        from .variability import audit_variability
        audit_variability(args.data_dir)
    elif args.command == "boost-experiment":
        from .boost import experiment
        experiment(args.data_dir, competitive=args.competitive)
    elif args.command == "boost-diagnose":
        from .diagnostics import diagnose
        diagnose(args.data_dir)
    elif args.command == "audit":
        from .data import audit
        audit(args.data_dir)
    elif args.command in ["baseline", "experiment"]:
        from .experiments import baseline, experiment
        {"baseline": baseline, "experiment": experiment}[args.command](args.data_dir)
    elif args.command == "tune":
        from .tuning import tune
        if not args.model:
            parser.error("tune requires --model")
        tune(args.data_dir, args.model, args.trials)
    elif args.command == "blend":
        from .ensemble import select
        select(args.data_dir)
    elif args.command in ["evaluate-final", "fit-final", "predict"]:
        from .predict import evaluate_final, fit_final, predict
        if args.command == "predict":
            predict(args.data_dir, args.output)
        else:
            {"evaluate-final": evaluate_final, "fit-final": fit_final}[args.command](args.data_dir)
    else:
        from .data import load
        from .submission import validate_submission
        from .io import status
        _, test, sample, _ = load(args.data_dir)
        result = validate_submission(args.file, test, sample)
        status(f"Fase 8: validate-submission aprobado. {result}")
        print(result)


if __name__ == "__main__":
    main()
