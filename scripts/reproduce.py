#!/usr/bin/env python3
"""
Reproducibility CLI for DeBERTaV3 Criteria Classifier.

Implements:
- US3: Reproducibility and lineage
- SC-003: Replay with ±0.5 pp tolerance
- T024: Download config, restore Hydra overrides, rerun evaluation, compare metrics
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add src to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from Project.SubProject.data.dataset import build_criterion_dataset
from Project.SubProject.engine.train_engine import evaluate
from Project.SubProject.models.model import DeBERTaV3Classifier
from Project.SubProject.utils.mlflow_utils import configure_mlflow, get_run_config, log_lineage_metadata
from Project.SubProject.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_run_metrics(run_id: str) -> dict:
    """Retrieve metrics from an MLflow run.

    Args:
        run_id: MLflow run ID.

    Returns:
        Dict of metric names to values.
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    metrics = {}
    for key, value in run.data.metrics.items():
        metrics[key] = value

    return metrics


def compare_metrics(original: dict, replayed: dict, tolerance: float = 0.005) -> dict:
    """Compare metrics between original and replayed runs.

    Args:
        original: Original run metrics.
        replayed: Replayed run metrics.
        tolerance: Absolute tolerance (default 0.005 = 0.5 percentage points).

    Returns:
        Dict with comparison results.
    """
    comparison = {}

    # Focus on key metrics
    key_metrics = [
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_precision",
        "test_recall",
    ]

    for metric in key_metrics:
        if metric in original and metric in replayed:
            orig_val = original[metric]
            replay_val = replayed[metric]
            delta = abs(orig_val - replay_val)
            within_tolerance = delta <= tolerance

            comparison[metric] = {
                "original": orig_val,
                "replayed": replay_val,
                "delta": delta,
                "within_tolerance": within_tolerance,
            }

    return comparison


def main():
    """Main reproducibility function."""
    parser = argparse.ArgumentParser(description="Reproduce an MLflow training run")
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="MLflow run ID to reproduce",
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="debertav3-criteria",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Metric tolerance (default 0.005 = 0.5 pp)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "train"],
        default="eval",
        help="Reproducibility mode: eval (fast, eval only) or train (full retraining)",
    )

    args = parser.parse_args()

    setup_logging()

    logger.info("=" * 80)
    logger.info("Reproducibility Check")
    logger.info("=" * 80)
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Tolerance: {args.tolerance} ({args.tolerance * 100:.1f} pp)")

    # Configure MLflow
    configure_mlflow(
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
    )

    # Retrieve original run config
    logger.info("\nRetrieving original run configuration...")
    cfg = get_run_config(args.run_id)

    if cfg is None:
        logger.error(f"Could not load config from run {args.run_id}")
        sys.exit(1)

    logger.info("✓ Config loaded successfully")
    logger.info(f"\nConfig preview:\n{OmegaConf.to_yaml(cfg)}")

    # Retrieve original metrics
    logger.info("\nRetrieving original metrics...")
    original_metrics = get_run_metrics(args.run_id)

    test_metrics_found = any(k.startswith("test_") for k in original_metrics.keys())
    if not test_metrics_found:
        logger.warning("No test metrics found in original run")

    logger.info(f"✓ Retrieved {len(original_metrics)} metrics")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nUsing device: {device}")

    # Seed for reproducibility
    seed_everything(cfg.train.seed)
    logger.info(f"✓ Seed set to {cfg.train.seed}")

    # Enable TF32 if requested
    if cfg.train.optimizations.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load tokenizer
    logger.info(f"\nLoading tokenizer: {cfg.model.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)

    # Build datasets
    logger.info("\nBuilding datasets...")
    datasets = build_criterion_dataset(
        redsm5_dir=cfg.data.redsm5_dir,
        dsm5_criteria_path=cfg.data.dsm5_criteria_path,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        random_state=cfg.train.seed,
    )

    logger.info(f"✓ Test samples: {len(datasets['test'])}")

    # Create test dataloader
    test_loader = DataLoader(
        datasets["test"],
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.data.dataloader.pin_memory,
    )

    # Start MLflow run for replay
    with mlflow.start_run(run_name=f"reproduce-{args.run_id[:8]}"):

        # Log lineage
        data_paths = [
            Path(cfg.data.redsm5_dir),
            Path(cfg.data.dsm5_criteria_path),
        ]
        log_lineage_metadata(
            cfg,
            data_paths=data_paths,
            additional_tags={
                "stage": "reproducibility",
                "original_run_id": args.run_id,
                "mode": args.mode,
            },
        )

        mlflow.log_params({
            "original_run_id": args.run_id,
            "mode": args.mode,
            "tolerance": args.tolerance,
        })

        if args.mode == "eval":
            # Download and load model from original run
            logger.info(f"\nDownloading model from run {args.run_id}...")

            try:
                client = mlflow.tracking.MlflowClient()
                model_uri = f"runs:/{args.run_id}/model"
                local_model_path = mlflow.pytorch.load_model(model_uri)

                # If load_model returns a PyTorch model directly
                if isinstance(local_model_path, torch.nn.Module):
                    model = local_model_path
                else:
                    # Otherwise download artifacts and load manually
                    model_path = client.download_artifacts(args.run_id, "model")
                    model = DeBERTaV3Classifier.from_pretrained(
                        model_path,
                        num_labels=2,
                        num_layers=cfg.model.head.num_layers,
                        hidden_dims=cfg.model.head.hidden_dims,
                        activation=cfg.model.head.activation,
                        dropout=cfg.model.head.dropout,
                        pooling_strategy=cfg.model.head.pooling_strategy,
                    )

                model = model.to(device)
                logger.info("✓ Model loaded successfully")

            except Exception as e:
                logger.error(f"Could not load model from run: {e}")
                logger.info("Falling back to retraining mode...")
                args.mode = "train"

        if args.mode == "train":
            # Full retraining (not implemented in this version)
            logger.error("Full retraining mode not yet implemented in reproduce.py")
            logger.info("Please use scripts/train.py with the downloaded config instead")
            sys.exit(1)

        # Run evaluation
        logger.info("\nRunning evaluation...")
        test_metrics, test_preds, test_labels = evaluate(model, test_loader, device, cfg)

        # Log replayed metrics
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"replayed_{metric_name}", value)

        logger.info(f"✓ Evaluation complete")
        logger.info(f"Replayed test metrics: {test_metrics}")

        # Compare metrics
        logger.info("\n" + "=" * 80)
        logger.info("Metric Comparison (SC-003: ±0.5 pp tolerance)")
        logger.info("=" * 80)

        # Build replayed metrics dict
        replayed_metrics = {f"test_{k}": v for k, v in test_metrics.items()}

        comparison = compare_metrics(original_metrics, replayed_metrics, tolerance=args.tolerance)

        all_within_tolerance = True

        for metric, result in comparison.items():
            status = "✓" if result["within_tolerance"] else "✗"
            logger.info(
                f"{status} {metric:20s}: "
                f"Original={result['original']:.4f}, "
                f"Replayed={result['replayed']:.4f}, "
                f"Delta={result['delta']:.4f}"
            )

            if not result["within_tolerance"]:
                all_within_tolerance = False

            # Log comparison metrics
            mlflow.log_metric(f"delta_{metric}", result["delta"])
            mlflow.log_metric(f"original_{metric}", result["original"])

        # Overall status
        logger.info("\n" + "=" * 80)
        if all_within_tolerance:
            logger.info("✓ REPRODUCIBILITY CHECK PASSED")
            logger.info(f"All metrics within ±{args.tolerance * 100:.1f} pp tolerance")
            mlflow.set_tag("reproducibility_status", "passed")
            exit_code = 0
        else:
            logger.info("✗ REPRODUCIBILITY CHECK FAILED")
            logger.info(f"Some metrics exceeded ±{args.tolerance * 100:.1f} pp tolerance")
            mlflow.set_tag("reproducibility_status", "failed")
            exit_code = 1

        logger.info("=" * 80)

        sys.exit(exit_code)


if __name__ == "__main__":
    main()
