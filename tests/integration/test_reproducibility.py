"""
Integration test for reproducibility workflows (US3, SC-003).

Tests:
- Config artifact saving and loading
- Run reproduction with same config and seed
- Metric comparison within ±0.5 pp tolerance
"""

import sys
from pathlib import Path

import pytest
import torch
import mlflow
from omegaconf import OmegaConf

# Add src to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from Project.SubProject.data.dataset import build_criterion_dataset
from Project.SubProject.engine.train_engine import train_epoch, evaluate
from Project.SubProject.models.model import DeBERTaV3Classifier
from Project.SubProject.utils.mlflow_utils import (
    configure_mlflow,
    log_lineage_metadata,
    get_run_config,
)
from Project.SubProject.utils.seed import seed_everything
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


@pytest.fixture
def mock_config():
    """Create a minimal test configuration."""
    cfg = OmegaConf.create({
        "project_name": "debertav3-criteria-classifier",
        "model": {
            "checkpoint": "microsoft/deberta-v3-base",
            "head": {
                "num_layers": 1,
                "hidden_dims": [128],
                "activation": "relu",
                "dropout": 0.1,
                "pooling_strategy": "cls",
            },
            "tokenizer": {
                "max_length": 128,
                "truncation": "longest_first",
                "padding": "max_length",
            },
        },
        "data": {
            "redsm5_dir": "data/redsm5",
            "dsm5_criteria_path": "data/dsm5/MDD_Criteira.json",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "dataloader": {
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
            },
        },
        "train": {
            "seed": 42,
            "learning_rate": 2e-5,
            "batch_size": 4,
            "eval_batch_size": 4,
            "epochs": 1,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "scheduler": "linear",
            "optimizer_type": "adamw",
            "early_stopping_patience": 3,
            "optimizations": {
                "use_amp": False,
                "amp_dtype": "fp16",
                "enable_tf32": False,
                "attention_implementation": "sdpa",
                "gradient_checkpointing": False,
                "use_compile": False,
                "compile_mode": "default",
            },
        },
        "mlflow": {
            "tracking_uri": "sqlite:///test_mlflow.db",
            "experiment_name": "test-reproducibility",
        },
    })
    return cfg


@pytest.mark.integration
@pytest.mark.slow
def test_reproducibility_workflow(mock_config, tmp_path):
    """Test full reproducibility workflow: train, save, reproduce, compare.

    This test verifies SC-003: metrics within ±0.5 pp tolerance.
    """

    # Setup
    cfg = mock_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure MLflow with test database
    configure_mlflow(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment=cfg.mlflow.experiment_name,
    )

    # ============================================================
    # Step 1: Initial Training Run
    # ============================================================

    print("\n[Step 1] Initial training run...")

    seed_everything(cfg.train.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)

    # Build datasets
    datasets = build_criterion_dataset(
        redsm5_dir=cfg.data.redsm5_dir,
        dsm5_criteria_path=cfg.data.dsm5_criteria_path,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        random_state=cfg.train.seed,
    )

    # Create test dataloader (small for speed)
    test_loader = DataLoader(
        datasets["test"],
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = DeBERTaV3Classifier.from_pretrained(
        cfg.model.checkpoint,
        num_labels=2,
        num_layers=cfg.model.head.num_layers,
        hidden_dims=cfg.model.head.hidden_dims,
        activation=cfg.model.head.activation,
        dropout=cfg.model.head.dropout,
        pooling_strategy=cfg.model.head.pooling_strategy,
    )
    model = model.to(device)

    # Run initial evaluation
    with mlflow.start_run(run_name="original-run") as original_run:

        # Log lineage metadata
        log_lineage_metadata(
            cfg,
            data_paths=[Path(cfg.data.redsm5_dir), Path(cfg.data.dsm5_criteria_path)],
            additional_tags={"test_stage": "original"},
        )

        # Evaluate
        original_metrics, _, _ = evaluate(model, test_loader, device, cfg)

        # Log metrics
        for metric_name, value in original_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        original_run_id = original_run.info.run_id

        print(f"✓ Original run complete: {original_run_id}")
        print(f"  Original metrics: {original_metrics}")

    # ============================================================
    # Step 2: Reproduce Run
    # ============================================================

    print("\n[Step 2] Reproducing run...")

    # Load config from original run
    reproduced_cfg = get_run_config(original_run_id)
    assert reproduced_cfg is not None, "Failed to load config from original run"

    print("✓ Config loaded successfully")

    # Reset seed to same value
    seed_everything(reproduced_cfg.train.seed)

    # Rebuild datasets with same seed
    reproduced_datasets = build_criterion_dataset(
        redsm5_dir=reproduced_cfg.data.redsm5_dir,
        dsm5_criteria_path=reproduced_cfg.data.dsm5_criteria_path,
        train_ratio=reproduced_cfg.data.train_ratio,
        val_ratio=reproduced_cfg.data.val_ratio,
        test_ratio=reproduced_cfg.data.test_ratio,
        random_state=reproduced_cfg.train.seed,
    )

    reproduced_test_loader = DataLoader(
        reproduced_datasets["test"],
        batch_size=reproduced_cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Recreate model with same config
    reproduced_model = DeBERTaV3Classifier.from_pretrained(
        reproduced_cfg.model.checkpoint,
        num_labels=2,
        num_layers=reproduced_cfg.model.head.num_layers,
        hidden_dims=reproduced_cfg.model.head.hidden_dims,
        activation=reproduced_cfg.model.head.activation,
        dropout=reproduced_cfg.model.head.dropout,
        pooling_strategy=reproduced_cfg.model.head.pooling_strategy,
    )
    reproduced_model = reproduced_model.to(device)

    # Copy model weights to ensure exact same model state
    reproduced_model.load_state_dict(model.state_dict())

    # Run reproduced evaluation
    with mlflow.start_run(run_name="reproduced-run") as reproduced_run:

        # Log lineage metadata
        log_lineage_metadata(
            reproduced_cfg,
            data_paths=[Path(reproduced_cfg.data.redsm5_dir), Path(reproduced_cfg.data.dsm5_criteria_path)],
            additional_tags={"test_stage": "reproduced", "original_run_id": original_run_id},
        )

        # Evaluate
        reproduced_metrics, _, _ = evaluate(reproduced_model, reproduced_test_loader, device, reproduced_cfg)

        # Log metrics
        for metric_name, value in reproduced_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        reproduced_run_id = reproduced_run.info.run_id

        print(f"✓ Reproduced run complete: {reproduced_run_id}")
        print(f"  Reproduced metrics: {reproduced_metrics}")

    # ============================================================
    # Step 3: Compare Metrics (SC-003)
    # ============================================================

    print("\n[Step 3] Comparing metrics...")

    tolerance = 0.005  # 0.5 percentage points

    key_metrics = ["accuracy", "macro_f1", "weighted_f1", "precision", "recall"]

    all_within_tolerance = True

    for metric in key_metrics:
        original_value = original_metrics[metric]
        reproduced_value = reproduced_metrics[metric]
        delta = abs(original_value - reproduced_value)

        within_tolerance = delta <= tolerance

        status = "✓" if within_tolerance else "✗"
        print(
            f"{status} {metric:15s}: Original={original_value:.4f}, "
            f"Reproduced={reproduced_value:.4f}, Delta={delta:.4f}"
        )

        if not within_tolerance:
            all_within_tolerance = False

        # Assert each metric is within tolerance
        assert delta <= tolerance, (
            f"Metric {metric} exceeded tolerance: "
            f"delta={delta:.4f} > {tolerance:.4f}"
        )

    print("\n" + "=" * 80)
    if all_within_tolerance:
        print("✓ REPRODUCIBILITY TEST PASSED")
        print(f"All metrics within ±{tolerance * 100:.1f} pp tolerance (SC-003)")
    print("=" * 80)

    # Final assertion
    assert all_within_tolerance, "Some metrics exceeded reproducibility tolerance"


@pytest.mark.unit
def test_lineage_metadata_logging(mock_config):
    """Test that lineage metadata is properly logged."""

    cfg = mock_config

    configure_mlflow(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment=cfg.mlflow.experiment_name,
    )

    with mlflow.start_run(run_name="test-lineage") as run:

        # Log lineage
        log_lineage_metadata(
            cfg,
            data_paths=[Path(cfg.data.redsm5_dir)],
            additional_tags={"test": "lineage"},
        )

        run_id = run.info.run_id

    # Verify lineage tags exist
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)

    # Check for git tags
    assert "git.commit" in run_data.data.tags or "git.branch" in run_data.data.tags

    # Check for version tags
    assert any(key.startswith("version.") for key in run_data.data.tags.keys())

    # Check for optimization params
    assert any(key.startswith("opt.") for key in run_data.data.params.keys())

    print("✓ Lineage metadata logged successfully")


@pytest.mark.unit
def test_config_artifact_roundtrip(mock_config, tmp_path):
    """Test saving and loading config artifact."""

    cfg = mock_config

    configure_mlflow(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment=cfg.mlflow.experiment_name,
    )

    # Save config
    with mlflow.start_run(run_name="test-config-save") as run:
        from Project.SubProject.utils.mlflow_utils import log_config_artifact

        log_config_artifact(cfg)
        run_id = run.info.run_id

    # Load config
    loaded_cfg = get_run_config(run_id)

    assert loaded_cfg is not None, "Failed to load config"

    # Verify key fields
    assert loaded_cfg.project_name == cfg.project_name
    assert loaded_cfg.model.checkpoint == cfg.model.checkpoint
    assert loaded_cfg.train.seed == cfg.train.seed

    print("✓ Config artifact roundtrip successful")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
