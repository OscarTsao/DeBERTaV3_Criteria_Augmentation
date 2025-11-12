#!/usr/bin/env python3
"""
Training script for DeBERTaV3 Criteria Matching Classifier.

Implements:
- FR-001 to FR-013: Core training requirements
- FR-017: Baseline configuration
- US1: Baseline training and evaluation
- Full MLflow lineage tracking (FR-009, US3)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from Project.SubProject.data.dataset import build_criterion_dataset, CriteriaDataset
from Project.SubProject.engine.train_engine import train_epoch, evaluate
from Project.SubProject.models.model import DeBERTaV3Classifier
from Project.SubProject.utils.mlflow_utils import (
    configure_mlflow,
    log_lineage_metadata,
)
from Project.SubProject.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def dry_run_batch(model, dataloader, device, cfg):
    """Validate single batch for tensor shapes and tokenization (FR-008)."""
    logger.info("Running dry-run validation on first batch...")
    model.eval()

    try:
        batch = next(iter(dataloader))

        # Check batch structure
        required_keys = ["input_ids", "attention_mask", "labels"]
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key in batch: {key}")

        # Move to device and check shapes
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        batch_size = batch["input_ids"].shape[0]
        seq_length = batch["input_ids"].shape[1]

        logger.info(f"Batch size: {batch_size}, Sequence length: {seq_length}")

        # Validate sequence length
        if seq_length > cfg.model.tokenizer.max_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds max_length {cfg.model.tokenizer.max_length}"
            )

        # Forward pass
        with torch.no_grad():
            if cfg.train.optimizations.use_amp:
                amp_dtype = torch.bfloat16 if cfg.train.optimizations.amp_dtype == "bf16" else torch.float16
                with autocast(dtype=amp_dtype):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

        # Check outputs
        if "loss" not in outputs:
            raise ValueError("Model did not return loss")
        if "logits" not in outputs:
            raise ValueError("Model did not return logits")

        logger.info(f"✓ Dry-run successful. Loss: {outputs['loss'].item():.4f}")
        logger.info(f"✓ Logits shape: {outputs['logits'].shape}")

    except Exception as e:
        logger.error(f"✗ Dry-run failed: {e}")
        raise


def create_optimizer(model, cfg):
    """Create optimizer with optional fused or 8-bit variants."""
    optimizer_type = cfg.train.optimizer_type
    lr = cfg.train.learning_rate
    weight_decay = cfg.train.weight_decay

    # Get parameters
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_type == "adamw_torch_fused":
        logger.info("Using fused AdamW optimizer")
        try:
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr, fused=True)
        except Exception as e:
            logger.warning(f"Fused AdamW failed ({e}), falling back to standard AdamW")
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    elif optimizer_type == "adamw_bnb_8bit":
        logger.info("Using 8-bit AdamW optimizer")
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=lr)
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to standard AdamW")
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    else:  # "adamw"
        logger.info("Using standard AdamW optimizer")
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    return optimizer


def create_scheduler(optimizer, cfg, num_training_steps):
    """Create learning rate scheduler."""
    scheduler_type = cfg.train.scheduler
    num_warmup_steps = int(cfg.train.warmup_ratio * num_training_steps)

    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    else:  # constant
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

    return scheduler


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    setup_logging(cfg)

    # Print resolved config
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))

    # Seed for reproducibility (FR-009)
    seed_everything(cfg.train.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Enable TF32 if requested (Optimization_List)
    if cfg.train.optimizations.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✓ TF32 enabled for matmul and cuDNN")

    # Configure MLflow (FR-006)
    configure_mlflow(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment=cfg.mlflow.experiment_name,
    )

    # Start MLflow run
    with mlflow.start_run(run_name=cfg.get("run_name", "baseline-train")):

        # Log lineage metadata (FR-009, US3, T023)
        data_paths = [
            Path(cfg.data.redsm5_dir),
            Path(cfg.data.dsm5_criteria_path),
        ]
        log_lineage_metadata(cfg, data_paths=data_paths, additional_tags={"stage": "train"})

        # Load tokenizer (FR-001)
        logger.info(f"Loading tokenizer: {cfg.model.checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)

        # Build datasets (FR-014, FR-016)
        logger.info("Building datasets...")
        datasets = build_criterion_dataset(
            redsm5_dir=cfg.data.redsm5_dir,
            dsm5_criteria_path=cfg.data.dsm5_criteria_path,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            random_state=cfg.train.seed,
        )

        logger.info(f"Train samples: {len(datasets['train'])}")
        logger.info(f"Val samples: {len(datasets['val'])}")
        logger.info(f"Test samples: {len(datasets['test'])}")

        # Get class weights (FR-015)
        class_weight = datasets["train"].get_class_weight()
        class_weight_tensor = torch.tensor(class_weight, dtype=torch.float32, device=device)
        logger.info(f"Class weights: {class_weight}")

        # Create dataloaders
        train_loader = DataLoader(
            datasets["train"],
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.data.dataloader.num_workers,
            pin_memory=cfg.data.dataloader.pin_memory,
            persistent_workers=cfg.data.dataloader.persistent_workers and cfg.data.dataloader.num_workers > 0,
        )

        val_loader = DataLoader(
            datasets["val"],
            batch_size=cfg.train.eval_batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for eval to avoid issues
            pin_memory=cfg.data.dataloader.pin_memory,
        )

        test_loader = DataLoader(
            datasets["test"],
            batch_size=cfg.train.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=cfg.data.dataloader.pin_memory,
        )

        # Create model (FR-003)
        logger.info("Creating model...")
        model = DeBERTaV3Classifier.from_pretrained(
            cfg.model.checkpoint,
            num_labels=2,
            num_layers=cfg.model.head.num_layers,
            hidden_dims=cfg.model.head.hidden_dims,
            activation=cfg.model.head.activation,
            dropout=cfg.model.head.dropout,
            pooling_strategy=cfg.model.head.pooling_strategy,
            attn_implementation=cfg.train.optimizations.attention_implementation,
        )
        model = model.to(device)

        # Enable gradient checkpointing if requested
        if cfg.train.optimizations.gradient_checkpointing:
            model.deberta.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing enabled")

        # Compile model if requested
        if cfg.train.optimizations.use_compile:
            logger.info(f"Compiling model with mode={cfg.train.optimizations.compile_mode}...")
            model = torch.compile(model, mode=cfg.train.optimizations.compile_mode)

        # Dry-run validation (FR-008)
        dry_run_batch(model, train_loader, device, cfg)

        # Create optimizer and scheduler (FR-004)
        optimizer = create_optimizer(model, cfg)

        num_training_steps = len(train_loader) * cfg.train.epochs // cfg.train.gradient_accumulation_steps
        scheduler = create_scheduler(optimizer, cfg, num_training_steps)

        # Log training hyperparameters (FR-005)
        mlflow.log_params({
            "model_checkpoint": cfg.model.checkpoint,
            "num_layers": cfg.model.head.num_layers,
            "hidden_dims": cfg.model.head.hidden_dims,
            "activation": cfg.model.head.activation,
            "dropout": cfg.model.head.dropout,
            "pooling_strategy": cfg.model.head.pooling_strategy,
            "learning_rate": cfg.train.learning_rate,
            "batch_size": cfg.train.batch_size,
            "eval_batch_size": cfg.train.eval_batch_size,
            "epochs": cfg.train.epochs,
            "warmup_ratio": cfg.train.warmup_ratio,
            "weight_decay": cfg.train.weight_decay,
            "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
            "scheduler": cfg.train.scheduler,
            "optimizer_type": cfg.train.optimizer_type,
            "max_length": cfg.model.tokenizer.max_length,
            "seed": cfg.train.seed,
        })

        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, cfg.train.epochs + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch}/{cfg.train.epochs}")
            logger.info(f"{'='*80}")

            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, cfg, epoch)

            # Log training metrics (FR-005)
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value, step=epoch)

            # Evaluate
            val_metrics, val_preds, val_labels = evaluate(model, val_loader, device, cfg)

            # Log validation metrics
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value, step=epoch)

            logger.info(
                f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                f"Val Loss={val_metrics['loss']:.4f}, Val F1={val_metrics['macro_f1']:.4f}"
            )

            # Early stopping
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                patience_counter = 0

                # Save best model
                model_path = Path("best_model")
                model_path.mkdir(exist_ok=True)
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)

                logger.info(f"✓ New best model saved (F1={best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= cfg.train.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

        # Test set evaluation
        logger.info("\n" + "=" * 80)
        logger.info("Final Test Set Evaluation")
        logger.info("=" * 80)

        # Load best model
        model = DeBERTaV3Classifier.from_pretrained(
            "best_model",
            num_labels=2,
            num_layers=cfg.model.head.num_layers,
            hidden_dims=cfg.model.head.hidden_dims,
            activation=cfg.model.head.activation,
            dropout=cfg.model.head.dropout,
            pooling_strategy=cfg.model.head.pooling_strategy,
        )
        model = model.to(device)

        test_metrics, test_preds, test_labels = evaluate(model, test_loader, device, cfg)

        # Log test metrics
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        logger.info(f"Test Results: {test_metrics}")

        # Log artifacts (FR-005, FR-012)
        # Confusion matrix and classification report are logged in evaluate()

        # Log best model
        mlflow.pytorch.log_model(model, "model")

        logger.info("\n✓ Training complete!")
        logger.info(f"Best Val F1: {best_val_f1:.4f}")
        logger.info(f"Test F1: {test_metrics['macro_f1']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
