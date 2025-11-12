#!/usr/bin/env python3
"""
Hyperparameter Optimization script using Optuna for DeBERTaV3 Criteria Classifier.

Implements:
- FR-007: Optuna hyperparameter search with MedianPruner
- FR-018: Wide search space for head architecture and training hyperparameters
- US2: HPO workflow with MLflow logging
- Full lineage tracking for each trial
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
import mlflow
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add src to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from Project.SubProject.data.dataset import build_criterion_dataset
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


def suggest_hyperparameters(trial: optuna.Trial, cfg: DictConfig) -> dict:
    """Suggest hyperparameters from configured search space (FR-018)."""
    ss = cfg.hpo.search_space
    params = {}

    # Classification head architecture
    params["head_layers"] = trial.suggest_int("head_layers", ss.head_layers.low, ss.head_layers.high)

    if params["head_layers"] > 0:
        params["hidden_dim"] = trial.suggest_int(
            "hidden_dim", ss.hidden_dim.low, ss.hidden_dim.high, log=ss.hidden_dim.log
        )
    else:
        params["hidden_dim"] = None  # Direct projection

    params["pooling_strategy"] = trial.suggest_categorical(
        "pooling_strategy", ss.pooling_strategy.choices
    )
    params["activation"] = trial.suggest_categorical("activation", ss.activation.choices)
    params["dropout"] = trial.suggest_float("dropout", ss.dropout.low, ss.dropout.high)

    # Training hyperparameters
    params["learning_rate"] = trial.suggest_float(
        "learning_rate", ss.learning_rate.low, ss.learning_rate.high, log=ss.learning_rate.log
    )
    params["batch_size"] = trial.suggest_categorical("batch_size", ss.batch_size.choices)
    params["warmup_ratio"] = trial.suggest_float("warmup_ratio", ss.warmup_ratio.low, ss.warmup_ratio.high)
    params["weight_decay"] = trial.suggest_float("weight_decay", ss.weight_decay.low, ss.weight_decay.high)
    params["scheduler"] = trial.suggest_categorical("scheduler", ss.scheduler.choices)

    return params


def create_trial_config(base_cfg: DictConfig, trial_params: dict) -> DictConfig:
    """Create trial-specific configuration by merging base config with trial params."""
    # Deep copy base config
    trial_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    # Update model head params
    trial_cfg.model.head.num_layers = trial_params["head_layers"]
    if trial_params["head_layers"] > 0:
        trial_cfg.model.head.hidden_dims = [trial_params["hidden_dim"]] * trial_params["head_layers"]
    else:
        trial_cfg.model.head.hidden_dims = []
    trial_cfg.model.head.pooling_strategy = trial_params["pooling_strategy"]
    trial_cfg.model.head.activation = trial_params["activation"]
    trial_cfg.model.head.dropout = trial_params["dropout"]

    # Update training params
    trial_cfg.train.learning_rate = trial_params["learning_rate"]
    trial_cfg.train.batch_size = trial_params["batch_size"]
    trial_cfg.train.warmup_ratio = trial_params["warmup_ratio"]
    trial_cfg.train.weight_decay = trial_params["weight_decay"]
    trial_cfg.train.scheduler = trial_params["scheduler"]

    return trial_cfg


def create_optimizer(model, cfg):
    """Create optimizer (reused from train.py)."""
    optimizer_type = cfg.train.optimizer_type
    lr = cfg.train.learning_rate
    weight_decay = cfg.train.weight_decay

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
        try:
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr, fused=True)
        except Exception:
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    elif optimizer_type == "adamw_bnb_8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=lr)
        except ImportError:
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    else:
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
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

    return scheduler


def objective(trial, base_cfg, tokenizer, datasets, device):
    """Optuna objective function for a single trial."""

    # Suggest hyperparameters
    trial_params = suggest_hyperparameters(trial, base_cfg)

    # Create trial config
    trial_cfg = create_trial_config(base_cfg, trial_params)

    # Seed for this trial
    seed_everything(base_cfg.train.seed + trial.number)

    # Start nested MLflow run for this trial
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):

        # Log lineage for this trial
        data_paths = [
            Path(trial_cfg.data.redsm5_dir),
            Path(trial_cfg.data.dsm5_criteria_path),
        ]
        log_lineage_metadata(
            trial_cfg,
            data_paths=data_paths,
            additional_tags={"stage": "hpo", "trial_number": str(trial.number)},
        )

        # Log trial hyperparameters
        mlflow.log_params({
            "trial_number": trial.number,
            **trial_params,
        })

        # Get class weights
        class_weight = datasets["train"].get_class_weight()
        class_weight_tensor = torch.tensor(class_weight, dtype=torch.float32, device=device)

        # Create dataloaders (use num_workers=0 for HPO to avoid multiprocessing issues)
        train_loader = DataLoader(
            datasets["train"],
            batch_size=trial_cfg.train.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = DataLoader(
            datasets["val"],
            batch_size=trial_cfg.train.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Create model
        model = DeBERTaV3Classifier.from_pretrained(
            trial_cfg.model.checkpoint,
            num_labels=2,
            num_layers=trial_cfg.model.head.num_layers,
            hidden_dims=trial_cfg.model.head.hidden_dims,
            activation=trial_cfg.model.head.activation,
            dropout=trial_cfg.model.head.dropout,
            pooling_strategy=trial_cfg.model.head.pooling_strategy,
            attn_implementation=trial_cfg.train.optimizations.attention_implementation,
        )
        model = model.to(device)

        # Enable gradient checkpointing if requested
        if trial_cfg.train.optimizations.gradient_checkpointing:
            model.deberta.gradient_checkpointing_enable()

        # Create optimizer and scheduler
        optimizer = create_optimizer(model, trial_cfg)

        num_training_steps = (
            len(train_loader) * trial_cfg.train.epochs // trial_cfg.train.gradient_accumulation_steps
        )
        scheduler = create_scheduler(optimizer, trial_cfg, num_training_steps)

        # Training loop with pruning
        best_val_f1 = 0.0

        for epoch in range(1, trial_cfg.train.epochs + 1):
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, trial_cfg, epoch)

            # Evaluate
            val_metrics, _, _ = evaluate(model, val_loader, device, trial_cfg)

            # Update best
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]

            # Log intermediate metrics
            mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)
            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("val_macro_f1", val_metrics["macro_f1"], step=epoch)
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)

            # Report to Optuna for pruning (FR-007: MedianPruner)
            trial.report(val_metrics["macro_f1"], epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                mlflow.set_tag("trial_status", "pruned")
                raise optuna.TrialPruned()

        # Log final best metric
        mlflow.log_metric("best_val_macro_f1", best_val_f1)
        mlflow.set_tag("trial_status", "completed")

        logger.info(f"Trial {trial.number} completed with best F1={best_val_f1:.4f}")

        return best_val_f1


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main HPO function."""
    setup_logging(cfg)

    logger.info("=" * 80)
    logger.info("Hyperparameter Optimization with Optuna")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))

    # Seed
    seed_everything(cfg.train.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Enable TF32 if requested
    if cfg.train.optimizations.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✓ TF32 enabled")

    # Configure MLflow
    configure_mlflow(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment=cfg.mlflow.experiment_name,
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)

    # Build datasets ONCE (reuse across trials)
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

    # Create Optuna study (FR-007: MedianPruner)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=cfg.hpo.pruner.n_startup_trials,
        n_warmup_steps=cfg.hpo.pruner.get("n_warmup_steps", 0),
    )

    sampler = optuna.samplers.TPESampler(seed=cfg.train.seed)

    study = optuna.create_study(
        study_name=cfg.hpo.study_name,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
    )

    # Start parent MLflow run
    with mlflow.start_run(run_name=f"hpo-{cfg.hpo.study_name}"):

        # Log parent run metadata
        data_paths = [
            Path(cfg.data.redsm5_dir),
            Path(cfg.data.dsm5_criteria_path),
        ]
        log_lineage_metadata(
            cfg,
            data_paths=data_paths,
            additional_tags={"stage": "hpo_parent", "study_name": cfg.hpo.study_name},
        )

        mlflow.log_params({
            "n_trials": cfg.hpo.n_trials,
            "study_name": cfg.hpo.study_name,
            "pruner_type": cfg.hpo.pruner.type,
            "sampler_type": "TPESampler",
        })

        # Run optimization
        logger.info(f"\nStarting {cfg.hpo.n_trials} trials...")
        study.optimize(
            lambda trial: objective(trial, cfg, tokenizer, datasets, device),
            n_trials=cfg.hpo.n_trials,
            show_progress_bar=True,
        )

        # Log study results
        logger.info("\n" + "=" * 80)
        logger.info("HPO Results")
        logger.info("=" * 80)
        logger.info(f"Number of finished trials: {len(study.trials)}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value (macro-F1): {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_trial.params}")

        # Log best trial info
        mlflow.log_metric("best_trial_number", study.best_trial.number)
        mlflow.log_metric("best_trial_value", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})

        # Save best trial config
        best_trial_params = study.best_trial.params
        best_trial_cfg = create_trial_config(cfg, best_trial_params)

        best_config_path = Path("best_trial_config.yaml")
        OmegaConf.save(best_trial_cfg, best_config_path)
        mlflow.log_artifact(str(best_config_path), artifact_path="config")
        logger.info(f"✓ Best trial config saved to {best_config_path}")

        # Generate and log visualizations
        try:
            import matplotlib.pyplot as plt

            # Optimization history
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig.savefig("optimization_history.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact("optimization_history.png")
            plt.close(fig)

            # Parameter importances
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            fig.savefig("param_importances.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact("param_importances.png")
            plt.close(fig)

            logger.info("✓ Visualizations generated and logged")

        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")

        logger.info("\n✓ HPO complete!")


if __name__ == "__main__":
    main()
