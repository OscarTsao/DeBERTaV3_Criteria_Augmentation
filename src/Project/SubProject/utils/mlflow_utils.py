from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def configure_mlflow(
    tracking_uri: Optional[str] = None,
    experiment: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Configure MLflow tracking URI and experiment.

    - `tracking_uri`: e.g. "file:./mlruns" or a remote server URI
    - `experiment`: experiment name (created if missing)
    - `tags`: default tags set for the active run (if any)
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)
    if tags:
        try:
            mlflow.set_tags(tags)
        except Exception:
            # set_tags requires an active run; ignore if none is active
            pass


def enable_autologging(enable: bool = True) -> None:
    """Enable or disable MLflow autologging with sensible defaults.

    Attempts framework-specific autologging when available and falls back
    to generic `mlflow.autolog`.
    """
    import mlflow

    if not enable:
        mlflow.autolog(disable=True)
        return

    # Prefer generic autolog (works for many frameworks in MLflow>=2)
    try:
        mlflow.autolog()
        return
    except Exception:
        pass

    # Fall back to common framework-specific autologging if present
    for mod_name in ("mlflow.pytorch", "mlflow.sklearn", "mlflow.xgboost", "mlflow.lightgbm"):
        try:
            mod = __import__(mod_name, fromlist=["autolog"])  # type: ignore
            getattr(mod, "autolog")()
        except Exception:
            continue


@contextlib.contextmanager
def mlflow_run(
    name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Context manager that starts and ends an MLflow run.

    Usage:
        with mlflow_run("demo", tags={"stage": "dev"}):
            ... your training loop ...
    """
    import mlflow

    with mlflow.start_run(run_name=name, nested=nested) as run:
        if tags:
            try:
                mlflow.set_tags(tags)
            except Exception:
                pass
        if params:
            try:
                mlflow.log_params(params)
            except Exception:
                pass
        yield run


# ============================================================================
# Lineage Tracking & Reproducibility (FR-009, US3)
# ============================================================================


def get_git_sha() -> Optional[str]:
    """Get current git commit SHA.

    Returns:
        7-character short SHA if in a git repo, otherwise None.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("Could not retrieve git SHA (not a git repo or git not available)")
        return None


def get_git_branch() -> Optional[str]:
    """Get current git branch name.

    Returns:
        Branch name if in a git repo, otherwise None.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_git_dirty() -> bool:
    """Check if git working tree has uncommitted changes.

    Returns:
        True if there are uncommitted changes, False otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode != 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file.

    Returns:
        Hex digest of file hash (first 16 characters).
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]
    except Exception as e:
        logger.warning(f"Could not hash {file_path}: {e}")
        return "unknown"


def compute_dataset_hashes(data_paths: List[Path]) -> Dict[str, str]:
    """Compute hashes for dataset files.

    Args:
        data_paths: List of paths to dataset files.

    Returns:
        Dict mapping filename to hash.
    """
    hashes = {}
    for path in data_paths:
        if path.exists() and path.is_file():
            hashes[path.name] = compute_file_hash(path)
        elif path.exists() and path.is_dir():
            # Hash all files in directory
            for file_path in sorted(path.glob("**/*")):
                if file_path.is_file():
                    rel_path = file_path.relative_to(path.parent)
                    hashes[str(rel_path)] = compute_file_hash(file_path)
    return hashes


def get_software_versions() -> Dict[str, str]:
    """Collect versions of key software dependencies.

    Returns:
        Dict mapping package name to version string.
    """
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }

    # Optional dependencies
    optional_packages = [
        "transformers",
        "datasets",
        "omegaconf",
        "hydra",
        "optuna",
        "mlflow",
        "scikit-learn",
        "numpy",
        "pandas",
    ]

    for package in optional_packages:
        try:
            mod = __import__(package)
            versions[package] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[package] = "not installed"

    return versions


def extract_optimization_flags(cfg: DictConfig) -> Dict[str, Any]:
    """Extract optimization flag settings from config.

    Args:
        cfg: Hydra config.

    Returns:
        Dict of optimization flags and their values.
    """
    flags = {}

    if "train" in cfg and "optimizations" in cfg.train:
        opt_cfg = cfg.train.optimizations
        flags["use_amp"] = opt_cfg.get("use_amp", False)
        flags["amp_dtype"] = opt_cfg.get("amp_dtype", "bf16")
        flags["enable_tf32"] = opt_cfg.get("enable_tf32", True)
        flags["attention_implementation"] = opt_cfg.get("attention_implementation", "sdpa")
        flags["gradient_checkpointing"] = opt_cfg.get("gradient_checkpointing", False)
        flags["use_compile"] = opt_cfg.get("use_compile", False)
        flags["compile_mode"] = opt_cfg.get("compile_mode", "default")

    if "train" in cfg:
        flags["optimizer_type"] = cfg.train.get("optimizer_type", "adamw")
        flags["gradient_accumulation_steps"] = cfg.train.get("gradient_accumulation_steps", 1)

    if "data" in cfg and "dataloader" in cfg.data:
        dl_cfg = cfg.data.dataloader
        flags["num_workers"] = dl_cfg.get("num_workers", 0)
        flags["pin_memory"] = dl_cfg.get("pin_memory", True)
        flags["persistent_workers"] = dl_cfg.get("persistent_workers", False)

    return flags


def log_config_artifact(cfg: DictConfig, artifact_name: str = "config.yaml") -> None:
    """Save resolved Hydra config as MLflow artifact.

    Args:
        cfg: Hydra config to save.
        artifact_name: Name for the artifact file.
    """
    import mlflow
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(cfg, f)
            temp_path = f.name

        mlflow.log_artifact(temp_path, artifact_path="config")
        os.unlink(temp_path)
        logger.info(f"Logged config artifact: {artifact_name}")
    except Exception as e:
        logger.error(f"Failed to log config artifact: {e}")


def log_lineage_metadata(
    cfg: DictConfig,
    data_paths: Optional[List[Path]] = None,
    additional_tags: Optional[Dict[str, str]] = None,
) -> None:
    """Log comprehensive lineage metadata for reproducibility (FR-009, US3).

    This function logs:
    - Git commit SHA, branch, and dirty status
    - Software versions (Python, PyTorch, transformers, etc.)
    - Optimization flags from config
    - Dataset file hashes
    - Resolved Hydra config artifact

    Args:
        cfg: Hydra config.
        data_paths: List of dataset file/directory paths to hash.
        additional_tags: Additional tags to log.
    """
    import mlflow

    try:
        # Git metadata
        git_sha = get_git_sha()
        git_branch = get_git_branch()
        git_dirty = check_git_dirty()

        if git_sha:
            mlflow.set_tag("git.commit", git_sha)
        if git_branch:
            mlflow.set_tag("git.branch", git_branch)
        mlflow.set_tag("git.dirty", str(git_dirty))

        # Software versions
        versions = get_software_versions()
        for package, version in versions.items():
            mlflow.set_tag(f"version.{package}", version)

        # Optimization flags
        opt_flags = extract_optimization_flags(cfg)
        for flag, value in opt_flags.items():
            mlflow.log_param(f"opt.{flag}", value)

        # Dataset hashes
        if data_paths:
            dataset_hashes = compute_dataset_hashes(data_paths)
            for filename, file_hash in dataset_hashes.items():
                mlflow.set_tag(f"data.hash.{filename}", file_hash)

        # Additional tags
        if additional_tags:
            mlflow.set_tags(additional_tags)

        # Log resolved config artifact
        log_config_artifact(cfg)

        logger.info("Logged lineage metadata successfully")

    except Exception as e:
        logger.error(f"Failed to log lineage metadata: {e}")


def get_run_config(run_id: str) -> Optional[DictConfig]:
    """Download and load config artifact from an MLflow run.

    Args:
        run_id: MLflow run ID.

    Returns:
        Loaded Hydra config, or None if not found.
    """
    import mlflow

    try:
        client = mlflow.tracking.MlflowClient()
        artifact_path = client.download_artifacts(run_id, "config/config.yaml")
        cfg = OmegaConf.load(artifact_path)
        logger.info(f"Loaded config from run {run_id}")
        return cfg
    except Exception as e:
        logger.error(f"Could not load config from run {run_id}: {e}")
        return None

