#!/usr/bin/env python3
"""Catalog a local checkpoint in SageMaker Model Registry without uploading it.

Only model-package metadata is sent to AWS. The checkpoint and normalization
files stay local, and the registered version has no inference specification.
Use --dry-run to inspect the requests without boto3, credentials, or network.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence, Tuple
import unicodedata


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("artifacts/models/context_ppo_seed0/final_model.zip")
DEFAULT_MODEL_GROUP = "RLSWAN-context-ppo-catalog"


def _artifact_details(path: Path) -> Tuple[str, int]:
    """Hash bytes only; never deserialize model or pickle files."""
    if not path.is_file():
        raise ValueError("Required artifact is not a file: {}".format(path))
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _registration_snapshot() -> Dict[str, str]:
    """Describe the repository at registration time, not at training time."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL, universal_newlines=True, timeout=5,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
            universal_newlines=True, timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {
            "registration_git_revision": "unavailable",
            "registration_git_worktree_status": "unavailable",
        }
    return {
        "registration_git_revision": revision,
        "registration_git_worktree_status": "dirty" if status.strip() else "clean",
    }


def _validate_metadata(metadata: Dict[str, str]) -> None:
    """Check SageMaker's customer-metadata limits before making AWS calls."""
    punctuation = "_.:/=+-@"
    for key, value in metadata.items():
        for label, text, limit in (("key", key, 128), ("value", value, 256)):
            valid_characters = all(
                unicodedata.category(char)[0] in ("L", "Z", "N")
                or char in punctuation for char in text
            )
            if not 1 <= len(text) <= limit or not valid_characters:
                raise ValueError(
                    "Metadata {} for '{}' must be 1-{} characters using letters, "
                    "spaces, numbers, or _. : / = + - @; rename the artifact "
                    "if its filename is incompatible.".format(label, key, limit)
                )


def build_registration_plan(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the complete request preview using only local information."""
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", args.model_group):
        raise ValueError(
            "--model-group must be 1-63 letters, digits, or hyphens and "
            "must start and end with a letter or digit."
        )
    if not args.region.strip():
        raise ValueError("--region must not be empty.")

    checkpoint = (REPO_ROOT / args.checkpoint.expanduser()).resolve()
    if args.normalization is None:
        normalization = checkpoint.parent / "vecnorm_stats.pkl"
    else:
        normalization = (REPO_ROOT / args.normalization.expanduser()).resolve()
    checkpoint_sha, checkpoint_size = _artifact_details(checkpoint)
    normalization_sha, normalization_size = _artifact_details(normalization)
    metadata = {
        "registration_mode": "metadata-only",
        "artifact_location": "local-only",
        "deployable": "false",
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": str(checkpoint_size),
        "normalization_filename": normalization.name,
        "normalization_sha256": normalization_sha,
        "normalization_size_bytes": str(normalization_size),
        "provenance_scope": "registration-snapshot-only",
        "training_provenance": "not-verified",
    }
    metadata.update(_registration_snapshot())
    _validate_metadata(metadata)
    package = {
        "ModelPackageGroupName": args.model_group,
        "ModelPackageDescription": (
            "RLSWAN local model catalog entry. Artifacts remain local; no "
            "upload or inference specification. Repository metadata describes "
            "the registration snapshot, not verified training provenance."
        ),
        "ModelApprovalStatus": "PendingManualApproval",
        "SourceUri": "urn:rlswan:sha256:" + checkpoint_sha,
        "CustomerMetadataProperties": metadata,
    }
    # Derive a <=36-character token from the entire request excluding the token.
    # Identical requests can safely be retried; changed metadata gets a new token.
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"))
    package["ClientToken"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return {
        "mode": "metadata-only",
        "region": args.region,
        "profile": args.profile,
        "local_artifacts": {
            "checkpoint": str(checkpoint),
            "normalization": str(normalization),
        },
        "describe_model_package_group": {"ModelPackageGroupName": args.model_group},
        "create_model_package_group_if_missing": {
            "ModelPackageGroupName": args.model_group,
            "ModelPackageGroupDescription": (
                "RLSWAN metadata-only catalog. Model artifacts remain local."
            ),
        },
        "create_model_package": package,
    }


def _is_missing_group(error: Any, group_name: str) -> bool:
    """Distinguish a missing group from unrelated SageMaker validation errors."""
    detail = error.response.get("Error", {})
    if detail.get("Code") not in (
        "ValidationException", "ResourceNotFound", "ResourceNotFoundException",
    ):
        return False
    message = str(detail.get("Message", "")).lower()
    return (
        group_name.lower() in message
        and re.search(r"model[\s_-]*package[\s_-]*group", message) is not None
        and any(phrase in message for phrase in (
            "does not exist", "not found", "could not find", "cannot find",
            "unable to find", "not exist",
        ))
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, help="AWS Region, for example eu-central-1.")
    parser.add_argument("--profile", help="Optional AWS shared-configuration profile.")
    parser.add_argument("--model-group", default=DEFAULT_MODEL_GROUP,
                        help="Model package group name (default: %(default)s).")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Local checkpoint, relative to the repo or absolute (default: %(default)s).")
    parser.add_argument("--normalization", type=Path,
                        help="Local statistics file; defaults to vecnorm_stats.pkl beside the checkpoint.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print an offline JSON preview; do not import boto3 or contact AWS.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_registration_plan(args)
    except (OSError, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        print("Error: AWS registration requires boto3. Install it with: "
              "python -m pip install boto3", file=sys.stderr)
        return 2

    os.environ.setdefault("AWS_SDK_UA_APP_ID", "AWSSkill-SageMaker")
    try:
        session = boto3.Session(region_name=args.region, profile_name=args.profile)
        client = session.client("sagemaker")
        group_created = False
        try:
            client.describe_model_package_group(**plan["describe_model_package_group"])
        except ClientError as error:
            # ValidationException is a common, unmodeled not-found response for
            # this operation. Only an explicit missing-group message allows create.
            if not _is_missing_group(error, args.model_group):
                raise
            client.create_model_package_group(**plan["create_model_package_group_if_missing"])
            group_created = True
        result = client.create_model_package(**plan["create_model_package"])
    except (BotoCoreError, ClientError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps({
        "model_package_arn": result["ModelPackageArn"],
        "model_group": args.model_group,
        "model_group_created": group_created,
        "region": args.region,
        "mode": "metadata-only",
        "artifacts": "local-only",
        "approval_status": "PendingManualApproval",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
