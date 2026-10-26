"""Offline checks for metadata-only registration; no AWS SDK or account needed."""

import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from scripts import register_sagemaker_model as registration


class FakeClientError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}}


class FakeBotoCoreError(Exception):
    pass


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.checkpoint = root / "final_model.zip"
        self.normalization = root / "vecnorm_stats.pkl"
        self.checkpoint.write_bytes(b"local checkpoint bytes")
        self.normalization.write_bytes(b"local normalization bytes")
        self.arguments = [
            "--region", "eu-central-1", "--checkpoint", str(self.checkpoint),
            "--normalization", str(self.normalization),
        ]
        snapshot = patch.object(registration, "_registration_snapshot", return_value={
            "registration_git_revision": "a" * 40,
            "registration_git_worktree_status": "clean",
        })
        snapshot.start()
        self.addCleanup(snapshot.stop)

    def run_with_fake_aws(self, client):
        session = Mock()
        session.client.return_value = client
        boto3 = types.ModuleType("boto3")
        boto3.Session = Mock(return_value=session)
        botocore = types.ModuleType("botocore")
        exceptions = types.ModuleType("botocore.exceptions")
        exceptions.ClientError = FakeClientError
        exceptions.BotoCoreError = FakeBotoCoreError
        botocore.exceptions = exceptions
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict("sys.modules", {
            "boto3": boto3, "botocore": botocore,
            "botocore.exceptions": exceptions,
        }), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = registration.main(self.arguments)
        session.client.assert_called_once_with("sagemaker")
        return status, output.getvalue(), errors.getvalue()

    def make_client(self):
        client = Mock(spec=[
            "describe_model_package_group", "create_model_package_group",
            "create_model_package",
        ])
        client.create_model_package.return_value = {
            "ModelPackageArn": "arn:aws:sagemaker:eu-central-1:123456789012:model-package/example/1",
        }
        return client

    def test_dry_run_hashes_local_files_without_importing_aws_sdk(self):
        original_import = builtins.__import__

        def block_sdk(name, *args, **kwargs):
            if name.split(".")[0] in ("boto3", "botocore"):
                self.fail("Dry run must not import the AWS SDK")
            return original_import(name, *args, **kwargs)

        output = io.StringIO()
        with patch("builtins.__import__", side_effect=block_sdk), contextlib.redirect_stdout(output):
            status = registration.main(self.arguments + ["--dry-run"])
        self.assertEqual(status, 0)
        plan = json.loads(output.getvalue())
        package = plan["create_model_package"]
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.assertEqual(package["SourceUri"], "urn:rlswan:sha256:" + digest)
        self.assertEqual(package["CustomerMetadataProperties"]["artifact_location"], "local-only")
        self.assertEqual(package["ModelApprovalStatus"], "PendingManualApproval")
        self.assertNotIn("InferenceSpecification", package)
        self.assertNotIn("ValidationSpecification", package)
        self.assertNotIn("ManagedConfiguration", plan["create_model_package_group_if_missing"])

    def test_existing_group_only_describes_and_registers_metadata(self):
        client = self.make_client()
        status, output, errors = self.run_with_fake_aws(client)
        self.assertEqual((status, errors), (0, ""))
        self.assertEqual([entry[0] for entry in client.mock_calls], [
            "describe_model_package_group", "create_model_package",
        ])
        self.assertFalse(json.loads(output)["model_group_created"])
        package = client.create_model_package.call_args.kwargs
        self.assertNotIn("InferenceSpecification", package)
        self.assertNotIn("ValidationSpecification", package)

    def test_explicitly_missing_group_is_created_before_registration(self):
        client = self.make_client()
        client.describe_model_package_group.side_effect = FakeClientError(
            "ValidationException",
            "Model Package Group RLSWAN-context-ppo-catalog does not exist.",
        )
        status, output, errors = self.run_with_fake_aws(client)
        self.assertEqual((status, errors), (0, ""))
        self.assertEqual([entry[0] for entry in client.mock_calls], [
            "describe_model_package_group", "create_model_package_group",
            "create_model_package",
        ])
        self.assertTrue(json.loads(output)["model_group_created"])

    def test_permissions_or_other_validation_failures_do_not_create_resources(self):
        for code, message in (
            ("AccessDeniedException", "Not authorized"),
            ("ValidationException", "Invalid model package group name"),
            ("ValidationException", "Model Package Group another-group does not exist."),
        ):
            with self.subTest(code=code, message=message):
                client = self.make_client()
                client.describe_model_package_group.side_effect = FakeClientError(code, message)
                status, output, errors = self.run_with_fake_aws(client)
                self.assertEqual(status, 1)
                self.assertEqual(output, "")
                self.assertIn(message, errors)
                client.create_model_package_group.assert_not_called()
                client.create_model_package.assert_not_called()

    def test_missing_normalization_fails_before_any_registration(self):
        self.normalization.unlink()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(registration.main(self.arguments + ["--dry-run"]), 2)

    def test_retry_token_is_stable_and_changes_with_normalization_bytes(self):
        args = registration._parser().parse_args(self.arguments)
        first = registration.build_registration_plan(args)["create_model_package"]
        second = registration.build_registration_plan(args)["create_model_package"]
        self.assertEqual(first["ClientToken"], second["ClientToken"])
        self.assertRegex(first["ClientToken"], r"^[A-Za-z0-9-]{1,36}$")
        self.normalization.write_bytes(b"changed normalization bytes")
        changed = registration.build_registration_plan(args)["create_model_package"]
        self.assertNotEqual(first["ClientToken"], changed["ClientToken"])


if __name__ == "__main__":
    unittest.main()
