# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Offline caption-option checks without importing CUDA or model dependencies.

These tests execute selected production definitions from their AST, injecting
only external dependencies. No API requests, model downloads, or training occur.
"""

import argparse
import ast
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import re
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch
import warnings


REPO = Path(__file__).resolve().parents[2]


def load_definitions(path, names, **dependencies):
    """Compile actual source definitions without their heavyweight imports."""
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    selected = []
    found = set()
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        if name in names:
            selected.append(node)
            found.add(name)
    if found != set(names):
        raise AssertionError(f"Missing production definitions in {path}: {set(names) - found}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    namespace = dict(
        argparse=argparse, json=json, Path=Path, re=re, warnings=warnings,
        __name__="caption_test_production", **dependencies,
    )
    exec(compile(module, str(REPO / path), "exec"), namespace)
    return namespace


class TestCaptionOptions(unittest.TestCase):
    def setUp(self):
        self.main = load_definitions(
            "main.py", ["get_args_parser", "_load_dataset_config", "_validate_runtime_args"]
        )
        self.parser = self.main["get_args_parser"]()

    def test_cli_defaults_and_model_selection(self):
        default = self.parser.parse_args([])
        self.assertFalse(default.augment_train)
        self.assertEqual(default.caption_model, "gpt-4o")
        enabled = self.parser.parse_args(
            ["--augment_train", "--caption_model", "custom-caption-model"]
        )
        self.assertTrue(enabled.augment_train)
        self.assertEqual(enabled.caption_model, "custom-caption-model")
        self.assertFalse(self.parser.parse_args(["--no_augment_train"]).augment_train)

    def test_enable_and_disable_flags_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.parser.parse_args(["--augment_train", "--no_augment_train"])
        self.assertEqual(error.exception.code, 2)

    def test_json_cannot_enable_augmentation_or_replace_default_model(self):
        args = self.parser.parse_args(["--dataset_config", "dataset.json"])
        config = {"augment_train": True, "caption_model": "json-model", "batch_size": 17}
        with patch("builtins.open", mock_open(read_data=json.dumps(config))):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.main["_load_dataset_config"](args)
        self.assertFalse(args.augment_train)
        self.assertEqual(args.caption_model, "gpt-4o")
        self.assertEqual(args.batch_size, 17)
        self.assertTrue(caught, "Ignoring caption config keys must be visible to the caller")

    def test_json_cannot_disable_explicit_cli_augmentation_or_change_model(self):
        args = self.parser.parse_args([
            "--dataset_config", "dataset.json", "--augment_train",
            "--caption_model", "cli-model", "--batch_size", "2",
        ])
        config = {"augment_train": False, "caption_model": "json-model", "batch_size": 13}
        with patch("builtins.open", mock_open(read_data=json.dumps(config))):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                self.main["_load_dataset_config"](args)
        self.assertTrue(args.augment_train)
        self.assertEqual(args.caption_model, "cli-model")
        self.assertEqual(args.batch_size, 13)  # Other JSON precedence is unchanged.

    def test_no_dataset_config_does_not_open_a_file(self):
        with patch("builtins.open") as opened:
            self.main["_load_dataset_config"](self.parser.parse_args([]))
        opened.assert_not_called()

    def test_empty_model_rejected_only_when_augmentation_is_enabled(self):
        args = self.parser.parse_args(["--augment_train", "--caption_model", "   "])
        with self.assertRaisesRegex(ValueError, "caption_model"):
            self.main["_validate_runtime_args"](args)
        args.augment_train = False
        self.main["_validate_runtime_args"](args)


class TestCaptionDatasetWiring(unittest.TestCase):
    def build(self, storage, split, **options):
        sources = SimpleNamespace(
            dataset_format=storage, csv_key="captions.csv", images_prefix="images",
            labels_prefix="labels",
        )
        factory = Mock(return_value=SimpleNamespace(ids_train=[0], ids_val=[1], ids_all=[0, 1]))
        transforms = Mock(return_value=object())
        wrapper = Mock()
        ns = load_definitions(
            "geo_datasets/dota_preprocess.py", ["build_dota", "_get_arg_float", "_get_arg_int"],
            _sources_from_args=Mock(return_value=sources), DotaModulatedDetection=factory,
            _dota_transforms=transforms, _WrappedDataset=wrapper, print=Mock(),
        )
        args = SimpleNamespace(tokenizer=object(), caption_model="selected-model", **options)
        ns["build_dota"](split, args)
        return factory.call_args.kwargs, transforms, wrapper

    def test_both_storage_modes_default_off_even_without_new_args(self):
        for storage in ("local", "s3"):
            with self.subTest(storage=storage):
                kwargs, transforms, _ = self.build(storage, "train")
                self.assertFalse(kwargs["augment_train"])
                self.assertEqual(kwargs["caption_model"], "selected-model")
                transforms.assert_called_once_with("train")

    def test_explicit_opt_in_and_image_transforms_for_both_storage_modes(self):
        for storage in ("local", "s3"):
            for enabled in (False, True):
                with self.subTest(storage=storage, enabled=enabled):
                    kwargs, transforms, wrapper = self.build(
                        storage, "train", augment_train=enabled
                    )
                    self.assertEqual(kwargs["augment_train"], enabled)
                    transforms.assert_called_once_with("train")
                    self.assertIs(wrapper.call_args.args[1], transforms.return_value)

    def test_validation_and_test_never_augment(self):
        for storage in ("local", "s3"):
            for split in ("val", "test"):
                with self.subTest(storage=storage, split=split):
                    kwargs, transforms, _ = self.build(storage, split, augment_train=True)
                    self.assertFalse(kwargs["augment_train"])
                    transforms.assert_called_once_with("val")

    def test_dataset_stores_model_but_does_not_augment_when_disabled(self):
        ns = load_definitions(
            "geo_datasets/dota_preprocess.py", ["DotaModulatedDetection"],
            Dataset=object, _ConvertDotaToTarget=Mock(), COCO=Mock(), _build_coco_like=Mock(),
        )
        dataset_type = ns["DotaModulatedDetection"]
        records = {0: {"caption": "A ship."}, 1: {"caption": "A plane."}}
        with patch.object(dataset_type, "_build_annotations", return_value=records):
            with patch.object(dataset_type, "_split_groups", return_value=([0], [1])):
                with patch.object(dataset_type, "_augment_captions_once") as augment:
                    with patch.object(dataset_type, "_explode_captions_no_gpt", return_value=[]) as explode:
                        dataset = dataset_type(
                            sources=SimpleNamespace(dataset_format="local"),
                            augment_train=False, caption_model="selected-model",
                        )
        self.assertEqual(dataset.caption_model, "selected-model")
        augment.assert_not_called()
        explode.assert_called_once()

    def test_existing_s3_cache_is_reused_without_generation_or_model(self):
        records = {0: {"caption": "A ship."}, 1: {"caption": "A plane."}}
        cache = Mock()
        cache.exists.return_value = True
        pq = ModuleType("pyarrow.parquet")
        pq.read_table = Mock(return_value=object())
        arrow = ModuleType("pyarrow")
        arrow.parquet = pq
        ns = load_definitions(
            "geo_datasets/dota_preprocess.py", ["DotaModulatedDetection"],
            Dataset=object, _ConvertDotaToTarget=Mock(), COCO=Mock(), _build_coco_like=Mock(),
            ARROW_CACHE=cache, CLASSES_JSON=cache, _table_to_dict=Mock(return_value=records),
        )
        dataset_type = ns["DotaModulatedDetection"]
        with patch.dict("sys.modules", {"pyarrow": arrow, "pyarrow.parquet": pq}):
            with patch.object(dataset_type, "_build_annotations") as build:
                with patch.object(dataset_type, "_split_groups", return_value=([0], [1])):
                    with patch.object(dataset_type, "_augment_captions_once") as augment:
                        with patch.object(dataset_type, "_explode_captions_no_gpt") as explode:
                            for enabled in (False, True):
                                dataset = dataset_type(
                                    sources=SimpleNamespace(dataset_format="s3"),
                                    augment_train=enabled,
                                )
                                self.assertIs(dataset.annotations, records)
        build.assert_not_called()
        augment.assert_not_called()
        explode.assert_not_called()
        self.assertEqual(pq.read_table.call_count, 2)


class TestCaptionGeneration(unittest.TestCase):
    def load_api(self, client):
        return load_definitions(
            "geo_datasets/dota_preprocess.py", ["_call_gpt", "_CAP_PROMPT"],
            _openai=client, _OPENAI_INITIALIZED=True, _OPENAI_WARNED=False, print=Mock(),
        )

    def test_selected_model_reaches_api_and_response_is_trimmed(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A ship.\nA plane."))]
        )
        ns = self.load_api(client)
        self.assertEqual(ns["_call_gpt"](["A ship is docked."], 1, model="custom-model"), ["A ship."])
        self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "custom-model")

    def test_api_failure_remains_best_effort(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("offline")
        ns = self.load_api(client)
        self.assertEqual(ns["_call_gpt"](["A ship."], 1, model="custom-model"), [])
        self.assertTrue(ns["_OPENAI_WARNED"])

    def test_no_needed_captions_does_not_initialize_client(self):
        ns = self.load_api(None)
        ns["_OPENAI_INITIALIZED"] = False
        self.assertEqual(ns["_call_gpt"](["A ship."], 0, model="custom-model"), [])
        self.assertFalse(ns["_OPENAI_INITIALIZED"])
        self.assertIsNone(ns["_openai"])

    def test_augmentation_passes_model_and_keeps_other_split_untouched(self):
        generated = ["A ship at a pier.", "A ship by a pier.", "A ship near a pier.", "A ship is docked."]
        call = Mock(return_value=generated)
        ns = load_definitions(
            "geo_datasets/dota_preprocess.py",
            ["DotaModulatedDetection", "_split", "_SPLIT_RE"], Dataset=object, _call_gpt=call,
        )
        dataset = object.__new__(ns["DotaModulatedDetection"])
        dataset.caption_model = "chosen-caption-model"
        records = {
            0: {"caption": "A ship beside a pier.", "image_path": "ship.png", "annotations": [{"bbox": [1, 2, 3, 4]}]},
            1: {"caption": "A plane on a runway.", "image_path": "plane.png", "annotations": []},
        }
        dataset._augment_captions_once(records, only_ids={0})
        call.assert_called_once()
        self.assertEqual(call.call_args.kwargs["model"], "chosen-caption-model")
        self.assertEqual(len(records), 6)
        self.assertEqual(records[1]["caption"], "A plane on a runway.")
        for key in range(2, 6):
            self.assertEqual(records[key]["image_path"], "ship.png")
            self.assertEqual(records[key]["annotations"], records[0]["annotations"])
            self.assertIsNot(records[key]["annotations"], records[0]["annotations"])


if __name__ == "__main__":
    unittest.main()
