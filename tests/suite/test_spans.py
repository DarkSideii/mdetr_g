# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Caption/class phrase boundaries and per-object token supervision."""

from copy import deepcopy
import importlib.util
import unittest

import torch

from .common import class_spans, fast_tokenizer, load_cases, production_maps_and_targets


class TestTokenAndObjectSupervision(unittest.TestCase):
    def test_generic_plural_compound_alias_and_word_boundaries(self):
        self.assertEqual(
            len(class_spans("Rovers and a rover, but not a roverlike shape.", "rover")), 2
        )
        self.assertEqual(len(class_spans("Two cargo-cranes and a cargo crane.", "cargo-crane")), 2)
        self.assertEqual(len(class_spans("A craft flies beside an aircraft.", "craft")), 1)
        self.assertEqual(
            len(
                class_spans(
                    "An inspection rover is next to a robot.", "inspection-rover", ["robot"]
                )
            ),
            2,
        )

    def test_production_generic_class_phrase_matcher_when_installed(self):
        if importlib.util.find_spec("geo_datasets.dota_text") is None:
            self.skipTest("Optional generic dota_text helper absent (old two-class preprocessing)")
        from geo_datasets.dota_text import class_spans as production_class_spans

        payload = load_cases()
        aliases = payload.get("aliases", {})
        for image in payload["images"]:
            for obj in image["objects"]:
                if obj.get("spans") is not None:
                    continue  # explicit user-provided character spans are an alternate contract
                args = (image["caption"], obj["class"], aliases.get(obj["class"], ()))
                self.assertEqual(
                    production_class_spans(*args),
                    class_spans(*args),
                    f"Production span extractor differs on class {obj['class']!r}",
                )

    def test_production_positive_map_per_object_is_fractional_and_exact(self):
        payload = load_cases()
        tokenizer, targets, positive_map = production_maps_and_targets(payload)
        self.assertEqual(positive_map.shape[0], sum(len(t["boxes"]) for t in targets))
        self.assertEqual(positive_map.shape[1], 256)
        self.assertTrue(
            torch.allclose(positive_map.sum(-1), torch.ones(len(positive_map)), atol=1e-4)
        )
        self.assertTrue(torch.all(positive_map[:, -1] == 0))  # no-object is never a phrase

        offset = 0
        for target in targets:
            tokenized = tokenizer(
                target["caption"],
                return_tensors="pt",
                padding="max_length",
                max_length=256,
                truncation=True,
            )
            self.assertEqual(len(target["tokens_positive"]), len(target["boxes"]))
            self.assertEqual(len(target["phrases_searched"]), len(target["boxes"]))
            for row, spans in zip(target["positive_map"], target["tokens_positive"]):
                expected = torch.zeros(256, dtype=torch.bool)
                for begin, end in spans:
                    self.assertGreater(end, begin)
                    self.assertLessEqual(end, len(target["caption"]))
                    first = tokenized.char_to_token(begin)
                    last = tokenized.char_to_token(end - 1)
                    self.assertIsNotNone(first, "Caption phrase starts outside tokenizer coverage")
                    self.assertIsNotNone(last, "Caption phrase ends outside tokenizer coverage")
                    expected[first : last + 1] = True
                self.assertTrue(
                    torch.equal(row > 0, expected),
                    "Production map must cover exactly the supplied character spans",
                )
                self.assertTrue(torch.equal((positive_map[offset] > 0), expected))
                offset += 1

        self.assertEqual(offset, len(positive_map))

    def test_criterion_rejects_swapped_or_empty_token_supervision(self):
        from mdetr.models.matcher import HungarianMatcher
        from mdetr.models.mdetr import SetCriterion

        payload = load_cases()
        tokenizer, targets, positive_map = production_maps_and_targets(payload)
        tokenized = tokenizer.batch_encode_plus(
            [t["caption"] for t in targets],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        batch, nqueries = len(targets), 12
        outputs = {
            "pred_logits": torch.randn(batch, nqueries, 256),
            "pred_boxes": torch.full((batch, nqueries, 4), 0.25),
            "tokenized": tokenized,
        }
        criterion = SetCriterion(
            255,
            HungarianMatcher(1, 5, 2),
            eos_coef=0.1,
            losses=["labels", "boxes"],
            temperature=0.07,
        )

        # Swapped rows retain valid normalization but supervise the wrong objects.
        distinct = next(
            (
                (offset, offset + i)
                for offset, t in zip(
                    [sum(len(x["boxes"]) for x in targets[:j]) for j in range(len(targets))],
                    targets,
                )
                for i in range(1, len(t["boxes"]))
                if t["phrases_searched"][i] != t["phrases_searched"][0]
            ),
            None,
        )
        if distinct is None:
            # Single-class user fixtures still need a two-class negative control.
            payload = {
                "images": [
                    {
                        "caption": "A module and a sensor are visible.",
                        "objects": [
                            {"class": "module", "box": [0.2, 0.2, 0.1, 0.1]},
                            {"class": "sensor", "box": [0.8, 0.8, 0.1, 0.1]},
                        ],
                    }
                ]
            }
            tokenizer, targets, positive_map = production_maps_and_targets(payload)
            tokenized = tokenizer.batch_encode_plus(
                [t["caption"] for t in targets],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            outputs["pred_logits"] = torch.randn(1, nqueries, 256)
            outputs["pred_boxes"] = torch.full((1, nqueries, 4), 0.25)
            outputs["tokenized"] = tokenized
            distinct = (0, 1)
        source, other = distinct
        wrong = positive_map.clone()
        wrong[source], wrong[other] = positive_map[other].clone(), positive_map[source].clone()
        with self.assertRaisesRegex(ValueError, "disagree|positive_map"):
            criterion(outputs, targets, wrong)

        empty = positive_map.clone()
        empty[0].zero_()
        with self.assertRaisesRegex(ValueError, "sum to one|positive_map"):
            criterion(outputs, targets, empty)

    def test_bad_fixture_class_span_fails_loudly(self):
        payload = deepcopy(load_cases())
        first = payload["images"][0]["objects"][0]
        first["class"] = "nonexistent-class-phrase"
        first.pop("spans", None)
        with self.assertRaisesRegex(ValueError, "Missing class phrase"):
            production_maps_and_targets(payload, tokenizer=fast_tokenizer(payload))
