from __future__ import annotations

import sys
import types
import unittest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.modules.setdefault("pretty_midi", types.ModuleType("pretty_midi"))

spec = importlib.util.spec_from_file_location("hybrid_under_test", ROOT / "src/minimt3/amt/hybrid.py")
assert spec is not None and spec.loader is not None
hybrid_under_test = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hybrid_under_test
spec.loader.exec_module(hybrid_under_test)

HybridRescueConfig = hybrid_under_test.HybridRescueConfig
hybrid_rescue_notes = hybrid_under_test.hybrid_rescue_notes
resolve_hybrid_preset = hybrid_under_test.resolve_hybrid_preset
from minimt3.symbolic.events import NoteEvent


class HybridRescueTest(unittest.TestCase):
    def test_extends_duplicate_long_note_without_adding_note(self) -> None:
        base = [NoteEvent(45, 0.0, 1.0, 80)]
        assistant = [NoteEvent(45, 0.02, 2.0, 92)]
        cfg = HybridRescueConfig(
            enabled=True,
            max_added_ratio=0.0,
            extend_duplicate_long_notes=True,
            extension_pitch_max=64,
            extension_min_duration_seconds=0.5,
            extension_min_gain_seconds=0.2,
            extension_max_gain_seconds=2.0,
        )

        notes, stats = hybrid_rescue_notes(base, assistant, duration=4.0, config=cfg)

        self.assertEqual(len(notes), 1)
        self.assertAlmostEqual(notes[0].end, 2.0)
        self.assertEqual(stats["hybrid_extended_long_notes"], 1.0)
        self.assertEqual(stats["hybrid_added_notes"], 0.0)

    def test_adds_context_supported_chord_tone(self) -> None:
        base = [NoteEvent(60, 1.0, 1.5, 80), NoteEvent(64, 1.01, 1.5, 78)]
        assistant = [NoteEvent(67, 1.02, 1.5, 85)]
        cfg = HybridRescueConfig(enabled=True, max_added_ratio=1.0, max_added_per_second=4.0)

        notes, stats = hybrid_rescue_notes(base, assistant, duration=4.0, config=cfg)

        self.assertEqual([note.pitch for note in notes], [60, 64, 67])
        self.assertEqual(stats["hybrid_added_chord_notes"], 1.0)

    def test_rejects_isolated_short_candidate(self) -> None:
        base = [NoteEvent(60, 1.0, 1.4, 80)]
        assistant = [NoteEvent(72, 2.0, 2.04, 90)]
        cfg = HybridRescueConfig(enabled=True, max_added_ratio=1.0, max_added_per_second=4.0)

        notes, stats = hybrid_rescue_notes(base, assistant, duration=4.0, config=cfg)

        self.assertEqual(len(notes), 1)
        self.assertEqual(stats["hybrid_rejected_isolated_short"], 1.0)

    def test_preset_alias_resolves_to_hybrid_score(self) -> None:
        alias = resolve_hybrid_preset("display_chord_long")
        score = resolve_hybrid_preset("hybrid_score")

        self.assertEqual(alias.to_json(), score.to_json())


if __name__ == "__main__":
    unittest.main()
