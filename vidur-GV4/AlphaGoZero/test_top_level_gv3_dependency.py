from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NESTED_GV3 = "vidur.mcts.Game_Versions.Game_Version3"


class TopLevelGV3DependencyTests(unittest.TestCase):
    def test_production_runner_uses_top_level_gv3_and_sibling_cpp(self) -> None:
        runner = importlib.import_module("vidur.bellman_v4_adv.arena_mcts_value_runnerCPP")

        self.assertEqual(
            runner.tester_runner.__name__,
            "vidur.Game_Version3.Model_Tester.runner",
        )
        self.assertEqual(
            runner.nlt.__name__,
            "vidur.Game_Version3.tests.native_logger_tests",
        )
        self.assertEqual(
            Path(runner._cpp_dir()).resolve(),
            (PACKAGE_ROOT / "Game_Version3_Cpp").resolve(),
        )
        self.assertFalse(any(name.startswith(NESTED_GV3) for name in sys.modules))

    def test_executable_pipeline_files_do_not_reference_nested_gv3(self) -> None:
        roots = (
            PACKAGE_ROOT / "AlphaGoZero",
            PACKAGE_ROOT / "bellman_v4_adv",
            PACKAGE_ROOT / "Game_Version3",
        )
        offenders: list[str] = []
        for root in roots:
            for suffix in ("*.py", "*.sh"):
                for path in root.rglob(suffix):
                    if path.resolve() == Path(__file__).resolve():
                        continue
                    if NESTED_GV3 in path.read_text(encoding="utf-8", errors="replace"):
                        offenders.append(str(path.relative_to(PACKAGE_ROOT)))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
