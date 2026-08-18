"""Meta-tests about the test suite itself.

An `if __name__ == "__main__"` block in the middle of a file silently hides every
class defined after it from direct invocation. That has now happened twice in this
project, both times hiding the very tests written to answer a critic — so it gets
a test of its own.
"""

import ast
import sys
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))


def test_modules():
    return sorted(p for p in TESTS.glob("test_*.py") if p.name != Path(__file__).name)


class TestSuiteLayout(unittest.TestCase):
    def test_no_module_defines_tests_after_its_main_block(self):
        offenders = []
        for path in test_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            main_line = None
            for node in tree.body:
                if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                        and isinstance(node.test.left, ast.Name)
                        and node.test.left.id == "__name__"):
                    main_line = node.lineno
                elif main_line is not None and isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    offenders.append(f"{path.name}:{node.lineno} {node.name} "
                                     f"is hidden by the __main__ block at line {main_line}")
        self.assertEqual(offenders, [])

    def test_direct_invocation_sees_every_test(self):
        """Running a file directly must collect as many tests as discovery does."""
        import subprocess

        loader = unittest.TestLoader()
        for path in test_modules():
            module = loader.loadTestsFromName(f"tests.{path.stem}")
            discovered = module.countTestCases()
            proc = subprocess.run([sys.executable, str(path), "-v"], cwd=str(ROOT),
                                  capture_output=True, text=True, timeout=600)
            reported = 0
            for line in proc.stderr.splitlines():
                if line.startswith("Ran ") and " test" in line:
                    reported = int(line.split()[1])
            self.assertEqual(reported, discovered,
                             f"{path.name}: direct run collected {reported}, "
                             f"discovery collects {discovered}")

    def test_every_module_under_lume_has_a_test_module(self):
        modules = {p.stem for p in (ROOT / "lume").glob("*.py")
                   if p.stem not in ("__init__", "__main__")}
        covered = {p.stem[len("test_"):] for p in test_modules()}
        # cli is exercised through test_app's CLI tests rather than its own module.
        self.assertEqual(modules - covered - {"cli"}, set())


if __name__ == "__main__":
    unittest.main()
