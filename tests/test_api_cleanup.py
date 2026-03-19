from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from src.api import cleanup_expired_outputs


class TestApiCleanup(unittest.TestCase):
    def test_cleanup_expired_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            old_dir = base / "old_task"
            new_dir = base / "new_task"
            old_dir.mkdir(parents=True, exist_ok=True)
            new_dir.mkdir(parents=True, exist_ok=True)
            (old_dir / "a.txt").write_text("x", encoding="utf-8")
            (new_dir / "b.txt").write_text("y", encoding="utf-8")

            now = time.time()
            old_time = now - 25 * 3600
            new_time = now - 1 * 3600
            old_file = old_dir / "a.txt"
            new_file = new_dir / "b.txt"
            for p in (old_dir, old_file):
                os_time = (old_time, old_time)
                Path(p).touch()
                os.utime(p, os_time)
            for p in (new_dir, new_file):
                os_time = (new_time, new_time)
                Path(p).touch()
                os.utime(p, os_time)

            removed = cleanup_expired_outputs(base, retention_hours=24)
            self.assertEqual(removed, 1)
            self.assertFalse(old_dir.exists())
            self.assertTrue(new_dir.exists())


if __name__ == "__main__":
    unittest.main()
