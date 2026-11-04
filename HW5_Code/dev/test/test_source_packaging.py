"""Check source packaging from the nested course checkout without cloud calls."""
import ast
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASE = '706b1a5fc94de4bc16a4a3684eafaad6b9e0dada'


class SourcePackaging(unittest.TestCase):
    def test_current_and_upstream_archives_have_trainer_root_paths(self):
        source = ROOT / 'dev/test/modal_gpu_tests.py'
        tree = ast.parse(source.read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'package_sources']
        namespace = dict(ROOT=ROOT, BASE=BASE, io=io, hashlib=hashlib,
                         subprocess=subprocess, tarfile=tarfile)
        exec(compile(tree, str(source), 'exec'), namespace)
        current, baseline, manifest = namespace['package_sources']()
        for payload in [current, baseline]:
            with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                names = set(archive.getnames())
                for name in ['Makefile', 'train_gpt2.cu', 'llmc/adamw.cuh']:
                    self.assertIn(name, names)
                self.assertFalse(any(n.startswith(('HW5_Code/', '.git/'))
                                     or n.endswith(('.bin', '.toml')) for n in names))
        self.assertEqual(manifest['train_gpt2.cu'],
                         hashlib.sha256((ROOT/'train_gpt2.cu').read_bytes()).hexdigest())
        expected = subprocess.check_output(['git', 'show', f'{BASE}:train_gpt2.cu'], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(baseline)) as archive:
            self.assertEqual(archive.extractfile('train_gpt2.cu').read(), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
