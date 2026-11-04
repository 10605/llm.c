"""Exercise coordinator restart recovery without launching cloud workers."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

source = Path(__file__).with_name('modal_gpu_tests.py')
tree = ast.parse(source.read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'durable_phase_calls']


class Recovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dest = Path(self.temp.name)
        self.spawn = Mock(side_effect=[SimpleNamespace(object_id=f'fc-{i}') for i in range(4)])
        self.reconnect = Mock(side_effect=lambda call_id: SimpleNamespace(object_id=call_id))
        self.commit = Mock()
        self.namespace = dict(hashlib=hashlib, json=json, sweep=SimpleNamespace(spawn=self.spawn),
                              volume=SimpleNamespace(commit=self.commit),
                              modal=SimpleNamespace(FunctionCall=SimpleNamespace(from_id=self.reconnect)))
        exec(compile(tree, str(source), 'exec'), self.namespace)
        self.call = self.namespace['durable_phase_calls']
        self.args = [b'source', b'baseline', 'test-run', self.dest, 'narrow', -16, 0, {'scheduler': 'constant'}]

    def test_restart_reuses_original_workers(self):
        first = self.call(*self.args)
        # Reloading the function models a fresh coordinator process.
        exec(compile(tree, str(source), 'exec'), self.namespace)
        second = self.namespace['durable_phase_calls'](*self.args)
        self.assertEqual([x.object_id for x in first], [x.object_id for x in second])
        self.assertEqual(self.spawn.call_count, 4)
        self.assertEqual(self.reconnect.call_count, 4)
        self.commit.assert_called_once()

    def test_changed_source_or_recipe_is_rejected(self):
        self.call(*self.args)
        for index, replacement in [(0, b'changed'), (1, b'other-base'), (6, -1), (7, {'scheduler': 'cosine'})]:
            changed = self.args.copy()
            changed[index] = replacement
            with self.assertRaises(ValueError):
                self.call(*changed)
        self.assertEqual(self.spawn.call_count, 4)
        self.reconnect.assert_not_called()

    def test_duplicate_worker_ids_are_rejected(self):
        self.call(*self.args)
        path = self.dest / 'narrow-calls.json'
        saved = json.loads(path.read_text())
        saved['call_ids'] = ['fc-0'] * 4
        path.write_text(json.dumps(saved))
        with self.assertRaises(ValueError):
            self.call(*self.args)
        self.assertEqual(self.spawn.call_count, 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
