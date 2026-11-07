"""Host checks for student log compatibility, S3 layout, and seed-copy safeguards."""
from pathlib import Path
import json
import math
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment_logs import parse_parameters, read_log, plot_runs, upload_plan, upload_run, download_logs
from prepare_expert import prepare


class FakeS3:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key):
        self.objects[key] = Path(path).read_bytes()

    def get_paginator(self, name):
        return self

    def paginate(self, **kwargs):
        # Include an unsafe key to verify that downloaded names cannot escape dest.
        yield {"Contents": [{"Key": k} for k in self.objects] + [
            {"Key": "experiments/group/../../escape.txt"}]}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])


class Helpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / "trial"
        self.run.mkdir()
        (self.run / "main.log").write_text(
            "s:10919 tel:3.5273\n"
            "s:10919 trl:3.6 lr:0.0007 norm:2 time (ms):2500\n"
            "s:10919 trl:3.5 lr:0.0006 norm:1 time (ms):2000\n"
            "s:10920 trl:nan lr:0.0006 norm:inf time (ms):2100\n"
            "s:10921 tel:3.4\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_metadata_formats_and_last_resume_block(self):
        p = self.run / "run_parameters.txt"
        p.write_text("layers=12 parameters=125262336\nscheduler=linear lr=0.001\n"
                     "scheduler=linear lr=0.002\n")
        self.assertEqual(parse_parameters(p)["learning_rate"], .002)
        self.assertEqual(parse_parameters(p)["num_parameters"], 125262336)
        p.write_text("| learning rate (LR) | 1e-4 |\n| num_parameters | 125262336 |\n")
        self.assertEqual(parse_parameters(p)["learning_rate"], .0001)
        self.assertEqual(parse_parameters(p)["num_parameters"], 125262336)

    def test_missing_metadata_is_allowed(self):
        self.assertEqual(parse_parameters(self.root / "missing"), {})

    def test_resumed_steps_and_nonfinite_gap(self):
        data = read_log(self.run / "main.log")
        self.assertEqual(len(data["streams"]["trl"]), 2)
        self.assertEqual(data["streams"]["trl"][0], (10919, 3.5))
        self.assertTrue(math.isnan(data["streams"]["trl"][1][1]))
        self.assertAlmostEqual(data["training_hours"], 4100 / 3_600_000)
        self.assertEqual(data["learning_rates"][0], (10919, .0006))

    def test_plot_without_hellaswag_or_metadata(self):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_runs([self.run], self.root / "plot.png")
        self.assertEqual(len(fig.axes), 2)
        self.assertGreater((self.root / "plot.png").stat().st_size, 1000)

    def test_upload_layout_and_optional_checkpoint_backup(self):
        (self.run / "model_00010919.bin").write_bytes(b"model")
        (self.run / "state_00010919_00000.bin").write_bytes(b"state")
        (self.run / "DONE_00010919").touch()
        self.assertEqual([key for _, key in upload_plan(self.run, "group")],
                         ["experiments/group/trial/main.log"])
        self.assertEqual(len(upload_plan(self.run, "group", True)), 4)
        with self.assertRaises(ValueError):
            upload_plan(self.run, "../other")
        client = FakeS3()
        upload_run(client, "bucket", self.run, "group", True)
        paths = download_logs(client, "bucket", "group", self.root / "downloaded")
        self.assertEqual([p.name for p in paths], ["main.log"])
        self.assertEqual(paths[0].read_bytes(), (self.run / "main.log").read_bytes())

    def test_two_validation_streams_resume_labels_and_upload(self):
        with (self.run / "main.log").open("a") as f:
            f.write("s:10919 tel2:4.1\ns:10921 tel2:nan\ns:10919 tel2:4.0\n")
        (self.run / "run_parameters.txt").write_text(
            "validation=data/expert/climbmix_students_val_*.bin\n"
            "validation_secondary=data/pretraining/climbmix_students_val_*.bin\n")
        data = read_log(self.run / "main.log")
        self.assertEqual(data["streams"]["tel2"][0], (10919, 4.0))
        self.assertTrue(math.isnan(data["streams"]["tel2"][1][1]))
        fig = plot_runs([self.run])
        labels = [line.get_label() for line in fig.axes[0].lines]
        self.assertTrue(any("selected-cluster validation" in x for x in labels))
        self.assertTrue(any("pretraining-mix validation" in x for x in labels))
        self.assertEqual(len(fig.axes[0].lines), 3)
        client = FakeS3()
        upload_run(client, "bucket", self.run, "group")
        download_logs(client, "bucket", "group", self.root / "roundtrip")
        restored = read_log(self.root / "roundtrip/group/trial/main.log")
        self.assertEqual(restored["streams"]["tel2"][0], (10919, 4.0))

    def test_seed_copy_and_existing_trial_guard(self):
        seed = self.root / "seed"
        seed.mkdir()
        model = [20240326, 5, 2048, 50257, 12, 12, 768, 50304, 0x4D555031] + [0] * 247
        state = [0] * 256
        state[0], state[10] = 20240527, 10919
        (seed / "model_00010919.bin").write_bytes(struct.pack("<256i", *model))
        (seed / "state_00010919_00000.bin").write_bytes(struct.pack("<256i", *state))
        (seed / "DONE_00010919").touch()
        target = self.root / "expert"
        prepare(target, seed)
        for f in seed.iterdir():
            self.assertEqual(f.read_bytes(), (target / f.name).read_bytes())
        (target / "main.log").write_text("existing training\n")
        with self.assertRaises(ValueError):
            prepare(target, seed)
        self.assertEqual((target / "main.log").read_text(), "existing training\n")
        (seed / "DONE_00010919").unlink()
        with self.assertRaises(ValueError):
            prepare(self.root / "incomplete", seed)


if __name__ == "__main__":
    unittest.main()
