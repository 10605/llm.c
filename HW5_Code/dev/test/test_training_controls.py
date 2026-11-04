"""Host-only tests of actual training CLI/configuration code; no GPU emulation.
Run: python3 dev/test/test_training_controls.py
The harness extracts code before CUDA startup/allocation. It cannot test kernels.
"""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / 'train_gpt2.cu').read_text()


def function(text, signature):
    start = text.index(signature)
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


class TrainingControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='llmc-host-')
        cls.directory = Path(cls.tmp.name)
        cls.exe = cls.directory / 'controls'
        config = SOURCE[SOURCE.index('typedef struct {'):SOURCE.index('} GPT2Config;') + len('} GPT2Config;')]
        builders = SOURCE[SOURCE.index('void gpt2_set_hyperparameters'):SOURCE.index('    gpt2_allocate_weights(model);', SOURCE.index('void gpt_build_from_descriptor'))] + '}\n'
        parser = SOURCE[SOURCE.index('int main(int argc'):SOURCE.index('    multi_gpu_config = multi_gpu_config_init', SOURCE.index('int main(int argc'))]
        unit = r'''
void unit_tests() {
    LearningRateScheduler lr;
    for (const char* type : {"cosine", "linear", "constant", "wsd"}) {
        lr_scheduler_init(&lr, type, 1.0f, 2, 10, 0.1f);
        assert(get_learning_rate(&lr, 0) == (!strcmp(type, "constant") ? 1.0f : 0.5f));
        assert(fabsf(get_learning_rate(&lr, 10) - (!strcmp(type, "constant") ? 1.0f : 0.1f)) < 1e-6f);
        for (int step = 0; step <= 10; step++) assert(isfinite(get_learning_rate(&lr, step)));
    }
    lr_scheduler_init(&lr, "linear", 1.0f, 2, 10, 0.1f);
    assert(fabsf(get_learning_rate(&lr, 6) - 0.55f) < 1e-6f);
    int stop = training_end_step(3, 10, 6, 2);
    assert(stop == 6);
    int updates = 0;
    for (int step = 3; step < stop; step++) ++updates;
    assert(updates == 3 && lr.train_num_batches == 10);
    assert(training_end_step(10, 10, -1, 0) == 10);
    assert(training_end_step(0, 0, 0, 0) == 0);
    // Binary header extension survives disk I/O and restores SP and muP.
    for (int enabled : {0, 1}) {
        int header[256] = {};
        write_parametrization(header + 8, enabled, 2.0f, 8.0f, 128.0f);
        FILE* f = tmpfile(); assert(f);
        assert(fwrite(header, sizeof(header), 1, f) == 1);
        rewind(f); memset(header, 0, sizeof(header));
        assert(fread(header, sizeof(header), 1, f) == 1); fclose(f);
        int me = 0; float mm = 1, ma = 1, ref = 0;
        ModelOptions opts = {};
        resolve_parametrization(256, opts, header + 8, true, &me, &mm, &ma, &ref);
        assert(me == enabled && mm == 2 && ma == 8 && ref == 128);
        assert(header[7] == 0 && header[14] == 0);
    }
    ModelOptions opts = {}; opts.enabled_set = true;
    int legacy[6] = {}, me = 0; float mm = 1, ma = 1, ref = 0;
    resolve_parametrization(256, opts, legacy, true, &me, &mm, &ma, &ref);
    assert(ref == 256);
    opts.mult_set = opts.attn_set = true; me = 1; mm = 2; ma = 8;
    resolve_parametrization(256, opts, legacy, true, &me, &mm, &ma, &ref);
    assert(ref == 128 && mm == 2 && ma == 8);
    Logger logger;
    logger_init(&logger, NULL, 0, 0);
    logger_log_train_timed(&logger, 0, 1, 2, 3, 4);
    logger_init(&logger, ".", 1, 0);
    logger_log_train_timed(&logger, 0, 1, 2, 3, 4);
    assert(access("main.log", F_OK) != 0);
    logger_init(&logger, ".", 0, 0);
    logger_log_train_timed(&logger, 0, 1, 2, 3, 4);
    logger_init(&logger, ".", 0, 1);
    logger_log_train_timed(&logger, 1, 2, 3, 4, 5);
    puts("host unit checks passed");
}
'''
        harness = r'''
#include <assert.h>
#include <unistd.h>
#include <initializer_list>
#include "llmc/train_config.h"
#include "llmc/schedulers.h"
#include "llmc/logger.h"
''' + config + r'''
struct GPT2 {
    GPT2Config config;
    int use_mup = 0;
    float mup_width_mult = 1, mup_base_attn_mult = 1, mup_reference_width = 0;
};
''' + builders + function(SOURCE, 'void error_usage(') + unit
        parser = parser.replace('    // read in the (optional)', '    if (argc == 2 && !strcmp(argv[1], "--unit")) { unit_tests(); return 0; }\n    // read in the (optional)', 1)
        harness += parser + r'''
    GPT2 model = {};
    model.use_mup = use_mup; model.mup_width_mult = mup_width_mult; model.mup_base_attn_mult = mup_base_attn_mult;
    if (ends_with_bin(load_filename)) {
        model.config = {1024, 50257, 50304, 2, 4, 256};
        int header[6] = {};
        if (strcmp(load_filename, "legacy.bin")) write_parametrization(header, 1, 2, 8, 128);
        apply_architecture(&model.config, model_options, true);
        resolve_parametrization(256, model_options, header, resume, &model.use_mup,
                               &model.mup_width_mult, &model.mup_base_attn_mult, &model.mup_reference_width);
        if (resume) start_step = 4; // the real load_state overwrites -ss with saved step
    } else {
        gpt_build_from_descriptor(&model, load_filename, mup_coord_check_width, mup_coord_check, &model_options);
    }
    int stop = training_end_step(start_step, max_steps == -1 ? 10 : max_steps, early_stopping_steps, warmup_iterations);
    printf("layers=%d channels=%d heads=%d mup=%d mm=%g ma=%g ref=%g start=%d stop=%d val=%d\n",
           model.config.num_layers, model.config.channels, model.config.num_heads, model.use_mup,
           model.mup_width_mult, model.mup_base_attn_mult, model.mup_reference_width, start_step, stop, val_max_steps);
    printf("train=%s val=%s tokenizer=%s hellaswag=%s\n", train_data_pattern, val_data_pattern, tokenizer_path, hellaswag_path);
    return 0;
}
'''
        (cls.directory / 'controls.cpp').write_text(harness)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined', '-I', str(ROOT), str(cls.directory / 'controls.cpp'), '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cli(self, *args, ok=True):
        result = subprocess.run([str(self.exe), *map(str, args)], cwd=self.directory, capture_output=True, text=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn('Sanitizer', result.stderr)
        return result.stdout + result.stderr

    def test_help(self):
        help_text = self.run_cli('--help')
        for flag in ['-1d', '-1c', '-1h', '-ss', '-es', '-id', '-iv', '-it', '-ih', '-me', '-mm', '-ma', '-mc', '-mw', '-mb']:
            self.assertIn(flag, help_text)

    def test_sp_descriptor_defaults(self):
        out = self.run_cli('-e', 'd12')
        self.assertIn('layers=12 channels=768 heads=12 mup=0 mm=1', out)

    def test_mup_descriptor_defaults(self):
        self.assertIn('mup=1 mm=1 ma=1 ref=768', self.run_cli('-e', 'd12', '-me', 1))

    def test_custom_architecture(self):
        for me in [0, 1]:
            out = self.run_cli('-1d', 3, '-1c', 256, '-1h', 4, '-me', me, '-mb', 128)
            self.assertIn(f'layers=3 channels=256 heads=4 mup={me} mm=2', out)

    def test_explicit_multiplier(self):
        self.assertIn('mm=2 ma=3 ref=128', self.run_cli('-1c', 256, '-1h', 4, '-me', 1, '-mm', 2, '-ma', 3))
        self.run_cli('-1c', 256, '-1h', 4, '-me', 1, '-mm', 3, ok=False)
        self.run_cli('-1c', 256, '-1h', 4, '-me', 1, '-mb', 128, '-mm', 1, ok=False)
        self.run_cli('-1c', 256, '-1h', 4, '-me', 1, ok=False)

    def test_coordinate_widths(self):
        for width in [64, 128, 256, 512, 1024, 2048]:
            for me in [0, 1]:
                out = self.run_cli('-mc', 1, '-mw', width, '-me', me, '-mm', width / 256, '-ma', 11.313)
                self.assertIn(f'layers=2 channels={width} heads=2', out)
                self.assertIn('ref=256', out)
        self.assertIn('layers=3 channels=128 heads=4', self.run_cli('-mc', 1, '-1d', 3, '-1c', 128, '-1h', 4))
        self.run_cli('-mc', 1, '-mw', 256, '-1c', 128, ok=False)
        self.run_cli('-mc', 1, '-e', 'model.bin', ok=False)

    def test_paths_and_flag_collisions(self):
        out = self.run_cli('-e', 'd6', '-id', 'train*', '-iv', 'val*', '-it', 'tokens.bin', '-ih', 'hella.bin', '-m', 7, '-me', 1, '-mm', 2, '-ma', 3)
        self.assertIn('val=7', out)
        self.assertIn('mup=1 mm=2 ma=3', out)
        self.assertIn('train=train* val=val* tokenizer=tokens.bin hellaswag=hella.bin', out)
        self.assertIn('train=old val=older', self.run_cli('-e', 'd6', '-i', 'old', '-j', 'older'))

    def test_schedule_and_early_stop(self):
        self.assertIn('start=3 stop=6', self.run_cli('-e', 'd6', '-ss', 3, '-es', 6, '-x', 10, '-u', 2))
        for args in [('-ss', 11), ('-es', 11), ('-ss', 6, '-es', 5), ('-u', 11), ('-k', 'bogus')]:
            self.run_cli('-e', 'd6', '-x', 10, *args, ok=False)

    def test_checkpoint_recovery_and_conflicts(self):
        out = self.run_cli('-e', 'saved.bin', '-y', 1, '-o', '.', '-ss', 8)
        self.assertIn('mup=1 mm=2 ma=8 ref=128 start=4', out)
        for args in [('-me', 0), ('-mm', 1), ('-ma', 2), ('-mb', 256), ('-1d', 3), ('-1h', 8), ('-1c', 512)]:
            self.run_cli('-e', 'saved.bin', *args, ok=False)
        self.run_cli('-e', 'legacy.bin', '-y', 1, '-o', '.', ok=False)
        self.run_cli('-e', 'legacy.bin', '-y', 1, '-o', '.', '-me', 0)
        self.run_cli('-e', 'legacy.bin', '-me', 1, ok=False)
        self.run_cli('-e', 'legacy.bin', '-me', 1, '-mm', 2, '-ma', 8)

    def test_malformed_arguments(self):
        for args in [('-me',), ('-meX', 1), ('-mx', 1), ('-ix', 'x'), ('-1c', '256x'), ('-1c', 0), ('-1c', -1), ('-mm', 'nan'), ('-ma', 'inf'), ('-mm', 0), ('-me', 2), ('-b', '99999999999999999999'), ('-mw', 1.5), ('-1c', 257, '-1h', 4), ('-pi', 'a'*256), ('-e', 'd12oops')]:
            with self.subTest(args=args):
                self.run_cli('-e', 'd6', *args, ok=False)

    def test_host_units(self):
        self.run_cli('--unit')
        log = (self.directory / 'main.log').read_text()
        self.assertEqual(len(log.splitlines()), 2)
        self.assertIn('time (ms):4.00', log)
        self.assertIn('s:1', log)


if __name__ == '__main__':
    unittest.main(verbosity=2)
