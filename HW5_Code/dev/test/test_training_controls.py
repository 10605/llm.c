"""Host-only tests of actual training CLI/configuration code; no GPU emulation.
Run: python3 dev/test/test_training_controls.py
The harness extracts code before CUDA startup/allocation. It cannot test kernels.
"""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest
from prepare_course_recipe import make_plan

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
    int batch_tokens = total_batch_size == -1 ? B*T : total_batch_size;
    int schedule_steps = resolve_training_steps(max_steps, token_budget, (size_t)batch_tokens*10, batch_tokens);
    int resolved_early_stop = resolve_early_stop_steps(early_stopping_steps, early_stopping_tokens,
                                                     token_budget, schedule_steps, batch_tokens);
    warmup_iterations = resolve_warmup(warmup_iterations, schedule_steps, lr_scheduler_type);
    weight_decay = resolve_weight_decay(weight_decay, weight_decay_timescale, learning_rate, schedule_steps);
    int stop = training_end_step(start_step, schedule_steps, resolved_early_stop, warmup_iterations);
    printf("layers=%d channels=%d heads=%d mup=%d mm=%g ma=%g ref=%g start=%d stop=%d val=%d\n",
           model.config.num_layers, model.config.channels, model.config.num_heads, model.use_mup,
           model.mup_width_mult, model.mup_base_attn_mult, model.mup_reference_width, start_step, stop, val_max_steps);
    printf("train=%s val=%s tokenizer=%s hellaswag=%s\n", train_data_pattern, val_data_pattern, tokenizer_path, hellaswag_path);
    printf("val2=%s\n", val2_data_pattern ? val2_data_pattern : "disabled");
    printf("preset=%d seed=%d batch_tokens=%d schedule_steps=%d schedule_tokens=%lld warmup=%d lr=%g decay=%g final=%g\n",
           course_preset, model_options.initialization_seed, batch_tokens, schedule_steps,
           (long long)schedule_steps*batch_tokens, warmup_iterations, learning_rate, weight_decay, final_learning_rate_frac);
    printf("requested_early_tokens=%lld stop_tokens=%lld\n", early_stopping_tokens, (long long)stop*batch_tokens);
    printf("context=%d capacity=%d scheduler=%s tau=%g\n", T, model.config.max_seq_len, lr_scheduler_type, weight_decay_timescale);
    printf("microbatch=%d checkpoint=%d retention=%d val_every=%d reset_loader=%d\n", B, checkpoint_every, checkpoints_keep, val_loss_every, reset_train_loader);
    return 0;
}
'''
        (cls.directory / 'controls.cpp').write_text(harness)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined', '-I', str(ROOT), str(cls.directory / 'controls.cpp'), '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cli(self, *args, ok=True, legacy=True):
        preset = ['-cp','0'] if legacy and args != ('--unit',) else []
        result = subprocess.run([str(self.exe), *preset, *map(str, args)], cwd=self.directory, capture_output=True, text=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn('Sanitizer', result.stderr)
        return result.stdout + result.stderr

    def test_help(self):
        help_text = self.run_cli('--help')
        for flag in ['-1d', '-1c', '-1h', '-ss', '-es', '-id', '-iv', '-it', '-ih', '-me', '-mm', '-ma', '-mc', '-mw', '-mb', '-cp', '-nt', '-et', '-se', '-wt']:
            self.assertIn(flag, help_text)

    def test_course_defaults_and_width_transfer(self):
        out = self.run_cli('-l', '.002', legacy=False)
        self.assertIn('layers=12 channels=768 heads=12 mup=1 mm=4 ma=4 ref=192', out)
        self.assertIn('preset=1 seed=42 batch_tokens=114688 schedule_steps=21838 schedule_tokens=2504556544 warmup=2183 lr=0.002', out)
        self.assertIn('context=2048 capacity=2048 scheduler=linear tau=0.1', out)
        self.assertIn('train=data/pretraining/climbmix_students_train_*.bin val=data/pretraining/climbmix_students_val_*.bin tokenizer=data/gpt2_tokenizer.bin', out)
        self.assertIn('val2=disabled', out)
        for width in [48,96,192,384,768]:
            out = self.run_cli('-l','.002','-1c',width,'-x',100,legacy=False)
            self.assertIn(f'channels={width} heads=12 mup=1 mm={width/192:g} ma=4 ref=192',out)
            self.assertIn('warmup=10',out)
        self.assertIn('layers=6 channels=768 heads=12',self.run_cli('-l','.002','-1d',6,legacy=False))
        self.assertIn('layers=6 channels=384 heads=12',self.run_cli('-l','.002','-e','d6',legacy=False))
        self.assertIn('mup=0 mm=1 ma=1',self.run_cli('-l','.002','-me',0,legacy=False))
        self.run_cli('-l','.002','-1c',64,legacy=False,ok=False)  # fixed 12 heads must divide width

    def test_required_lr_and_explicit_epoch(self):
        self.assertIn('requires an explicit learning rate', self.run_cli(legacy=False, ok=False))
        self.assertIn('requires an explicit learning rate',
                      self.run_cli('-y', 1, '-o', '.', legacy=False, ok=False))
        self.assertIn('REQUIRED', self.run_cli('--help', legacy=False))
        default = self.run_cli('-l', '.002', legacy=False)
        self.assertIn('microbatch=8 checkpoint=1000 retention=2 val_every=1000 reset_loader=0', default)
        self.assertIn('val=72', default)
        epoch = self.run_cli('-l', '.002', '-x', -1, legacy=False)
        self.assertIn('schedule_steps=10 schedule_tokens=1146880 warmup=1', epoch)
        for flags in [('-nt', 2561), ('-x', -1, '-nt', 2561), ('-nt', 2561, '-x', -1)]:
            self.assertIn('schedule_steps=11', self.run_cli('-l', '.002', '-d', 256, *flags, legacy=False))
        legacy = self.run_cli('-e', 'd12')
        self.assertIn('microbatch=4 checkpoint=0 retention=0 val_every=20', legacy)
        self.assertIn('schedule_steps=10', legacy)

    def test_student_stops_and_dataset_switch(self):
        fresh = self.run_cli('-l', '.002', '-es', 2000, legacy=False)
        self.assertIn('start=0 stop=2000', fresh)
        self.assertIn('schedule_steps=21838', fresh)
        self.assertIn('warmup=2183', fresh)
        resumed = self.run_cli('-l', '.002', '-e', 'saved.bin', '-y', 1, '-o', '.',
                              '-rd', 1, '-id', 'cluster/train*.bin', '-es', 3000, legacy=False)
        self.assertIn('start=4 stop=3000', resumed)
        self.assertIn('reset_loader=1', resumed)
        self.assertIn('train=cluster/train*.bin', resumed)
        self.assertIn('schedule_steps=21838', resumed)
        self.run_cli('-l', '.002', '-rd', 1, legacy=False, ok=False)
        self.assertIn('warmup=200', self.run_cli('-l', '.002', '-x', 2000, legacy=False))

    def test_optional_second_validation(self):
        out = self.run_cli('-l', '.002', '-iv', 'data/expert/*val*.bin',
                           '-iv2', 'data/pretraining/*val*.bin', legacy=False)
        self.assertIn('val=data/expert/*val*.bin', out)
        self.assertIn('val2=data/pretraining/*val*.bin', out)
        self.assertIn('-iv2', self.run_cli('--help'))
        self.run_cli('-iv2', ok=False)

    def test_course_overrides_and_legacy_order(self):
        for flags in [('-cp',0,'-l','.002'),('-l','.002','-cp',0)]:
            out=self.run_cli('-e','d6',*flags,legacy=False)
            self.assertIn('heads=6 mup=0 mm=1',out)
            self.assertIn('warmup=0 lr=0.002 decay=0 final=1',out)
        out=self.run_cli('-l','.002','-1c',64,'-1h',2,'-mb',64,'-ma',8,'-se',43,legacy=False)
        self.assertIn('heads=2 mup=1 mm=1 ma=8 ref=64',out)
        self.assertIn('seed=43',out)
        self.assertIn('warmup=0',self.run_cli('-l','.002','-x',500,'-k','constant',legacy=False))
        self.assertIn('warmup=0',self.run_cli('-l','.002','-x',500,'-u',0,legacy=False))
        self.assertIn('warmup=50',self.run_cli('-l','.002','-x',500,'-es',10,legacy=False))

    def test_timescale_decay_transfer_and_overrides(self):
        def values(*flags):
            out = self.run_cli('-nt',256000,'-d',256,'-l','.002',*flags,legacy=False)
            return dict(re.findall(r'(\w+)=([^\s]+)',out))
        base = values()
        self.assertAlmostEqual(float(base['decay']), 5, places=5)
        self.assertEqual(base['schedule_steps'], '1000')
        # Changing LR or batch automatically compensates decay to preserve tau.
        self.assertAlmostEqual(float(values('-l','.004')['decay']), 2.5, places=5)
        self.assertAlmostEqual(float(values('-d',512)['decay']), 10, places=5)
        self.assertAlmostEqual(float(values('-nt',512000)['decay']), 2.5, places=5)
        self.assertAlmostEqual(float(values('-wt','.2')['decay']), 2.5, places=5)
        for width in [48,96,192,768]:
            self.assertEqual(values('-1c',width)['decay'],base['decay'])
        # Early stopping, including resume, cannot shorten the decay horizon.
        self.assertEqual(values('-et',1280)['decay'],base['decay'])
        self.assertEqual(values('-e','saved.bin','-y',1,'-o','.','-et',1280)['decay'],base['decay'])
        for flags in [('-c','.3','-l','.004'),('-l','.004','-c','.3')]:
            self.assertEqual(values(*flags)['tau'],'-1')
            self.assertEqual(values(*flags)['decay'],'0.3')
        self.assertEqual(values('-c',0)['decay'],'0')
        self.assertEqual(values('-cp',0,'-e','d12','-wt','.1')['decay'],base['decay'])

    def test_timescale_validation_and_no_updates(self):
        for flags in [('-wt',0),('-wt',-1),('-wt','nan'),('-wt','inf'),
                      ('-wt','.1','-c','.1'),('-c','.1','-wt','.1'),
                      ('-l',0),('-x',1),('-wt','1e-30'),('-l','1e-37','-wt','1e-2')]:
            self.run_cli('-l','.002','-x',100,*flags,legacy=False,ok=False)
        self.assertIn('decay=0',self.run_cli('-l','.002','-x',0,legacy=False))
        self.assertIn('decay=0',self.run_cli('-l','.002','-x',1,'-l',0,'-c',0,legacy=False))

    def test_prepared_course_recipe_matches_real_cli(self):
        plan=make_plan('train_*.bin','val_*.bin','tokenizer.bin')
        self.assertEqual(plan['batch_tokens'],114688)
        self.assertEqual(plan['updates'],21838)
        self.assertEqual(plan['actual_tokens'],2504556544)
        self.assertEqual(len(plan['trials']),5)
        self.assertIsNone(plan['selected_lr'])
        for trial in plan['trials']:
            out=self.run_cli('-l','.002',*trial['command'][1:],legacy=False)
            values=dict(re.findall(r'(\w+)=([^\s]+)',out))
            self.assertEqual(int(values['schedule_steps']),plan['updates'])
            self.assertEqual(values['context'],'2048')
            self.assertEqual(values['capacity'],'2048')
            self.assertEqual(values['heads'],'12')
            self.assertAlmostEqual(float(values['decay']),trial['base_weight_decay'],places=5)
        # Proxy width does not reduce the target token budget or batch.
        smaller=make_plan('train','val','tok',width=48)
        self.assertEqual(smaller['actual_tokens'],plan['actual_tokens'])
        self.assertEqual(smaller['batch_tokens'],plan['batch_tokens'])
        for kwargs in [dict(width=64),dict(tokens=0),dict(microbatch=16),dict(rates=[]),
                       dict(rates=[0]),dict(rates=[1e100]),dict(rates=[1e-50]),
                       dict(timescale=0),dict(timescale=float('nan'))]:
            with self.assertRaises(ValueError): make_plan('train','val','tok',**kwargs)

    def test_token_budgets_and_seed_validation(self):
        out=self.run_cli('-l','.002','-nt',1001,'-d',256,'-se',0,'-c',0,legacy=False)
        self.assertIn('seed=0 batch_tokens=256 schedule_steps=4 schedule_tokens=1024 warmup=0',out)
        out=self.run_cli('-l','.002','-nt',2490000000,legacy=False)
        self.assertIn(f'schedule_steps={(2490000000+114687)//114688}',out)
        for flags in [('-nt',0),('-nt',-1),('-nt','2.49e9'),('-nt','9223372036854775808'),
                      ('-nt','9223372036854775807','-d',1),('-nt',1024,'-x',4),
                      ('-se',-1),('-se','42.5'),('-se','2147483648'),('-cp',2),('-cp',),('-u',-2)]:
            with self.subTest(flags=flags): self.run_cli('-l','.002',*flags,legacy=False,ok=False)

    def test_course_defaults_yield_to_checkpoint_metadata(self):
        out=self.run_cli('-l','.002','-e','saved.bin','-y',1,'-o','.',legacy=False)
        self.assertIn('layers=2 channels=256 heads=4 mup=1 mm=2 ma=8 ref=128 start=4',out)
        self.run_cli('-l','.002','-e','saved.bin','-1h',12,legacy=False,ok=False)
        self.run_cli('-l','.002','-e','saved.bin','-mb',192,legacy=False,ok=False)
        self.run_cli('-l','.002','-e','legacy.bin',legacy=False,ok=False)
        self.assertIn('mup=0',self.run_cli('-l','.002','-e','legacy.bin','-me',0,legacy=False))

    def test_token_early_stop_and_schedule(self):
        out=self.run_cli('-l','.002','-nt',25600,'-d',256,'-et',1281,legacy=False)
        self.assertIn('start=0 stop=6',out)
        self.assertIn('schedule_steps=100 schedule_tokens=25600 warmup=10',out)
        self.assertIn('requested_early_tokens=1281 stop_tokens=1536',out)
        # Equal token limits round to the same complete update.
        self.assertIn('stop=4',self.run_cli('-l','.002','-nt',1001,'-et',1001,'-d',256,'-c',0,legacy=False))
        # Token stopping also works with the default or an explicit step horizon.
        self.assertIn('stop=4',self.run_cli('-l','.002','-et',1001,'-d',256,legacy=False))
        self.assertIn('stop=4',self.run_cli('-l','.002','-et',1001,'-d',256,'-x',20,legacy=False))
        for batch in [262144,524288,1048576]:
            out=self.run_cli('-l','.002','-nt',2621440000,'-et',524288000,'-d',batch,legacy=False)
            self.assertIn(f'stop={524288000//batch}',out)
            self.assertIn('stop_tokens=524288000',out)
            self.assertIn(f'warmup={2621440000//batch//10}',out)
        # Checkpoint fixture resumes at step 4: -et is absolute, not additive.
        resume=('-e','saved.bin','-y',1,'-o','.','-nt',25600,'-d',256)
        self.assertIn('start=4 stop=5',self.run_cli('-l','.002',*resume,'-et',1025,legacy=False))
        self.assertIn('start=4 stop=4',self.run_cli('-l','.002',*resume,'-et',769,legacy=False))
        self.run_cli('-l','.002',*resume,'-et',768,legacy=False,ok=False)

    def test_token_early_stop_invalid_limits(self):
        for flags in [('-et',0),('-et',-1),('-et','1e6'),('-et','9223372036854775808'),
                      ('-et','9223372036854775807'),('-et',1001,'-es',4),
                      ('-et',1001,'-es',0),('-et',2561),('-et',1002,'-nt',1001)]:
            with self.subTest(flags=flags):
                self.run_cli('-l','.002','-x',-1,'-d',256,*flags,legacy=False,ok=False)

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
