"""Run the real host initialization and model-header I/O with host memory transfers.
No kernels run. SP remains identical to the pinned base; muP differs only in WTE.
"""
from pathlib import Path
import os
import subprocess
import tempfile
from test_training_controls import ROOT, SOURCE, function

BASE = '706b1a5fc94de4bc16a4a3684eafaad6b9e0dada'
baseline = subprocess.run(['git', 'show', f'{BASE}:train_gpt2.cu'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
config = SOURCE[SOURCE.index('typedef struct {'):SOURCE.index('} GPT2Config;') + len('} GPT2Config;')]
code = r'''
#include <assert.h>
#include <initializer_list>
#include <unistd.h>
#include "llmc/train_config.h"
#include "llmc/utils.h"
#include "llmc/rand.h"
#define printf0 printf
#define cudaCheck(x) ((void)(x))
using floatX = float;
const int NUM_PARAMETER_TENSORS = 16;
const int PRECISION_FP32 = 0, PRECISION_BF16 = 1, PRECISION_FP16 = 2, PRECISION_MODE = PRECISION_FP32;
const int main_stream = 0, IO_BUF_SIZE = 4096, cudaMemcpyHostToDevice = 0;
int cudaDeviceSynchronize() { return 0; }
int cudaMemcpy(void* dst, const void* src, size_t size, int) { memcpy(dst, src, size); return 0; }
void device_to_file(FILE* file, void* data, size_t n, int, int) { fwriteCheck(data, 1, n, file); }
void file_to_device(void* data, FILE* file, size_t n, int, int) { freadCheck(data, 1, n, file); }
''' + config + r'''
struct GPT2 {
    GPT2Config config;
    int use_mup = 0;
    float mup_width_mult = 1, mup_base_attn_mult = 1, mup_reference_width = 0;
    size_t param_elements[16], param_sizeof[16], num_parameters_bytes = 0;
    void* params_memory = nullptr;
};
''' + function(SOURCE, 'void fill_in_parameter_sizes(') + r'''
void gpt2_allocate_weights(GPT2* model) {
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters_bytes = 0;
    for (int i = 0; i < 16; i++) model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];
    model->params_memory = mallocCheck(model->num_parameters_bytes);
}
'''
for signature in ['void gpt2_set_hyperparameters(', 'void gpt3_set_hyperparameters(', 'void gpt_build_from_descriptor(', 'void gpt2_write_to_checkpoint(', 'void gpt2_build_from_checkpoint(']:
    code += function(SOURCE, signature) + '\n'
code += function(baseline, 'void gpt_build_from_descriptor(').replace('gpt_build_from_descriptor(', 'gpt_build_original(', 1)
code += r'''
int main() {
    for (int width : {48, 96, 192, 384, 768}) for (int enabled : {0, 1}) {
        GPT2 actual = {}, original = {}, standard = {};
        actual.use_mup = original.use_mup = enabled;
        actual.mup_width_mult = original.mup_width_mult = (float)width / 192;
        actual.mup_base_attn_mult = original.mup_base_attn_mult = 11.313f;
        ModelOptions opts = {}; opts.mult_set = true;
        gpt_build_from_descriptor(&actual, "d2", width, 1, &opts);
        gpt_build_original(&original, "d2", width, 1);
        assert(actual.num_parameters_bytes == original.num_parameters_bytes);
        if (enabled) {
            // The new WTE must equal SP's original random embedding, while all
            // other tensors must retain the original muP initialization exactly.
            standard.use_mup = 0;
            gpt_build_original(&standard, "d2", width, 1);
            memcpy(original.params_memory, standard.params_memory,
                   actual.param_elements[0] * sizeof(float));
            free(standard.params_memory);
        }
        assert(!memcmp(actual.params_memory, original.params_memory, actual.num_parameters_bytes));
        if (enabled) {
            float* weights = (float*)actual.params_memory;
            bool nonzero_embedding = false;
            for (size_t i = 0; i < (size_t)actual.config.vocab_size * width; i++) nonzero_embedding |= weights[i] != 0;
            assert(nonzero_embedding);
            size_t qkv_offset = 0;
            for (int i = 0; i < 4; i++) qkv_offset += actual.param_elements[i];
            for (int layer = 0; layer < 2; layer++) for (int i = 0; i < width * width; i++)
                assert(weights[qkv_offset + layer * 3 * width * width + i] == 0);
        }
        gpt2_write_to_checkpoint(&actual, "roundtrip.bin");
        GPT2 restored = {};
        gpt2_build_from_checkpoint(&restored, "roundtrip.bin");
        assert(restored.use_mup == enabled);
        assert(restored.mup_width_mult == actual.mup_width_mult);
        assert(restored.mup_reference_width == 192);
        assert(restored.mup_base_attn_mult == 11.313f);
        assert(!memcmp(actual.params_memory, restored.params_memory, actual.num_parameters_bytes));
        free(actual.params_memory); free(original.params_memory); free(restored.params_memory);
    }
    // Explicit architecture flags equivalent to the original coordinate model.
    GPT2 custom = {}, original = {};
    ModelOptions opts = {}; opts.layers = 2; opts.channels = 64; opts.heads = 2;
    gpt_build_from_descriptor(&custom, "d12", 256, 0, &opts);
    gpt_build_original(&original, "d2", 64, 1);
    assert(custom.num_parameters_bytes == original.num_parameters_bytes);
    assert(!memcmp(custom.params_memory, original.params_memory, custom.num_parameters_bytes));
    free(custom.params_memory); free(original.params_memory);
    puts("PASS: five-width SP parity; muP changes only WTE to SP's random embedding; zero queries and checkpoint round trips");
}
'''
with tempfile.TemporaryDirectory(prefix='llmc-init-') as temp:
    path = Path(temp)
    (path / 'init.cpp').write_text(code)
    subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined', '-I', str(ROOT), str(path / 'init.cpp'), '-o', str(path / 'init')], check=True)
    subprocess.run([str(path / 'init')], cwd=path, check=True)
