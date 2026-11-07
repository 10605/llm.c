// Exercise the actual full-model forward/backward on an independent fixture.
#define TESTING
#include "../../llmc/cuda_common.h"
#ifdef MUP_GRADIENT_TRACE
#undef NVTX_RANGE_FN
#define NVTX_RANGE_FN() do { fprintf(stderr, "ENTER %s\n", __FUNCTION__); cudaCheck(cudaDeviceSynchronize()); fprintf(stderr, "SYNC %s\n", __FUNCTION__); } while (0)
#endif
#include "../../train_gpt2.cu"
#include <algorithm>
#include <vector>

void dump_float_tensor(const std::string& path, const floatX* device, size_t count) {
    std::vector<floatX> raw(count);
    std::vector<float> values(count);
    cudaCheck(cudaMemcpy(raw.data(), device, count * sizeof(floatX), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) values[i] = (float)raw[i];
    FILE* file = fopenCheck(path.c_str(), "wb");
    fwriteCheck(values.data(), sizeof(float), count, file);
    fcloseCheck(file);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 3) return 2;
    char empty[] = "", method[] = "mpi";
    multi_gpu_config = multi_gpu_config_init(-1, -1, -1, empty, empty, method);
    common_start(false, true); // Disable TF32 for FP32 gradient verification.
    FILE* file = fopenCheck(argv[1], "rb");
    int header[10];
    freadCheck(header, sizeof(int), 10, file);
    if (header[0] != 20261103) return 3;
    const int C = header[1], L = header[2], H = header[3], V = header[4];
    const int B = header[5], T = header[6];
    GPT2 model;
    gpt2_init_common(&model);
    model.config = {T, V, V, L, H, C};
    model.use_mup = header[7];
    memcpy(&model.mup_width_mult, &header[8], sizeof(float));
    memcpy(&model.mup_base_attn_mult, &header[9], sizeof(float));
    model.recompute = 0;
    model.gelu_fusion = 0;
    model.use_master_weights = 0;
    gpt2_allocate_weights(&model);
    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);
    gpt2_allocate_state(&model, B, T);

    // Small-vocabulary fixtures need extra space for the existing layernorm
    // scratch reduction, which normally fits in the full-vocabulary output.
    size_t scratch_bytes = (32 + 4 * C * deviceProp.multiProcessorCount) * sizeof(float);
    size_t output_bytes = model.acts_specs[18].size * sizeof(floatX);
    floatX* diagnostic_output;
    cudaCheck(cudaMalloc((void**)&diagnostic_output, std::max(scratch_bytes, output_bytes)));
    model.acts.output = diagnostic_output;

    std::vector<int> inputs(B*T), targets(B*T);
    std::vector<float> source(model.num_parameters);
    std::vector<floatX> params(model.num_parameters);
    freadCheck(inputs.data(), sizeof(int), B*T, file);
    freadCheck(targets.data(), sizeof(int), B*T, file);
    freadCheck(source.data(), sizeof(float), model.num_parameters, file);
    fcloseCheck(file);
    for (size_t i = 0; i < model.num_parameters; ++i) params[i] = (floatX)source[i];
    cudaCheck(cudaMemcpy(model.params_memory, params.data(), model.num_parameters_bytes, cudaMemcpyHostToDevice));
    const std::string prefix = argv[2];
    dump_float_tensor(prefix + ".params.bin", (floatX*)model.params_memory, model.num_parameters);
    gpt2_forward(&model, inputs.data(), B, T, 0, NULL);
    dump_float_tensor(prefix + ".logits.bin", model.acts.output, B*T*V);
    gpt2_backward_and_reduce(&model, inputs.data(), targets.data(), 1, 0);
    dump_float_tensor(prefix + ".grads.bin", (floatX*)model.grads_memory, model.num_parameters);
    printf("GRADIENT_FIXTURE loss=%.9g parameters=%zu width=%d multiplier=%.9g mup=%d\n",
           model.mean_loss, model.num_parameters, C, model.mup_width_mult, model.use_mup);
    cudaCheck(cudaFree(diagnostic_output));
    gpt2_free(&model);
    common_free(model);
    multi_gpu_config_free(&multi_gpu_config);
    return 0;
}
