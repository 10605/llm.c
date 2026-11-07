// Diagnostic only: dump actual trainer states for an independent PyTorch audit.
#define TESTING
#include "../../train_gpt2.cu"
#include <algorithm>
#include <vector>

void dump_tensor(const std::string& path, const float* device, size_t count) {
    std::vector<float> values(count);
    cudaCheck(cudaMemcpy(values.data(), device, count*sizeof(float), cudaMemcpyDeviceToHost));
    FILE* f = fopenCheck(path.c_str(), "wb");
    fwriteCheck(values.data(), sizeof(float), count, f);
    fcloseCheck(f);
}

int main(int argc, char** argv) {
    static_assert(sizeof(floatX) == sizeof(float), "This parity fixture requires FP32");
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 6) return 2;
    char empty[]="", method[]="mpi";
    multi_gpu_config = multi_gpu_config_init(-1,-1,-1,empty,empty,method);
    common_start(false, true); // FP32, TF32 disabled.
    FILE* f = fopenCheck(argv[1], "rb");
    int h[10]; freadCheck(h,sizeof(int),10,f);
    if(h[0]!=20261103) return 3;
    const int C=h[1], L=h[2], H=h[3], V=h[4], B=h[5], T=h[6];
    GPT2 model; gpt2_init_common(&model);
    model.config={T,V,V,L,H,C}; model.use_mup=h[7];
    memcpy(&model.mup_width_mult,&h[8],sizeof(float));
    memcpy(&model.mup_base_attn_mult,&h[9],sizeof(float));
    model.recompute=0; model.gelu_fusion=0; model.use_master_weights=1;
    gpt2_allocate_weights(&model);
    set_zero_configs(&multi_gpu_config,0,model.num_parameters);
    gpt2_allocate_state(&model,B,T);
    size_t scratch_bytes=(32+4*C*deviceProp.multiProcessorCount)*sizeof(float);
    float* output;
    cudaCheck(cudaMalloc((void**)&output,std::max(scratch_bytes,model.acts_specs[18].size*sizeof(float))));
    model.acts.output=output;
    std::vector<int> inputs(B*T),targets(B*T);
    std::vector<float> params(model.num_parameters);
    freadCheck(inputs.data(),sizeof(int),B*T,f); freadCheck(targets.data(),sizeof(int),B*T,f);
    freadCheck(params.data(),sizeof(float),params.size(),f); fcloseCheck(f);
    cudaCheck(cudaMemcpy(model.params_memory,params.data(),model.num_parameters_bytes,cudaMemcpyHostToDevice));
    const int steps=atoi(argv[3]); const float lr=atof(argv[4]),wd=atof(argv[5]);
    for(int step=0;step<steps;step++) {
        const std::string prefix=std::string(argv[2])+"-"+std::to_string(step);
        dump_tensor(prefix+".params.bin",(float*)model.params_memory,model.num_parameters);
        gpt2_forward(&model,inputs.data(),B,T,step,NULL);
        dump_tensor(prefix+".logits.bin",model.acts.output,B*T*V);
        gpt2_backward_and_reduce(&model,inputs.data(),targets.data(),1,0);
        dump_tensor(prefix+".grads.bin",(float*)model.grads_memory,model.num_parameters);
        const float norm=gpt2_calculate_grad_norm(&model,&multi_gpu_config);
        const float scale=norm>1.0f?1.0f/norm:1.0f;
        gpt2_update(&model,lr,.9f,.95f,1e-8f,wd,scale,step+1,&multi_gpu_config);
        dump_tensor(prefix+".updated.bin",(float*)model.params_memory,model.num_parameters);
        dump_tensor(prefix+".moment1.bin",model.m_memory,model.num_parameters);
        dump_tensor(prefix+".moment2.bin",model.v_memory,model.num_parameters);
        printf("PARITY step=%d loss=%.9g norm=%.9g scale=%.9g\n",step,model.mean_loss,norm,scale);
    }
    cudaCheck(cudaFree(output)); gpt2_free(&model); common_free(model);
    multi_gpu_config_free(&multi_gpu_config);
}
