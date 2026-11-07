/* Host-only validation for training options and checkpoint parametrization. */
#ifndef TRAIN_CONFIG_H
#define TRAIN_CONFIG_H

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

inline void require_config(bool condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "Configuration error: %s\n", message);
        exit(EXIT_FAILURE);
    }
}

inline int parse_int(const char* option, const char* value) {
    char* end;
    errno = 0;
    long result = strtol(value, &end, 10);
    require_config(value[0] && end != value && !*end && errno != ERANGE &&
                   result >= INT_MIN && result <= INT_MAX, option);
    return (int)result;
}

inline int parse_positive_int(const char* option, const char* value) {
    int result = parse_int(option, value);
    require_config(result > 0, option);
    return result;
}

inline long long parse_token_budget(const char* option, const char* value) {
    char* end;
    errno = 0;
    long long result = strtoll(value, &end, 10);
    require_config(value[0] && end != value && !*end && errno != ERANGE && result > 0,
                   option);
    return result;
}

inline int course_preset_from_args(int argc, char** argv) {
    int enabled = 1;
    // Resolve defaults before parsing overrides, independent of flag order.
    for (int i = 1; i < argc; i += 2) {
        if (!strcmp(argv[i], "-cp")) {
            require_config(i + 1 < argc, "-cp requires 0 or 1");
            enabled = parse_int("-cp", argv[i + 1]);
            require_config(enabled == 0 || enabled == 1, "-cp requires 0 or 1");
        }
    }
    return enabled;
}

inline int resolve_training_steps(int max_steps, long long token_budget,
                                  size_t dataset_tokens, int total_batch) {
    require_config(total_batch > 0 && max_steps >= -1, "Invalid training budget");
    require_config(token_budget == -1 || token_budget > 0, "Invalid token budget");
    require_config(token_budget == -1 || max_steps == -1, "Use either -nt tokens or -x steps, not both");
    unsigned long long steps = token_budget > 0 ? (unsigned long long)((token_budget - 1) / total_batch + 1) :
        (max_steps == -1 ? dataset_tokens / total_batch : (unsigned long long)max_steps);
    require_config(steps < INT_MAX, "Training budget exceeds the supported step count");
    return (int)steps;
}

inline int resolve_warmup(int requested, int schedule_steps, const char* scheduler) {
    require_config(requested >= -1 && schedule_steps >= 0, "Invalid warmup or schedule length");
    int warmup = requested == -1 ? (!strcmp(scheduler, "constant") ? 0 : schedule_steps / 10) : requested;
    require_config(warmup <= schedule_steps, "Warmup must fit within the full training schedule");
    return warmup;
}

// Power Lines convention: tau is normalized by the FULL schedule at peak LR.
// muP multiplies hidden-weight decay by width and divides LR by width, so
// base_lr * base_decay is also the actual decay product for every decayed tensor.
inline float resolve_weight_decay(float fixed_decay, float timescale, float peak_lr,
                                  int schedule_steps) {
    require_config(isfinite(fixed_decay) && fixed_decay >= 0 &&
                   isfinite(timescale) && (timescale == -1 || timescale > 0),
                   "Invalid weight decay or AdamW timescale");
    if (timescale == -1 || schedule_steps == 0) { return timescale == -1 ? fixed_decay : 0.f; }
    require_config(schedule_steps > 0 && isfinite(peak_lr) && peak_lr > 0,
                   "Timescale-based decay requires positive peak LR and full schedule length");
    double tau_steps = (double)timescale * schedule_steps;
    require_config(tau_steps >= 1.0,
                   "-wt timescale is too short for this schedule; increase -nt or -wt, or set explicit -c");
    float decay = (float)(1.0 / ((double)peak_lr * tau_steps));
    require_config(isfinite(decay) && decay > 0, "Derived weight decay is not representable");
    return decay;
}

inline int resolve_early_stop_steps(int early_steps, long long early_tokens,
                                    long long token_budget, int schedule_steps, int total_batch) {
    require_config(early_steps >= -1 && (early_tokens == -1 || early_tokens > 0), "Invalid early-stop budget");
    require_config(early_tokens == -1 || early_steps == -1, "Use either -et tokens or -es steps, not both");
    if (early_tokens == -1) { return early_steps; }
    require_config(total_batch > 0 && schedule_steps >= 0 && schedule_steps < INT_MAX,
                   "Invalid training schedule");
    require_config(token_budget == -1 || early_tokens <= token_budget, "-et must not exceed -nt");
    require_config(early_tokens <= (long long)schedule_steps * total_batch,
                   "-et must fit within the full training schedule");
    // Absolute token position, including previous updates on checkpoint resume.
    return (int)((early_tokens - 1) / total_batch + 1);
}

inline float parse_float(const char* option, const char* value) {
    char* end;
    errno = 0;
    float result = strtof(value, &end);
    require_config(value[0] && end != value && !*end && errno != ERANGE && isfinite(result), option);
    return result;
}

inline void copy_option(char* dest, size_t capacity, const char* value) {
    require_config(strlen(value) < capacity, "Option string too long");
    strcpy(dest, value);
}

inline bool same_float(float a, float b) {
    return fabsf(a - b) <= 1e-6f * fmaxf(fabsf(a), fabsf(b));
}

inline bool valid_scheduler(const char* name) {
    return !strcmp(name, "cosine") || !strcmp(name, "linear") ||
           !strcmp(name, "constant") || !strcmp(name, "wsd");
}

inline int training_end_step(int start, int schedule_steps, int early_stop, int warmup) {
    require_config(schedule_steps >= 0 && schedule_steps < INT_MAX && warmup >= 0 &&
                   warmup <= schedule_steps, "Warmup must fit within the full training schedule");
    int end = early_stop == -1 ? schedule_steps : early_stop;
    require_config(start >= 0 && end >= start && end <= schedule_steps,
                   "Require starting/resumed step <= resolved early stop <= full schedule (-nt, -x, or one epoch)");
    return end;
}

struct ModelOptions {
    int layers, channels, heads;
    int reference_width; // -mb; zero means unspecified
    bool enabled_set, mult_set, attn_set, coord_width_set;
    bool course_defaults;
    int initialization_seed = 42; // only used when initializing fresh weights
};

template<class Config>
void apply_architecture(Config* config, const ModelOptions& opts, bool checkpoint) {
    require_config(!checkpoint || ((!opts.layers || opts.layers == config->num_layers) &&
                   (!opts.channels || opts.channels == config->channels) &&
                   (!opts.heads || opts.heads == config->num_heads)),
                   "Architecture flags disagree with checkpoint dimensions");
    if (opts.layers) { config->num_layers = opts.layers; }
    if (opts.channels) { config->channels = opts.channels; }
    if (opts.heads) { config->num_heads = opts.heads; }
    else if (opts.course_defaults && !checkpoint) { config->num_heads = 12; }
    // Allocate positional embeddings for the course context before weights are
    // initialized. Saved checkpoint capacities and legacy descriptors are kept.
    if (opts.course_defaults && !checkpoint) { config->max_seq_len = 2048; }
    require_config(config->num_layers > 0 && config->num_heads > 0 && config->channels > 0,
                   "Layers, channels and heads must be positive");
    require_config(config->channels % config->num_heads == 0, "Channels must be divisible by heads");
}

// Extension in previously zero-filled header padding; tensor/state payloads are unchanged.
// Model slots 8..13; optimizer-state slots 40..45.
inline float header_float(const int* slot) {
    float value;
    static_assert(sizeof(float) == sizeof(int), "Checkpoint format requires 32-bit float/int");
    memcpy(&value, slot, sizeof(value));
    return value;
}

inline void write_parametrization(int* header, int enabled, float mult, float attn, float reference) {
    header[0] = 0x4d555031; // MUP1
    header[1] = 1;
    header[2] = enabled;
    memcpy(header + 3, &mult, sizeof(mult));
    memcpy(header + 4, &attn, sizeof(attn));
    memcpy(header + 5, &reference, sizeof(reference));
}

inline void validate_parametrization(const int* header) {
    require_config(header[0] == 0x4d555031 && header[1] == 1,
                   "Unknown checkpoint parametrization metadata");
    require_config(header[2] == 0 || header[2] == 1, "Invalid checkpoint parametrization");
    for (int i = 3; i <= 5; i++) {
        float value = header_float(header + i);
        require_config(isfinite(value) && value > 0, "Invalid checkpoint muP multiplier/reference");
    }
}

inline void resolve_parametrization(int channels, const ModelOptions& opts, const int* header,
                                   bool resuming, int* enabled, float* mult, float* attn, float* reference) {
    bool saved = header && header[0] != 0;
    if (saved) {
        validate_parametrization(header);
        require_config(!opts.enabled_set || *enabled == header[2], "-me conflicts with checkpoint");
        require_config(!opts.mult_set || same_float(*mult, header_float(header + 3)), "-mm conflicts with checkpoint");
        require_config(!opts.attn_set || same_float(*attn, header_float(header + 4)), "-ma conflicts with checkpoint");
        require_config(!opts.reference_width || same_float((float)opts.reference_width, header_float(header + 5)),
                       "-mb conflicts with checkpoint");
        *enabled = header[2];
        *mult = header_float(header + 3);
        *attn = header_float(header + 4);
        *reference = header_float(header + 5);
        require_config(same_float((float)channels / *mult, *reference), "Checkpoint width/reference mismatch");
        return;
    }
    if (header && (resuming || *enabled)) {
        require_config(opts.enabled_set && (!*enabled || (opts.mult_set && opts.attn_set)),
                       "Legacy checkpoint has no muP metadata: explicitly supply -me (and -mm, -ma for muP)");
    }
    // Implicit course defaults must never conflict with checkpoint metadata or
    // override an explicitly selected legacy multiplier/reference convention.
    int reference_width = opts.reference_width;
    if (!header && opts.course_defaults && *enabled && !reference_width && !opts.mult_set) {
        reference_width = 192;
    }
    require_config(!*enabled || !opts.channels || reference_width || opts.mult_set,
                   "Custom muP channels require -mb reference width or explicit -mm");
    if (reference_width) {
        float derived = (float)channels / reference_width;
        require_config(!opts.mult_set || same_float(*mult, derived), "-mm disagrees with channels / -mb");
        if (!opts.mult_set) { *mult = derived; }
        *reference = (float)reference_width;
    } else {
        *reference = (float)channels / *mult;
        require_config(isfinite(*reference) && *reference >= 1 && same_float(*reference, roundf(*reference)),
                       "-mm must imply a positive integral reference width (channels / -mm)");
    }
    require_config((*enabled == 0 || *enabled == 1) && isfinite(*mult) && *mult > 0 &&
                   isfinite(*attn) && *attn > 0, "Invalid muP settings");
}

#endif
