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
                   "Require starting/resumed step <= -es <= full schedule (-x or one epoch)");
    return end;
}

struct ModelOptions {
    int layers, channels, heads;
    int reference_width; // -mb; zero means unspecified
    bool enabled_set, mult_set, attn_set, coord_width_set;
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
    require_config(!*enabled || !opts.channels || opts.reference_width || opts.mult_set,
                   "Custom muP channels require -mb reference width or explicit -mm");
    if (opts.reference_width) {
        float derived = (float)channels / opts.reference_width;
        require_config(!opts.mult_set || same_float(*mult, derived), "-mm disagrees with channels / -mb");
        if (!opts.mult_set) { *mult = derived; }
        *reference = (float)opts.reference_width;
    } else {
        *reference = (float)channels / *mult;
        require_config(isfinite(*reference) && *reference >= 1 && same_float(*reference, roundf(*reference)),
                       "-mm must imply a positive integral reference width (channels / -mm)");
    }
    require_config((*enabled == 0 || *enabled == 1) && isfinite(*mult) && *mult > 0 &&
                   isfinite(*attn) && *attn > 0, "Invalid muP settings");
}

#endif
