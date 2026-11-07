// Fixed validation batches drawn across the supplied split, without rewriting data.
#ifndef VALIDATION_H
#define VALIDATION_H
#include <limits.h>
#include "dataloader.h"

typedef struct {
    int count;
    size_t *shards;
    size_t *samples;
} ValidationPlan;

ValidationPlan validation_plan_init(DataLoader *loader, int maximum, int spread) {
    assert(!loader->should_shuffle && (maximum == -1 || maximum > 0));
    size_t n = loader->glob_result.gl_pathc;
    size_t *ends = (size_t*)mallocCheck(n * sizeof(size_t));
    size_t total = 0;
    for (size_t shard = 0; shard < n; ++shard) {
        dataloader_load_shard_(loader, (int)shard);
        total += loader->shard_num_samples;
        ends[shard] = total;
    }
    size_t count = maximum == -1 || (size_t)maximum > total ? total : (size_t)maximum;
    if (count == 0 || count > INT_MAX) {
        fprintf(stderr, "Invalid number of validation batches\n");
        exit(EXIT_FAILURE);
    }
    ValidationPlan plan = {(int)count, (size_t*)mallocCheck(count*sizeof(size_t)),
                          (size_t*)mallocCheck(count*sizeof(size_t))};
    size_t shard = 0;
    for (size_t i = 0; i < count; ++i) {
        // Midpoints of equal-sized intervals in the ordered complete-batch list.
        // If count == total, this visits every complete batch exactly once.
        size_t index = spread ? (size_t)(((long double)i + 0.5L) * total / count) : i;
        while (index >= ends[shard]) ++shard;
        plan.shards[i] = shard;
        plan.samples[i] = index - (shard ? ends[shard-1] : 0);
    }
    free(ends);
    dataloader_reset(loader);
    return plan;
}

void validation_load_batch(DataLoader *loader, const ValidationPlan *plan, int index) {
    assert(index >= 0 && index < plan->count);
    if (loader->current_shard_idx != plan->shards[index]) {
        dataloader_resume(loader, plan->shards[index], plan->samples[index]);
    } else {
        loader->current_sample_idx = plan->samples[index];
    }
    dataloader_load_batch(loader);
}

void validation_plan_free(ValidationPlan *plan) {
    free(plan->shards);
    free(plan->samples);
}
#endif
