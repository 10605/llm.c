"""Exercise the real C++ loader across checkpoints, unequal shards and epochs."""
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CPP = r'''
#include "llmc/dataloader.h"
void tail(FILE *f, DataLoader *l) {
    if (l->should_shuffle) {
        fwriteCheck(&l->glob_result.gl_pathc, sizeof(size_t), 1, f);
        fwriteCheck(l->shard_indices, sizeof(int), l->glob_result.gl_pathc, f);
        fwriteCheck(&l->shard_num_samples, sizeof(size_t), 1, f);
        fwriteCheck(l->intra_shard_indices, sizeof(int), l->shard_num_samples, f);
        fwriteCheck(&l->shuffle_rng, sizeof(mt19937_state), 1, f);
    }
    rewind(f);
}
void equal_next(DataLoader *a, DataLoader *b, int count) {
    for (int i=0; i<count; ++i) {
        dataloader_next_batch(a); dataloader_next_batch(b);
        assert(memcmp(a->inputs,b->inputs,4*sizeof(int))==0);
        assert(memcmp(a->targets,b->targets,4*sizeof(int))==0);
        assert(a->current_shard_idx==b->current_shard_idx);
        assert(a->current_sample_idx==b->current_sample_idx);
    }
}
int main(int argc,char **argv) {
    for (int shuffle=0;shuffle<=1;++shuffle) {
        // Every offset covers short/long shards, exact shard ends and epochs.
        for (int position=0;position<80;++position) {
            DataLoader a={},b={};
            dataloader_init(&a,argv[1],1,4,0,1,shuffle);
            dataloader_init(&b,argv[1],1,4,0,1,shuffle);
            for (int i=0;i<position;++i) dataloader_next_batch(&a);
            FILE *f=tmpfile(); assert(f); tail(f,&a);
            dataloader_restore_state(&b,f,shuffle,a.current_shard_idx,a.current_sample_idx,0);
            equal_next(&a,&b,100); fclose(f);
            dataloader_free(&a); dataloader_free(&b);
        }
    }
    DataLoader old={},fresh={},switched={};
    dataloader_init(&old,argv[1],1,4,0,1,1);
    for (int i=0;i<37;++i) dataloader_next_batch(&old);
    dataloader_init(&fresh,argv[2],1,4,0,1,1);
    dataloader_init(&switched,argv[2],1,4,0,1,1);
    FILE *f=tmpfile(); assert(f); tail(f,&old);
    dataloader_restore_state(&switched,f,1,old.current_shard_idx,old.current_sample_idx,1);
    equal_next(&fresh,&switched,71); fclose(f);
    // A later resume of the new dataset must continue, not reset again.
    DataLoader resumed={}; dataloader_init(&resumed,argv[2],1,4,0,1,1);
    f=tmpfile(); assert(f); tail(f,&switched);
    dataloader_restore_state(&resumed,f,1,switched.current_shard_idx,switched.current_sample_idx,0);
    equal_next(&resumed,&switched,100); fclose(f);
    dataloader_free(&old); dataloader_free(&fresh);
    dataloader_free(&switched); dataloader_free(&resumed);
    puts("160 resume positions plus dataset switch/re-resume passed");
}
'''


class DatasetResume(unittest.TestCase):
    def test_real_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for dataset, lengths, offset in [('old', (13, 25, 41), 0), ('new', (21, 33), 1000)]:
                for i, n in enumerate(lengths):
                    tokens = [offset + 100*i + j for j in range(n)]
                    header = [20240520, 1, n] + [0]*253
                    (root/f'{dataset}-{i}.bin').write_bytes(struct.pack('<256i', *header)+struct.pack(f'<{n}H', *tokens))
            (root/'test.cpp').write_text(CPP)
            subprocess.run(['c++', '-std=c++17', '-O1', '-fsanitize=address,undefined', '-I', str(ROOT), str(root/'test.cpp'), '-o', str(root/'test')], check=True)
            subprocess.run([str(root/'test'), str(root/'old-*.bin'), str(root/'new-*.bin')], check=True)


if __name__ == '__main__':
    unittest.main()
