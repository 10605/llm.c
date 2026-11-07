"""Exercise the actual validation reader with unequal shards and two ranks."""
from pathlib import Path
import os
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ValidationSampling(unittest.TestCase):
    def test_split_coverage_full_pass_repeatability_and_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            # 1, 3 and 2 complete batches; the final lookahead must not wrap.
            for shard,count in enumerate([1,3,2]):
                tokens=list(range(shard*100,shard*100+count*8+1))
                (tmp/f'val_{shard}.bin').write_bytes(
                    struct.pack('<256i',20240520,1,len(tokens),*([0]*253))+
                    struct.pack('<'+'H'*len(tokens),*tokens))
            source=r'''
#include "llmc/validation.h"
int main() {
    for (int rank=0; rank<2; ++rank) {
        DataLoader loader;
        dataloader_init(&loader,"val_*.bin",1,4,rank,2,0);
        ValidationPlan sampled=validation_plan_init(&loader,3,1);
        assert(sampled.count==3);
        int expected[]={100,116,208}; // global batch indices 1, 3 and 5
        for (int repeat=0; repeat<2; ++repeat) {
            dataloader_reset(&loader);
            for (int i=0;i<3;++i) {
                validation_load_batch(&loader,&sampled,i);
                assert(loader.inputs[0]==expected[i]+rank*4);
                assert(loader.targets[3]==expected[i]+rank*4+4);
            }
        }
        ValidationPlan all=validation_plan_init(&loader,-1,1);
        ValidationPlan capped=validation_plan_init(&loader,100,1);
        assert(all.count==6 && capped.count==6);
        int complete[]={0,100,108,116,200,208};
        for(int i=0;i<6;++i) {
            validation_load_batch(&loader,&all,i);
            assert(loader.inputs[0]==complete[i]+rank*4);
            assert(capped.shards[i]==all.shards[i] && capped.samples[i]==all.samples[i]);
        }
        ValidationPlan legacy=validation_plan_init(&loader,2,0);
        validation_load_batch(&loader,&legacy,0);assert(loader.inputs[0]==rank*4);
        validation_load_batch(&loader,&legacy,1);assert(loader.inputs[0]==100+rank*4);
        validation_plan_free(&sampled);validation_plan_free(&all);
        validation_plan_free(&capped);validation_plan_free(&legacy);dataloader_free(&loader);
    }
}
'''
            (tmp/'test.cpp').write_text(source)
            subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O1',
                '-fsanitize=address,undefined','-I',str(ROOT),str(tmp/'test.cpp'),'-o',str(tmp/'test')],check=True)
            subprocess.run([str(tmp/'test')],cwd=tmp,check=True)


if __name__=='__main__':unittest.main()
