# Build the HW5 student AMI

Instructor baseline-build procedure, updated 2026-11-06. Students should use the
prepared image named in the handout and `~/hw5/START-HERE.md`, not rebuild the AMI.
The v3 release adds the shared HF pretraining training and validation data and
dual-validation logging to v2. Its exact installed files and acceptance results
are in `/opt/hw5/release`; v2 evidence is preserved in `/opt/hw5/release-v2`.
Preserve the student helper and SFT additions when making subsequent images.
Run baseline build steps below only on a temporary builder.
The generated image includes a student quick start at `~/hw5/START-HERE.md` and
its build and acceptance records under `/opt/hw5/release`.

AWS confirmed `ami-01924b6996b062ee3` is available in **us-east-1**. Its name is
**Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20250424**,
architecture x86_64, owner `898082745236`, root EBS size 75 GiB, and G5 support.
That is the base image, not the finished student AMI. It is an older image, so
apply Ubuntu updates and recheck the GPU before building.

Include the executable/source, protected seed, tokenizer, HF pretraining training
and validation shards, and SFT environment. Students download their selected cluster
for Part 2; Part 1 needs no data download.
Do not include the completed expert/generalist or instructor credentials.

## 1. Launch and connect

In EC2, select **us-east-1**, launch the AMI above as an **On-Demand g5.xlarge**,
name it `hw5-ami-builder`, and use a new or existing SSH key. Restrict SSH to your
IP. Increase its root **gp3 EBS** volume to **100 GiB**. Use the EBS filesystem for
all course assets: instance-store contents do not become part of the AMI.

For cross-account distribution, decide encryption now. A snapshot encrypted with
the default AWS-managed EBS KMS key cannot be shared. Use a customer-managed key
with the required student-account permissions, or unencrypted course-only storage
if permitted by your course policy. No sensitive credentials belong in either.

On your Mac, replace the two placeholders, then connect:

```bash
AWS_KEY="$HOME/Downloads/your-key.pem"
AWS_HOST="YOUR_INSTANCE_PUBLIC_DNS"
chmod 400 "$AWS_KEY"
ssh -i "$AWS_KEY" "ubuntu@$AWS_HOST"
```

Run the remaining Linux commands as `ubuntu` unless marked **Mac**. Stop if a
command fails rather than continuing to the imaging step.

## 2. System packages and CUDA toolkit

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential clang git curl wget unzip xz-utils \
  ca-certificates python3 python3-venv libomp-dev zlib1g-dev tmux
sudo reboot
```

Reconnect after the reboot. Verify the GPU, then select CUDA 12.8 explicitly:

```bash
nvidia-smi
test -x /usr/local/cuda-12.8/bin/nvcc
```

If that toolkit is absent, install the toolkit (not an unrelated driver stack):

```bash
cd /tmp
curl -fLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

Persist the compiler path and verify it:

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> "$HOME/.bashrc"
export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc --version
mkdir -p "$HOME/hw5"
```

`nvidia-smi` must show the A10G. Its displayed CUDA version describes driver
capability; `nvcc --version` identifies the selected compiler toolkit.

## 3. Install the trainer's cuDNN and source

Keep cuDNN 9.2.1.18 in its own directory. The binary below records that library
path, avoiding a global override of PyTorch's libraries.

```bash
cd /tmp
curl -fLO https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.2.1.18_cuda12-archive.tar.xz
tar -xf cudnn-linux-x86_64-9.2.1.18_cuda12-archive.tar.xz
sudo mkdir -p /opt/hw5/cudnn
sudo cp -a cudnn-linux-x86_64-9.2.1.18_cuda12-archive/include /opt/hw5/cudnn/
sudo cp -a cudnn-linux-x86_64-9.2.1.18_cuda12-archive/lib /opt/hw5/cudnn/

git clone --depth 1 --branch v1.5.2 https://github.com/NVIDIA/cudnn-frontend.git "$HOME/cudnn-frontend"
cd "$HOME/hw5"
git clone --branch mup https://github.com/10605/llm.c.git
cd llm.c
git checkout 495d43172e69c85c2119a05dfe0f956340929cf7
```

This commit contains the published course defaults. The updated SFT additions and student helper overlay
prepared with this guide are separate local files until published; copy the
prepared additions archive in step 4. Do not use the old `code.zip` unmodified:
its model reader ignores muP metadata and its generator defaults to interaction.

## 4. Copy the protected seed and updated SFT package

In a **second terminal on your Mac**, with `AWS_KEY` and `AWS_HOST` set as above:

```bash
HW5_LOCAL="/Users/jrast/Documents/F26 10605/hw5"
SEED_LOCAL="$HW5_LOCAL/llm.c-10605/modal-results/20261104T192825Z-climbmix-seed-fresh/halfway-seed"
scp -i "$AWS_KEY" -r "$SEED_LOCAL" "ubuntu@$AWS_HOST:/home/ubuntu/hw5/"
scp -i "$AWS_KEY" \
  "$HW5_LOCAL/output/ami/hw5-ami-additions.tar.gz" \
  "$HW5_LOCAL/output/ami/hw5-ami-additions.tar.gz.sha256" \
  "ubuntu@$AWS_HOST:/home/ubuntu/hw5/"
```

For v2, also copy the student helper overlay and its checksum (the overlay
archive is rooted at the repository, not `HW5_Code/`):

```bash
scp -i "$AWS_KEY" \
  "$HW5_LOCAL/output/ami/build-20261105T084500Z-v2/student-overlay.tar.gz" \
  "$HW5_LOCAL/output/ami/build-20261105T084500Z-v2/student-overlay.tar.gz.sha256" \
  "ubuntu@$AWS_HOST:/home/ubuntu/hw5/"
```

Back on **EC2**:

```bash
cd "$HOME/hw5"
sha256sum -c hw5-ami-additions.tar.gz.sha256
tar -xzf hw5-ami-additions.tar.gz -C llm.c/HW5_Code
cd llm.c/HW5_Code
sha256sum -c AMI_ADDITIONS_SHA256SUMS
cd "$HOME/hw5"
sha256sum -c student-overlay.tar.gz.sha256
tar -xzf student-overlay.tar.gz -C llm.c

sudo cp -a "$HOME/hw5/halfway-seed" /opt/hw5/seed
sudo chown -R root:root /opt/hw5/seed
sudo chmod -R a-w /opt/hw5/seed
```

Verify the two large files against their already audited hashes:

```bash
cd /opt/hw5/seed
sha256sum -c <<'CHECKSUMS'
a5633cb587e8b67a772fbbcf8c615659adbe4eb936421355679226abe304af1e  model_00010919.bin
90e75eee93b288c82dc4b534cfa5538081e968682f11f2d5358764b554fd7c82  state_00010919_00000.bin
CHECKSUMS
test -f DONE_00010919
```

All three checkpoint files are needed for expert continuation. The seed contains
10,919 updates, 1,252,278,272 tokens and a 2048-token context. Preserve its JSON
metadata as provenance. Its historical README describes the instructor's frozen
run; use the updated repository instructions for student LR choices.

## 5. Compile for the A10G

```bash
cd "$HOME/hw5/llm.c/HW5_Code"
make train_gpt2cu PRECISION=BF16 USE_CUDNN=1 GPU_COMPUTE_CAPABILITY=86 \
  NO_MULTI_GPU=1 NO_USE_MPI=1 \
  NVCC_INCLUDES="-I$HOME/cudnn-frontend/include -I/opt/hw5/cudnn/include" \
  NVCC_LDFLAGS="-lcublas -lcublasLt -lnvidia-ml -L/opt/hw5/cudnn/lib -lcudnn -Xlinker -rpath -Xlinker /opt/hw5/cudnn/lib"
ldd ./train_gpt2cu
```

Check that no library says `not found`, and that cuDNN resolves to
`/opt/hw5/cudnn/lib`. Keep the source and these build options so students can
rebuild. Do not copy an H100-only executable into the AMI.

## 6. Miniconda, Python and SFT

If `conda` already works, reuse it. Otherwise install it as the `ubuntu` user:

```bash
cd /tmp
curl -fLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
conda config --set auto_activate_base false
```

Use the official installer checksum listing if verifying the installer before
execution. Create the environment and preload the tokenizer cache:

```bash
conda create -n hw5-sft --override-channels -c conda-forge python=3.11 pip -y
conda activate hw5-sft
cd "$HOME/hw5/llm.c/HW5_Code"
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r sft/requirements.txt
python -m pip install huggingface_hub
python -m pip install -r requirements-student.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -c 'import tiktoken; tiktoken.get_encoding("gpt2")'
python -m pip freeze > "$HOME/hw5/python-packages.txt"
```

Keep custom cuDNN directories out of `LD_LIBRARY_PATH` when running PyTorch.
Check `aws --version` as well. If AWS CLI v2 is absent, install it using the
[official AWS CLI instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
No saved instructor AWS/Hugging Face/GitHub credentials should be required by
the student image; use public data or each student's own authorized access.

## 7. Prepare tokenizer and shared pretraining data

Include all 162 pretraining training shards and all 32 pretraining validation shards,
unchanged from the pinned HF release. Do not construct a separate validation set.
Run from `HW5_Code/` with `hw5-sft` active:

```bash
python - <<'PYDATA'
import hashlib, json, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

manifest = json.loads(Path('/opt/hw5/seed/dataset.json').read_text())
pretraining = manifest['pretraining']
out = Path('data/pretraining')
out.mkdir(parents=True, exist_ok=True)
from huggingface_hub import HfApi
files = [x for x in HfApi().list_repo_tree(pretraining['dataset'], path_in_repo='llmc/pretraining',
    repo_type='dataset', revision=pretraining['revision']) if x.path.endswith('.bin')]
for entry in files:
    record = {'repository_path': entry.path, 'name': Path(entry.path).name, 'sha256': entry.lfs.sha256}
    source = hf_hub_download(pretraining['dataset'], record['repository_path'],
        repo_type='dataset', revision=pretraining['revision'])
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == record['sha256']
    shutil.copyfile(source, out / record['name'])
tokenizer = hf_hub_download('karpathy/llmc-starter-pack', 'gpt2_tokenizer.bin', repo_type='dataset')
assert hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest() == pretraining['tokenizer_sha256']
shutil.copyfile(tokenizer, 'data/gpt2_tokenizer.bin')
print('HF pretraining training and validation data verified')
PYDATA
```

For the builder test, download cluster 01. Students change `CLUSTER` to their
assigned cluster. The dataset revision remains fixed:

```bash
CLUSTER=cluster_01
hf download jrast/climbmix-students-tokenized --repo-type dataset \
  --revision f30c5f5ea5085e7cc39212be709521effafd6003 \
  --include "llmc/$CLUSTER/*.bin" --local-dir data/downloads
```

The training glob is now `data/downloads/llmc/cluster_01/climbmix_students_train_*.bin`.
Students can use their cluster's held-out shard with `-iv` when measuring
specialization. Add `-iv2 'data/pretraining/*val*.bin'` for the common
pretraining validation curve. Both use HF validation files, at the same `-v`
interval and `-m` microbatch limit.

## 8. Test expert continuation and resumption

These commands run **20 then 10 updates**, not the full experiment. They use an
explicit test LR. Students choose their own `-l` for actual expert training.

```bash
cd "$HOME/hw5/llm.c/HW5_Code"
mkdir -p runs/ami-smoke
cp /opt/hw5/seed/model_00010919.bin \
   /opt/hw5/seed/state_00010919_00000.bin \
   /opt/hw5/seed/DONE_00010919 runs/ami-smoke/

TEST_LR=0.0012
TRAIN_GLOB='data/downloads/llmc/cluster_01/climbmix_students_train_*.bin'
./train_gpt2cu -l "$TEST_LR" -o runs/ami-smoke -y 1 -rd 1 \
  -id "$TRAIN_GLOB" -es 10939 -n 10 -v 10 \
  -iv 'data/downloads/llmc/cluster_01/*val*.bin' -iv2 'data/pretraining/*val*.bin'

./train_gpt2cu -l "$TEST_LR" -o runs/ami-smoke -y 1 \
  -id "$TRAIN_GLOB" -es 10949 -n 10 -v 10 \
  -iv 'data/downloads/llmc/cluster_01/*val*.bin' -iv2 'data/pretraining/*val*.bin'
```

Check that the first command starts at 10919 and the second at 10939; the
schedule remains 21838 updates with 2183 warmup updates, microbatch 8 and global
batch 114688. Losses/gradient norms must be finite and complete checkpoint triples
must exist at the stopping points. `-rd 1` appears only on the first dataset
switch. Recheck the protected seed hashes after these tests.

## 9. Test the SFT code and one-sequence generation

```bash
python sft/test_sft.py --checkpoint /opt/hw5/seed/model_00010919.bin -v
python sft/generate.py --model /opt/hw5/seed/model_00010919.bin \
  --mode completion --prompt 'The experiment showed that' \
  --max-new-tokens 32 --device cuda --seed 42

python sft/chat_sft.py --model /opt/hw5/seed/model_00010919.bin \
  --out-dir runs/ami-sft-smoke --device cuda --compile 0 --lr 0.0001 \
  --device-batch-size 1 --max-seq-len 1024 --total-batch-size 1024 \
  --num-train-convs 32 --long-convs truncate --num-iterations 2 \
  --eval-every 1 --eval-rows 2 --sample-every 0 --data-workers 1

python sft/generate.py --model runs/ami-sft-smoke/model_sft_00002.bin \
  --mode chat --prompt 'Explain why the sky is blue.' \
  --max-new-tokens 32 --device cuda --seed 42
```

Require finite losses, a saved SFT model, and one sequence followed by exit from
each generation command. This checks installation and model compatibility, not
SFT quality. Before publication also check the actual intended SFT microbatch,
context and compilation setting on the g5; a small smoke test does not establish
that a larger batch fits. Students do not need interactive chatting.

## 10. Record the environment and clean the builder

Save the Git commit, additions archive/hash, build command, `nvcc --version`,
`nvidia-smi`, package list and test results with the image-release notes. Keep
the protected seed, shared pretraining training and validation files, tokenizer
and installed software.
Remove the temporary test runs, builder's cluster training download, duplicate
seed copy, installers and unnecessary download caches once the checks pass.

Before sharing, remove instructor SSH authorized keys/private keys, shell
history and any AWS/GitHub/Hugging Face credential files or tokens. Retain
cloud-init so new instances receive their own launch key and unique SSH host
keys. Follow AWS's shared-Linux-AMI preparation guide for host-key and instance
identity cleanup; do this last, since cleanup may remove your SSH access.
Do not copy your entire Mac home directory onto the builder.

## 11. Create, launch-test and share

After cleaning SSH keys and cloud-init identity, shut down the builder. Create
the image from that **stopped** instance; do not reboot the sanitized builder
before capture, since its old launch key could be installed again.
In EC2: **Instances → select builder → Actions → Image and templates → Create image**.
Name it `hw5-f26-v1`. Wait for the AMI state to become `available`; image creation
time varies with snapshots and volume size.

Launch a new **g5.xlarge** from the new AMI with a different SSH key, ideally in
a student account. Verify login, GPU, seed hashes, Conda activation, the assigned
cluster download, short trainer continuation/resume and single-sequence generation.
This fresh-instance test is required: a working builder alone is insufficient.

Then use **AMIs → Actions → Edit AMI permissions → Private → Add account ID**
to grant student accounts access. Share the KMS key too if applicable. Publish
the **new AMI ID and us-east-1**, instance type, disk requirement, cluster-download
command and `conda activate hw5-sft`. Do not publish the old base AMI ID as the
finished image. Terminate the builder/test instances when no longer needed;
retain the final AMI and backing snapshots.

## References and validation limits

- [AWS: create an EBS-backed AMI](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-an-ami-ebs.html)
- [AWS: prepare a shared Linux AMI](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/building-shared-amis.html)
- [AWS: share with specific accounts and KMS constraints](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/sharingamis-explicit.html)
- [NVIDIA: CUDA 12.8 installation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/)
- [NVIDIA: cuDNN 9.2.1 installation](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.2.1/installation/linux.html)
- [Conda installation/checksums](https://docs.conda.io/projects/conda/en/stable/user-guide/install/)
- [PyTorch 2.10 CUDA 12.8 wheels](https://pytorch.org/get-started/previous-versions/)
- [Hugging Face pinned downloads](https://huggingface.co/docs/huggingface_hub/guides/download)

The build's acceptance record must cover CPU regression checks with the real
seed, A10G seed evaluation, training/save/resume, SFT/save/load/generation, and
the intended SFT batch/compilation settings. The fresh-AMI launch test establishes
that the captured image also works with a new instance identity and SSH key.
Short compatibility checks do not establish full-run stability or SFT quality.
