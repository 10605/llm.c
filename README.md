# 10605: llm.c and μP

- **[μP exercise](HW5_Code/doc/mup/exercise.md):** reproduce learning-rate transfer
  across model widths, then investigate schedules, weight decay and training length.
- **[HW5_Code](HW5_Code/):** the validated standard/μP trainer and experiment tools.

```sh
git clone --branch mup https://github.com/10605/llm.c.git
cd llm.c/HW5_Code
```

The exercise uses final held-out validation loss as its primary metric. μP width
transfer is approximate, and the learning rate can change with the training
horizon and schedule. The documented short experiments do not establish transfer
at the full 124M model's 20-TPP budget.

There is one training codebase in `HW5_Code/`: standard parametrization and μP
are selected with `-me 0` and `-me 1`. **The default course preset enables μP,
fixes heads at 12, and derives width scaling from reference width 192.** See the
[course defaults and student knobs](HW5_Code/doc/mup/exercise.md#course-preset-and-student-knobs).
`-se` selects initialization seed; `-nt` specifies the full token budget and `-et`
stops at an earlier token count while preserving that LR schedule. The integrated trainer retains the course's
architecture, data-path, scheduling, continuation and early-stop controls.

The course recipe uses context 2048, global batch 114688, linear LR decay with
10% warmup, and AdamW timescale `-wt 0.1`. Weight decay adjusts with peak LR and
the full token horizon; explicit `-c` selects fixed decay. The batch extrapolation
is provisional, and the full-budget base LR still requires a matched proxy sweep.

The imported implementation derives from [GatiAher/llm.c](https://github.com/GatiAher/llm.c/tree/mup)
and [karpathy/llm.c](https://github.com/karpathy/llm.c). Attribution and the MIT
license are retained in [HW5_Code/LICENSE](HW5_Code/LICENSE). See the
[source manifest](HW5_Code/doc/mup/source-provenance.json) for the exact snapshot.
