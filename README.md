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
are selected with `-me 0` and `-me 1`. The integrated trainer retains the course's
architecture, data-path, scheduling, continuation and early-stop controls.

The imported implementation derives from [GatiAher/llm.c](https://github.com/GatiAher/llm.c/tree/mup)
and [karpathy/llm.c](https://github.com/karpathy/llm.c). Attribution and the MIT
license are retained in [HW5_Code/LICENSE](HW5_Code/LICENSE). See the
[source manifest](HW5_Code/doc/mup/source-provenance.json) for the exact snapshot.
