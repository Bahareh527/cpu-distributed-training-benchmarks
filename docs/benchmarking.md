# Benchmarking protocol

Distributed timing is easy to misinterpret. Use this protocol before making performance claims.

## Controlled variables

Keep the following constant across methods:

- model architecture and initialization seed;
- training and test samples;
- global batch size and learning rate;
- number of epochs;
- CPU thread budget per worker;
- PyTorch version and build;
- operating system and hardware power mode.

`global_batch_size` is divided across workers. For example, a global batch of 128 means a local
batch of 64 with two workers and 32 with four workers.

## Timing definition

Each epoch starts after a worker barrier. Distributed epoch time is the maximum wall-clock training
time across workers. Evaluation and CSV writing are outside the timed section.

For the centralized method, worker count excludes the dedicated server process. This makes the
communication topology clear but must be considered when comparing total CPU use.

## Recommended experiment

1. Run one untimed warm-up epoch.
2. Run at least five measured repetitions for each method and worker count.
3. Randomize method order to reduce background-load bias.
4. Record processor model, physical/logical core counts, memory, OS, and PyTorch version.
5. Report median epoch time and an uncertainty interval, not only the fastest run.
6. Confirm accuracy and loss remain comparable before interpreting speedup.

Speedup is calculated as:

```text
speedup(N) = mean_single_epoch_time / mean_distributed_epoch_time(N)
```

Values above one indicate improvement relative to the measured single-process baseline under the
same configuration.
