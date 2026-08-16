# BatchTune

BatchTune is a lightweight PyTorch utility for finding either the maximum safe batch size or the batch size with the best measured training throughput.

- **Safe maximum batch size** — estimates the memory limit without intentionally triggering OOM.
- **Best batch size** — benchmarks throughput and selects the most efficient batch size for the current training pipeline and hardware.
- Supports **CUDA** and **MPS**.
- Allows passing custom **DataLoader** parameters without changing the usual PyTorch workflow.
- **Verbose output** — shows memory probes and throughput benchmarks for monitoring and manual analysis.

## Usage

```python
best_bs = find_best_batch_size(
    model,
    optimizer,
    loss_fn,
    dataset,
    device=device,
    verbose=True,
)
```

Created by [Denys Bondarchuk](https://github.com/imlockser). Feel free to reach out or contribute to the project.
