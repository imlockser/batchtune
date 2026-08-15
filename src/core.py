import copy
from collections.abc import Callable, Mapping, Sized
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import OutOfMemoryError, nn, optim
from torch.utils.data import DataLoader, Dataset

from src.utils import (
    empty_device_cache,
    get_device_memory_limit,
    get_device_used_memory,
    get_system_available_memory,
    get_system_total_memory,
    synchronize_device,
)

GIB = 1024**3


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: Callable,
    batch: tuple,
    device: torch.device,
) -> None:
    X, y = batch

    non_blocking = device.type == "cuda"

    X = X.to(device, non_blocking=non_blocking)
    y = y.to(device, non_blocking=non_blocking)

    optimizer.zero_grad()

    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()


def lr_fit(inputs: list[int], targets: list[int]) -> tuple[float, float]:
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)

    slope = sum((x - x.mean()) * (y - y.mean())) / sum((x - x.mean()) ** 2)
    intercept = y.mean() - slope * x.mean()

    return intercept, slope


def find_max_batch_size(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: Callable,
    dataset: Dataset,
    *,
    device: torch.device,
    step: int = 64,
    memory_batches: int = 5,
    device_safety_factor: float = 0.9,
    system_reserve_factor: float = 0.15,
    dataloader_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> int:
    if not isinstance(dataset, Sized):
        raise TypeError("Dataset must implement __len__")

    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    model.to(device)
    model.train()

    def reset_state() -> None:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        optimizer.zero_grad(set_to_none=True)

    def create_dataloader(batch_size: int) -> DataLoader:
        kwargs = {
            **(dataloader_kwargs or {}),
            "dataset": dataset,
            "batch_size": batch_size,
        }
        return DataLoader(**kwargs)

    def train_trial(loader: DataLoader) -> tuple[int, int]:
        device_mem_history = []
        system_mem_history = []

        iterator = iter(loader)

        for _ in range(memory_batches):
            batch = next(iterator)
            train_step(model, optimizer, loss_fn, batch, device)

            synchronize_device(device)

            device_mem_history.append(get_device_used_memory(device))
            system_mem_history.append(get_system_available_memory())

        return max(device_mem_history), min(system_mem_history)

    exponent = 1

    max_dataset_bs = len(dataset) // memory_batches
    max_dataset_bs -= max_dataset_bs % step

    device_mem_limit = get_device_memory_limit(device)
    device_mem_limit = int(device_mem_limit * device_safety_factor)

    system_total_mem = get_system_total_memory()
    system_mem_reserve = int(system_total_mem * system_reserve_factor)

    bs_history = []
    device_mem_history = []
    system_mem_history = []

    device_fit = None
    system_fit = None

    while True:
        empty_device_cache(device)

        current_bs = min(2**exponent, max_dataset_bs)
        loader = create_dataloader(current_bs)

        current_device_mem, current_system_mem = train_trial(loader)

        if verbose:
            print(
                f"{'[probe]':<12}"
                f"{f'bs={current_bs}':<12}"
                f"{f'device={current_device_mem / GIB:.2f}/{device_mem_limit / GIB:.2f} GiB':<32}"
                f"{f'system_available={current_system_mem / GIB:.2f} GiB':<34}"
                f"reserve={system_mem_reserve / GIB:.2f} GiB"
            )

        bs_history.append(current_bs)
        device_mem_history.append(current_device_mem)
        system_mem_history.append(current_system_mem)

        if len(bs_history) >= 2:
            device_b0, device_b1 = lr_fit(bs_history, device_mem_history)
            system_b0, system_b1 = lr_fit(bs_history, system_mem_history)

            if device_b1 > 0:
                device_fit = (device_b0, device_b1)

            if system_b1 < 0:
                system_fit = (system_b0, system_b1)

        if current_bs == max_dataset_bs:
            if (
                current_device_mem < device_mem_limit
                and current_system_mem > system_mem_reserve
            ):
                reset_state()
                return current_bs
            break

        next_bs = min(2 ** (exponent + 1), max_dataset_bs)

        next_device_mem = None
        next_system_mem = None

        if device_fit is not None:
            device_b0, device_b1 = device_fit
            next_device_mem = device_b0 + device_b1 * next_bs

        if system_fit is not None:
            system_b0, system_b1 = system_fit
            next_system_mem = system_b0 + system_b1 * next_bs

        if verbose:
            parts = [
                f"{'[predict]':<12}",
                f"{f'bs={next_bs}':<12}",
            ]

            if next_device_mem is not None:
                parts.append(
                    f"{f'device={next_device_mem / GIB:.2f}/{device_mem_limit / GIB:.2f} GiB':<32}"
                )
            else:
                parts.append(f"{'device=n/a':<32}")

            if next_system_mem is not None:
                parts.append(
                    f"{f'system_available={next_system_mem / GIB:.2f} GiB':<34}"
                )
            else:
                parts.append(f"{'system_available=n/a':<34}")

            parts.append(f"reserve={system_mem_reserve / GIB:.2f} GiB")

            print("".join(parts))

        if next_device_mem is not None and next_device_mem >= device_mem_limit:
            break

        if next_system_mem is not None and next_system_mem <= system_mem_reserve:
            break

        exponent += 1

    max_candidates = []

    if device_fit is not None:
        device_b0, device_b1 = device_fit

        device_max_bs = int((device_mem_limit - device_b0) / device_b1)
        device_max_bs = min(device_max_bs, max_dataset_bs)
        device_max_bs -= device_max_bs % step

        max_candidates.append(device_max_bs)

    if system_fit is not None:
        system_b0, system_b1 = system_fit

        system_max_bs = int((system_mem_reserve - system_b0) / system_b1)
        system_max_bs = min(system_max_bs, max_dataset_bs)
        system_max_bs -= system_max_bs % step

        max_candidates.append(system_max_bs)

    if not max_candidates:
        raise RuntimeError("Unable to estimate memory growth")

    max_bs = min(max_candidates)

    while True:
        empty_device_cache(device)

        try:
            loader = create_dataloader(max_bs)

            current_device_mem, current_system_mem = train_trial(loader)

            if verbose:
                print(
                    f"{'[verify]':<12}"
                    f"{f'bs={max_bs}':<12}"
                    f"{f'device={current_device_mem / GIB:.2f}/{device_mem_limit / GIB:.2f} GiB':<32}"
                    f"{f'system_available={current_system_mem / GIB:.2f} GiB':<34}"
                    f"reserve={system_mem_reserve / GIB:.2f} GiB"
                )

            if (
                current_device_mem < device_mem_limit
                and current_system_mem > system_mem_reserve
            ):
                break
        except OutOfMemoryError:
            if verbose:
                print(f"{'[verify]':<12}{f'bs={max_bs}':<12}{'OOM':<32}")

        max_bs -= step

    reset_state()

    return max_bs


def find_best_batch_size(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: Callable,
    dataset: Dataset,
    *,
    device: torch.device,
    step: int = 64,
    memory_batches: int = 5,
    device_safety_factor: float = 0.9,
    system_reserve_factor: float = 0.15,
    coarse_bins: int = 10,
    patience: int = 3,
    fine_radius: int = 4,
    warmup_batches: int = 3,
    coarse_benchmark_batches: int = 8,
    fine_benchmark_batches: int = 10,
    final_benchmark_batches: int = 10,
    top_k: int = 3,
    final_repeats: int = 3,
    dataloader_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> int:
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    model.to(device)
    model.train()

    def reset_state() -> None:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        optimizer.zero_grad(set_to_none=True)

    def create_dataloader(batch_size: int) -> DataLoader:
        kwargs = {
            **(dataloader_kwargs or {}),
            "dataset": dataset,
            "batch_size": batch_size,
        }
        return DataLoader(**kwargs)

    def benchmark_throughput(loader: DataLoader, benchmark_batches: int) -> float:
        iterator = iter(loader)

        def next_batch():
            nonlocal iterator

            try:
                return next(iterator)
            except StopIteration:
                iterator = iter(loader)
                return next(iterator)

        for _ in range(warmup_batches):
            batch = next_batch()
            train_step(model, optimizer, loss_fn, batch, device)

        synchronize_device(device)

        samples = 0
        start = perf_counter()

        for _ in range(benchmark_batches):
            batch = next_batch()
            train_step(model, optimizer, loss_fn, batch, device)
            samples += len(batch[0])

        synchronize_device(device)

        elapsed = perf_counter() - start
        return samples / elapsed

    if verbose:
        print("\n[stage] Finding maximum safe batch size")

    max_bs = find_max_batch_size(
        model,
        optimizer,
        loss_fn,
        dataset,
        device=device,
        step=step,
        memory_batches=memory_batches,
        device_safety_factor=device_safety_factor,
        system_reserve_factor=system_reserve_factor,
        dataloader_kwargs=dataloader_kwargs,
        verbose=verbose,
    )

    if verbose:
        print(f"[result] max_batch_size={max_bs}")
        print("\n[stage] Coarse throughput search")

    coarse_candidates = np.linspace(step, max_bs, coarse_bins, dtype=int)
    coarse_candidates = (coarse_candidates // step) * step
    coarse_candidates = np.unique(coarse_candidates)[::-1].tolist()

    best_coarse_bs = 0
    best_coarse_throughput = 0
    no_improvement_count = 0

    for bs in coarse_candidates:
        if no_improvement_count >= patience:
            break

        empty_device_cache(device)
        reset_state()

        loader = create_dataloader(bs)

        throughput = benchmark_throughput(loader, coarse_benchmark_batches)

        if throughput > best_coarse_throughput:
            best_coarse_bs = bs
            best_coarse_throughput = throughput
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if verbose:
            print(
                f"{'[coarse]':<12}{f'bs={bs}':<12}throughput={throughput:.2f} samples/sec"
            )

    fine_min = max(
        step,
        best_coarse_bs - fine_radius * step,
    )

    fine_max = min(
        max_bs,
        best_coarse_bs + fine_radius * step,
    )

    fine_candidates = list(range(fine_max, fine_min - 1, -step))

    if verbose:
        print(
            "[result] "
            f"best_coarse_bs={best_coarse_bs} "
            f"throughput={best_coarse_throughput:.2f} samples/sec"
        )
        print(f"\n[stage] Fine throughput search ({fine_min}..{fine_max}, step={step})")

    fine_results = []

    for bs in fine_candidates:
        empty_device_cache(device)
        reset_state()

        loader = create_dataloader(bs)

        throughput = benchmark_throughput(loader, fine_benchmark_batches)

        fine_results.append((bs, throughput))

        if verbose:
            print(
                f"{'[fine]':<12}{f'bs={bs}':<12}throughput={throughput:.2f} samples/sec"
            )

    top_candidates = sorted(
        fine_results, key=lambda candidate: candidate[1], reverse=True
    )[:top_k]

    if verbose:
        print("\n[stage] Final candidate verification")

        for i, (bs, throughput) in enumerate(top_candidates, start=1):
            print(
                f"{'[candidate]':<12}"
                f"{f'rank={i}/{len(top_candidates)}':<12}"
                f"{f'bs={bs}':<12}"
                f"fine_throughput={throughput:.2f} samples/sec"
            )

    results = {bs: [] for bs, _ in top_candidates}

    candidate_bs = list(results)

    for repeat in range(final_repeats):
        shift = repeat % len(candidate_bs)
        rotated = candidate_bs[shift:] + candidate_bs[:shift]

        for bs in rotated:
            empty_device_cache(device)
            reset_state()

            loader = create_dataloader(bs)

            throughput = benchmark_throughput(loader, final_benchmark_batches)

            results[bs].append(throughput)

            if verbose:
                print(
                    f"{'[verify]':<12}"
                    f"{f'round={repeat + 1}/{final_repeats}':<14}"
                    f"{f'bs={bs}':<12}"
                    f"throughput={throughput:.2f} samples/sec"
                )

    best_bs = 0
    best_score = 0.0

    for bs, throughputs in results.items():
        score = float(np.median(throughputs))

        if score > best_score:
            best_bs = bs
            best_score = score

        if verbose:
            print(f"{'[score]':<12}{f'bs={bs}':<12}median={score:.2f} samples/sec")

    if verbose:
        print(
            f"[result] best_bs={best_bs} median_throughput={best_score:.2f} samples/sec"
        )

    reset_state()

    return best_bs
