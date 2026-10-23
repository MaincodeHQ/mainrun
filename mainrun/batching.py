"""Batch arithmetic shared by training and tests.

A batch needs ``block_size * batch_size + 1`` tokens so the shifted targets
still have a full final window. Keeping the boundary decisions here makes the
exact-fit cases testable without importing torch or the dataset stack.
"""

from __future__ import annotations


def batch_span(block_size: int, batch_size: int) -> int:
    if block_size < 1 or batch_size < 1:
        raise ValueError("block_size and batch_size must be positive")
    return block_size * batch_size + 1


def batches_per_epoch(token_count: int, block_size: int, batch_size: int) -> int:
    """Number of complete shifted batches in one sequential pass.

    ``token_count // C`` overcounts an exact fit: a split of ``C + 1`` tokens
    has one batch, and a split of exactly ``C`` tokens has none.
    """
    span = batch_span(block_size, batch_size)
    if token_count < span:
        return 0
    return (token_count - 1) // (span - 1)


def require_enough_tokens(token_count: int, block_size: int, batch_size: int) -> None:
    span = batch_span(block_size, batch_size)
    if token_count < span:
        raise ValueError(
            f"training split has {token_count} tokens; at least {span} are needed "
            f"for one batch of block_size={block_size} x batch_size={batch_size}"
        )


def next_batch_ptr(ptr: int, token_count: int, block_size: int, batch_size: int) -> int:
    """Wrap only when the next span does not fit, not when it exactly fits."""
    span = batch_span(block_size, batch_size)
    if ptr + span > token_count:
        return 0
    return ptr
