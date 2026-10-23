"""Boundary tests for sequential training batches (gh #11)."""

from __future__ import annotations

import unittest

from mainrun.batching import (
    batch_span,
    batches_per_epoch,
    next_batch_ptr,
    require_enough_tokens,
)


class BatchArithmeticTests(unittest.TestCase):
    def test_exact_fit_is_a_complete_batch(self) -> None:
        # C = 16 needs C + 1 tokens; 49 tokens therefore has three batches.
        self.assertEqual(batches_per_epoch(49, 4, 4), 3)

    def test_exact_block_count_without_the_extra_target_is_not_a_batch(self) -> None:
        self.assertEqual(batches_per_epoch(32, 4, 4), 1)
        self.assertEqual(batches_per_epoch(16, 4, 4), 0)

    def test_starts_match_one_sequential_pass(self) -> None:
        def starts(token_count: int) -> list[int]:
            ptr = 0
            seen = []
            for _ in range(batches_per_epoch(token_count, 4, 4)):
                seen.append(ptr)
                ptr = next_batch_ptr(ptr, token_count, 4, 4)
                ptr += 16
            return seen

        self.assertEqual(starts(49), [0, 16, 32])
        self.assertEqual(starts(32), [0])
        self.assertEqual(starts(50), [0, 16, 32])

    def test_wrap_happens_only_when_the_next_span_does_not_fit(self) -> None:
        self.assertEqual(next_batch_ptr(32, 49, 4, 4), 32)
        self.assertEqual(next_batch_ptr(48, 49, 4, 4), 0)
        self.assertEqual(next_batch_ptr(17, 32, 4, 4), 0)

    def test_minimum_tokens_are_validated(self) -> None:
        self.assertEqual(batch_span(4, 4), 17)
        require_enough_tokens(17, 4, 4)
        with self.assertRaisesRegex(ValueError, "at least 17"):
            require_enough_tokens(16, 4, 4)

    def test_non_positive_sizes_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            batch_span(0, 4)
        with self.assertRaises(ValueError):
            batch_span(4, 0)


if __name__ == "__main__":
    unittest.main()
