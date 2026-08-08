"""Performance / throughput test for the scoring service."""
import time

import numpy as np


def test_batch_throughput(service, applicant_batch):
    """Score a large batch and assert acceptable latency + throughput."""
    # Arrange: build a 1,000-row batch by sampling with replacement.
    rng = np.random.RandomState(0)
    big = applicant_batch.iloc[rng.randint(0, len(applicant_batch), size=1000)].reset_index(drop=True)

    # Act
    start = time.perf_counter()
    out = service.predict(big)
    elapsed = time.perf_counter() - start
    throughput = len(big) / elapsed

    # Assert
    assert len(out) == len(big), "Output row count must match input"
    assert elapsed < 5.0, f"Scoring 1,000 applicants took {elapsed:.2f}s (>5s budget)"
    assert throughput > 50.0, f"Throughput {throughput:.0f}/s below 50/s floor"
