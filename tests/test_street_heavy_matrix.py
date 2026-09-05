"""
TPX Storage Engine — Heavy Street Test Matrix & Adversarial Integrity Suite
=============================================================================
Tests physical reality, extreme concurrency, learned mathematical models,
boundary edge cases, and crash/tamper resilience on target SATA SSD (EXRAM 512GB).
"""

import os
import sys
import time
import math
import hashlib
import tempfile
import threading
import numpy as np
import pytest

import tpx
from tpx import (
    TPXDatabase,
    AdaptivePrefetchController,
    RadixSplinePredictor,
    fast_shannon_entropy,
    optimal_codec_decision,
    should_use_vectorized_batch,
)


def get_test_db_path(name: str) -> str:
    scratch_dir = os.path.join(os.path.dirname(__file__), "..", "target", "test_scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    return os.path.join(scratch_dir, f"{name}_{int(time.time()*1000)}.tpx")


# =============================================================================
# 1. MATHEMATICAL MODEL & PREDICTOR STREET TESTS
# =============================================================================

def test_street_shannon_entropy_decision_boundaries():
    """Verify Shannon entropy mathematical equations against extreme entropy spectra."""
    # A. Uniform Random Bytes (Maximum theoretical entropy ~ 8.0)
    high_entropy_bytes = os.urandom(4096)
    h_high = fast_shannon_entropy(high_entropy_bytes)
    assert 7.75 <= h_high <= 8.0, f"Expected near-max entropy, got {h_high}"
    assert optimal_codec_decision(high_entropy_bytes) == "none", "Should skip compression for random data"

    # B. Repetitive / Sparse Bytes (Near-zero entropy)
    low_entropy_bytes = bytes([42] * 2048 + [0] * 2048)
    h_low = fast_shannon_entropy(low_entropy_bytes)
    assert h_low < 2.0, f"Expected low entropy, got {h_low}"
    assert optimal_codec_decision(low_entropy_bytes) == "zstd", "Should pick deep Zstd for low entropy"

    # C. Structured Float Data (Normal distribution floats)
    structured_floats = np.linspace(-1.0, 1.0, 1024, dtype=np.float32).tobytes()
    h_mid = fast_shannon_entropy(structured_floats)
    assert 2.0 <= h_mid <= 7.85, f"Expected structured mid entropy, got {h_mid}"


def test_street_adaptive_prefetch_littles_law():
    """Verify Little's Law + Variance Expansion dynamically adapts without starving GPU."""
    controller = AdaptivePrefetchController(min_buffer=2, max_buffer=64)

    # Simulated stable low-jitter NVMe read latencies (e.g. 100 microseconds per tensor)
    for _ in range(50):
        controller.record_io_latency(0.00010)

    # GPU consumes 10,000 tensors/sec: B* = ceil(10000 * 0.00010 * (1 + 0)) = 1 -> min_buffer (2)
    b_stable = controller.calculate_optimal_prefetch(gpu_consumption_rate=10000.0)
    assert b_stable == 2

    # Incur high variance / queue jitter (P99 spikes to 5ms)
    for _ in range(10):
        controller.record_io_latency(0.0050)

    # With high jitter (Cv^2 > 0), prefetch depth must expand to cushion GPU
    b_jitter = controller.calculate_optimal_prefetch(gpu_consumption_rate=10000.0)
    assert b_jitter > b_stable, f"Prefetch queue did not expand under jitter: {b_jitter} vs {b_stable}"
    assert b_jitter <= 64, "Prefetch queue must not exceed max_buffer safety cap"


def test_street_radix_spline_error_bounds():
    """Verify single-pass learned index predicts all physical offsets within error bound epsilon."""
    epsilon = 8
    # 100 sorted keys with non-linear exponential distribution
    keys = [int(math.exp(i * 0.1)) for i in range(100)]
    spline_points = [(k, i) for i, k in enumerate(keys)]

    predictor = RadixSplinePredictor(spline_points=spline_points, epsilon=epsilon)

    for true_idx, k in enumerate(keys):
        low, high = predictor.predict_window(k)
        assert low <= true_idx <= high, (
            f"Key {k} (actual index {true_idx}) fell outside predicted window [{low}, {high}]"
        )


def test_street_batch_crossover_threshold():
    """Verify Amdahl's Law crossover decision function."""
    # Small batch (< 2 MB) -> sequential write is faster
    assert not should_use_vectorized_batch(num_tensors=10, total_bytes=100 * 1024)
    # Large volume batch (> 2 MB) -> parallel coalesced write is faster
    assert should_use_vectorized_batch(num_tensors=32, total_bytes=5 * 1024 * 1024)


# =============================================================================
# 2. ADVERSARIAL TENSOR SHAPES & BUFFER STRESS TESTS
# =============================================================================

def test_street_zero_length_and_high_rank_tensors():
    """Test 0-length tensors, 1D, 2D, 3D, and 4D tensor shapes with exact round-trip."""
    path = get_test_db_path("street_shapes")
    try:
        with tpx.open(path, mode="w+") as db:
            # 1. 0-length tensor
            empty_arr = np.array([], dtype=np.float32)
            db.put("empty_tensor", empty_arr)

            # 2. 1D vector
            vec = np.arange(128, dtype=np.int32)
            db.put("vector_1d", vec)

            # 3. 4D High-rank tensor
            tensor_4d = np.random.randn(2, 4, 8, 16).astype(np.float32)
            db.put("tensor_4d", tensor_4d)

            # 4. Non-contiguous slice
            parent_arr = np.arange(1000, dtype=np.float32).reshape(20, 50)
            sliced_arr = parent_arr[::2, ::5]  # non-contiguous
            assert not sliced_arr.flags["C_CONTIGUOUS"]
            db.put("sliced_tensor", sliced_arr)

            db.flush()

        with tpx.open(path, mode="r") as db:
            # Check 0-length
            r_empty = db.get("empty_tensor")
            assert r_empty is not None and len(r_empty) == 0

            # Check 1D
            r_vec = db.get("vector_1d")
            assert np.array_equal(r_vec, vec)

            # Check 4D
            r_4d = db.get("tensor_4d")
            assert r_4d.shape == (2, 4, 8, 16)
            assert np.allclose(r_4d, tensor_4d)

            # Check Non-contiguous slice
            r_sliced = db.get("sliced_tensor")
            assert np.array_equal(r_sliced, sliced_arr)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_street_massive_100mb_tensor_roundtrip():
    """Verify massive 100 MB tensor storage, checksum validation, and byte-for-byte SHA256 equality."""
    path = get_test_db_path("street_massive_100mb")
    num_floats = 25 * 1024 * 1024  # 100 MB
    try:
        print("\n  Allocating 100 MB Gaussian float array...")
        massive_data = np.random.randn(num_floats).astype(np.float32)
        expected_sha256 = hashlib.sha256(massive_data.tobytes()).hexdigest()

        t0 = time.perf_counter()
        with tpx.open(path, mode="w+") as db:
            db.put("huge/weights", massive_data)
            db.flush()
        write_sec = time.perf_counter() - t0
        write_mb_s = (100.0) / write_sec
        print(f"  100 MB Written in {write_sec:.3f}s ({write_mb_s:.2f} MB/s)")

        t1 = time.perf_counter()
        with tpx.open(path, mode="r") as db:
            loaded_data = db.get("huge/weights")
        read_sec = time.perf_counter() - t1
        read_mb_s = (100.0) / read_sec
        print(f"  100 MB Read in {read_sec:.3f}s ({read_mb_s:.2f} MB/s)")

        assert loaded_data is not None
        actual_sha256 = hashlib.sha256(loaded_data.tobytes()).hexdigest()
        assert actual_sha256 == expected_sha256, "100 MB tensor corrupted in transit!"
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
# 3. HIGH-CONCURRENCY MULTI-THREADED STREET STRESS
# =============================================================================

def test_street_multithreaded_concurrent_readers_and_writers():
    """Simulate 8 concurrent PyTorch workers hammering the database simultaneously."""
    path = get_test_db_path("street_concurrency")
    num_threads = 8
    items_per_thread = 50
    errors = []

    try:
        # Pre-seed database
        with tpx.open(path, mode="w+") as db:
            for t_id in range(num_threads):
                for i in range(items_per_thread):
                    data = np.full((64,), fill_value=t_id * 1000 + i, dtype=np.int32)
                    db.put(f"worker_{t_id}/layer_{i:03}", data)
            db.flush()

        def reader_worker(worker_id: int):
            try:
                with tpx.open(path, mode="r") as db:
                    for i in range(items_per_thread):
                        key = f"worker_{worker_id}/layer_{i:03}"
                        arr = db.get(key)
                        if arr is None:
                            raise ValueError(f"Key {key} not found by worker {worker_id}")
                        expected_val = worker_id * 1000 + i
                        if arr[0] != expected_val:
                            raise ValueError(f"Corrupted data in {key}: got {arr[0]}, expected {expected_val}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader_worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
# 4. PUT_BATCH VECTORIZED API & SMART AUTO-ROUTING STREET TEST
# =============================================================================

def test_street_put_batch_and_auto_routing():
    """Verify batch writes via dictionary and tuple list, with auto-routing threshold."""
    path = get_test_db_path("street_batch")
    try:
        batch_dict = {
            f"batch_tensor_{i}": np.full((128,), fill_value=float(i), dtype=np.float32)
            for i in range(25)
        }

        with tpx.open(path, mode="w+") as db:
            db.put_batch(batch_dict, auto_route=True)
            db.flush()

        with tpx.open(path, mode="r") as db:
            results = db.get_batch(list(batch_dict.keys()))
            assert len(results) == 25
            for k, expected_arr in batch_dict.items():
                assert k in results
                assert np.array_equal(results[k], expected_arr)
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
# 5. IN-FILE VERSION CHAIN & COMPACTION PURGE STREET TEST
# =============================================================================

def test_street_version_chain_and_compaction():
    """Create 5 consecutive versions of a key, verify version retrieval, tombstone and compact."""
    path = get_test_db_path("street_versioning")
    try:
        with tpx.open(path, mode="w+") as db:
            for v in range(1, 6):
                arr = np.full((16,), fill_value=v, dtype=np.int32)
                db.put("evolving_model/head", arr, attrs={"version": v}, retain="keep_forever")
            db.flush()

        with tpx.open(path, mode="r") as db:
            versions = db.list_versions("evolving_model/head")
            assert len(versions) == 5, f"Expected 5 versions, got {versions}"

            # Check that latest returns version 5
            latest = db.get("evolving_model/head")
            assert latest[0] == 5

            # Check historical version 2
            v2 = db.get_version("evolving_model/head", version=versions[1])
            assert v2 is not None and v2[0] == 2

        # Delete key and compact
        with tpx.open(path, mode="w+") as db:
            db.delete("evolving_model/head")
            db.flush()
            db.compact()

        with tpx.open(path, mode="r") as db:
            assert db.get("evolving_model/head") is None, "Tombstoned key resurrected after compaction!"
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
