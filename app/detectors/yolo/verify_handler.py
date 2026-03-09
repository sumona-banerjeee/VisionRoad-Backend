"""
Async verification handler — submit, poll, drain verify futures.

Manages the ThreadPoolExecutor for VL/SAM3 verification callbacks and
processes completed futures to confirm, override, or reject detections.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, wait

from app.detectors.yolo.config import MAX_VERIFY_CONCURRENT, ALL_CLASSES

logger = logging.getLogger(__name__)

_async_verify_executor = ThreadPoolExecutor(
    max_workers=MAX_VERIFY_CONCURRENT, thread_name_prefix="verify_async"
)


def get_verify_executor() -> ThreadPoolExecutor:
    """Return the async-verify executor (for lifespan shutdown)."""
    return _async_verify_executor


def submit_verification(
    tid: int,
    frame,
    bbox: tuple,
    class_name: str,
    verify_fn,
    tracker_state,
):
    """Submit a single async verification future if eligible."""
    if (
        tid in tracker_state.verify_cache
        or tid in tracker_state.pending_verify
        or len(tracker_state.pending_verify) >= MAX_VERIFY_CONCURRENT
    ):
        return

    frame_copy = frame.copy()
    future = _async_verify_executor.submit(verify_fn, frame_copy, bbox, class_name)
    tracker_state.pending_verify[tid] = {
        "future": future,
        "class_name": class_name,
    }


def process_completed_futures(tracker_state, perf_timings: dict):
    """Check pending verify futures and apply results retroactively."""
    done_tids = []
    for tid, pending in tracker_state.pending_verify.items():
        future = pending["future"]
        if not future.done():
            continue
        done_tids.append(tid)

        try:
            vl_result = future.result(timeout=0)
        except Exception as e:
            logger.warning(f"Verify async error for tid={tid}: {e}")
            tracker_state.rejection_stats["vl_errors"] += 1
            continue

        if not vl_result:
            tracker_state.rejection_stats["vl_errors"] += 1
            continue

        tracker_state.verify_stats["total_verified"] += 1
        tracker_state.verify_cache[tid] = vl_result

        # Accumulate verify elapsed time into perf_timings
        _vl_elapsed = vl_result.pop("_vl_elapsed_s", 0.0)
        perf_timings["verification"]["total"] += _vl_elapsed
        perf_timings["verification"]["count"] += 1

        vl_category = vl_result.get("category")
        vl_confidence = vl_result.get("confidence")
        belongs = vl_result.get("belongs_to_category", False)
        yolo_class = pending["class_name"]

        if vl_category == yolo_class and belongs:
            # Tier 1: Verify agrees — mark as verified
            tracker_state.verify_stats["verified_success"] += 1
            if tid in tracker_state.confirmed:
                tracker_state.confirmed[tid]["vl_verified"] = True
                tracker_state.confirmed[tid]["vl_confidence"] = vl_confidence
                tracker_state.confirmed[tid]["vl_category"] = vl_category
        elif (
            vl_category
            and vl_category != "null"
            and vl_category in ALL_CLASSES
            and belongs
            and vl_confidence in ("high", "medium")
        ):
            # Tier 2: Verify disagrees but has valid class — override
            logger.info(
                f"Verify async override tid={tid}: YOLO={yolo_class} → VL={vl_category} "
                f"(conf={vl_confidence})"
            )
            tracker_state.verify_stats["verified_success"] += 1
            tracker_state.verify_stats["vl_overrides"] += 1
            if tid in tracker_state.confirmed:
                old_class = tracker_state.confirmed[tid]["type"]
                if old_class in tracker_state.counted_ids:
                    tracker_state.counted_ids[old_class].discard(tid)
                if vl_category in tracker_state.counted_ids:
                    tracker_state.counted_ids[vl_category].add(tid)
                tracker_state.confirmed[tid]["type"] = vl_category
                tracker_state.confirmed[tid]["vl_verified"] = True
                tracker_state.confirmed[tid]["vl_confidence"] = vl_confidence
                tracker_state.confirmed[tid]["vl_category"] = vl_category
                tracker_state.tracker_class_lock[tid] = vl_category
        else:
            # Tier 3: Verify rejects — remove the confirmed detection
            tracker_state.verify_stats["verified_failed"] += 1
            tracker_state.rejection_stats["vl_mismatch"] += 1
            logger.info(
                f"Verify async rejected tid={tid}: YOLO={yolo_class}, "
                f"VL={vl_category} (conf={vl_confidence}, belongs={belongs})"
            )
            if tid in tracker_state.confirmed:
                old_class = tracker_state.confirmed[tid]["type"]
                if old_class in tracker_state.counted_ids:
                    tracker_state.counted_ids[old_class].discard(tid)
                del tracker_state.confirmed[tid]
            tracker_state.rejected_tids.add(tid)
            tracker_state.verify_cache[tid] = vl_result

    for tid in done_tids:
        del tracker_state.pending_verify[tid]


def drain_pending(tracker_state, perf_timings: dict):
    """Drain remaining verify futures after video processing completes."""
    if not tracker_state.pending_verify:
        return

    logger.info(
        f"Draining {len(tracker_state.pending_verify)} pending verify futures..."
    )
    remaining_futures = [p["future"] for p in tracker_state.pending_verify.values()]
    VL_TIMEOUT = int(os.getenv("VL_TIMEOUT_SECONDS", "30"))
    done, not_done = wait(remaining_futures, timeout=VL_TIMEOUT)

    # Process completed ones
    process_completed_futures(tracker_state, perf_timings)

    # Cancel any that didn't finish in time
    if tracker_state.pending_verify:
        logger.warning(
            f"{len(tracker_state.pending_verify)} verify futures timed out — cancelling"
        )
        for tid in list(tracker_state.pending_verify.keys()):
            tracker_state.pending_verify[tid]["future"].cancel()
            tracker_state.rejection_stats["vl_errors"] += 1
            del tracker_state.pending_verify[tid]
