"""
Async verification management for the YOLO detection pipeline.

Handles submitting verification futures, processing completed results,
and draining remaining futures after the main loop finishes.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, wait

from app.detectors.yolo.config import ALL_CLASSES, MAX_VERIFY_CONCURRENT

logger = logging.getLogger(__name__)


def submit_verification(
    executor: ThreadPoolExecutor,
    verify_fn,
    tid: int,
    frame,
    bbox: tuple,
    class_name: str,
    state,
):
    """
    Submit an async verification future for the given detection.

    Only submits if:
      - tid is not already cached or pending
      - we haven't hit the concurrency limit

    Args:
        state: TrackerState instance (uses state.verify_cache, state.pending_verify)
    """
    if (
        tid not in state.verify_cache
        and tid not in state.pending_verify
        and len(state.pending_verify) < MAX_VERIFY_CONCURRENT
    ):
        frame_copy = frame.copy()
        bbox_copy = bbox
        class_copy = class_name
        future = executor.submit(verify_fn, frame_copy, bbox_copy, class_copy)
        state.pending_verify[tid] = {
            "future": future,
            "class_name": class_copy,
        }


def process_completed_futures(state, perf_timings: dict):
    """
    Check pending verify futures and apply results retroactively.

    Modifies state.confirmed, state.counted_ids, state.rejected_tids,
    state.verify_cache, state.verify_stats, state.rejection_stats.

    Args:
        state: TrackerState instance
        perf_timings: dict with "verification" key for timing accumulation
    """
    done_tids = []
    for tid, pending in state.pending_verify.items():
        future = pending["future"]
        if not future.done():
            continue
        done_tids.append(tid)

        try:
            vl_result = future.result(timeout=0)
        except Exception as e:
            logger.warning(f"Verify async error for tid={tid}: {e}")
            state.rejection_stats["vl_errors"] += 1
            continue

        if not vl_result:
            state.rejection_stats["vl_errors"] += 1
            continue

        state.verify_stats["total_verified"] += 1
        state.verify_cache[tid] = vl_result

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
            state.verify_stats["verified_success"] += 1
            if tid in state.confirmed:
                state.confirmed[tid]["vl_verified"] = True
                state.confirmed[tid]["vl_confidence"] = vl_confidence
                state.confirmed[tid]["vl_category"] = vl_category

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
            state.verify_stats["verified_success"] += 1
            state.verify_stats["vl_overrides"] += 1
            if tid in state.confirmed:
                old_class = state.confirmed[tid]["type"]
                if old_class in state.counted_ids:
                    state.counted_ids[old_class].discard(tid)
                if vl_category in state.counted_ids:
                    state.counted_ids[vl_category].add(tid)
                state.confirmed[tid]["type"] = vl_category
                state.confirmed[tid]["vl_verified"] = True
                state.confirmed[tid]["vl_confidence"] = vl_confidence
                state.confirmed[tid]["vl_category"] = vl_category
                state.tracker_class_lock[tid] = vl_category
        else:
            # Tier 3: Verify rejects — remove the confirmed detection
            state.verify_stats["verified_failed"] += 1
            state.rejection_stats["vl_mismatch"] += 1
            logger.info(
                f"Verify async rejected tid={tid}: YOLO={yolo_class}, "
                f"VL={vl_category} (conf={vl_confidence}, belongs={belongs})"
            )
            if tid in state.confirmed:
                old_class = state.confirmed[tid]["type"]
                if old_class in state.counted_ids:
                    state.counted_ids[old_class].discard(tid)
                del state.confirmed[tid]
            state.rejected_tids.add(tid)
            state.verify_cache[tid] = vl_result  # Prevent re-submission

    for tid in done_tids:
        del state.pending_verify[tid]


def drain_pending(state, perf_timings: dict, timeout: int):
    """
    Drain all remaining pending verify futures after the main loop ends.

    Waits up to `timeout` seconds, then cancels any remaining futures.

    Args:
        state: TrackerState instance
        perf_timings: dict for timing accumulation
        timeout: max seconds to wait for remaining futures
    """
    if not state.pending_verify:
        return

    logger.info(f"Draining {len(state.pending_verify)} pending verify futures...")
    remaining_futures = [p["future"] for p in state.pending_verify.values()]
    done, not_done = wait(remaining_futures, timeout=timeout)

    # Process completed ones
    process_completed_futures(state, perf_timings)

    # Cancel any that didn't finish in time
    if state.pending_verify:
        logger.warning(
            f"{len(state.pending_verify)} verify futures timed out — cancelling"
        )
        for tid in list(state.pending_verify.keys()):
            state.pending_verify[tid]["future"].cancel()
            state.rejection_stats["vl_errors"] += 1
            del state.pending_verify[tid]
