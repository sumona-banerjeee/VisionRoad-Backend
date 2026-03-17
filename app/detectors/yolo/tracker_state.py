"""
TrackerState — encapsulates all mutable tracking state for a single video run.

Centralises tracker_history, confirmed detections, counted_ids,
spatial_locations, class locks, rejection/verify stats, and stale eviction.
"""

import logging
from collections import defaultdict, deque

from app.detectors.yolo.config import ALL_CLASSES

logger = logging.getLogger(__name__)


class TrackerState:
    """
    Holds every mutable dict/set used during the frame-processing loop.

    Create one instance per video processing run.
    """

    def __init__(self, has_verify: bool = False):
        self.has_verify = has_verify

        # Tracker-id → deque of timestamps (capped at 50)
        self.tracker_history: dict = defaultdict(lambda: deque(maxlen=50))
        # Tracker-id → confirmed detection info dict
        self.confirmed: dict = {}
        # Class name → set of confirmed tracker ids
        self.counted_ids: dict = {cls: set() for cls in ALL_CLASSES}
        # Class name → deque of {center, time, bbox} for spatial dedup
        self.spatial_locations: dict = defaultdict(deque)
        # Tracker-id → locked class name (prevents class flicker)
        self.tracker_class_lock: dict = {}
        # Set of tracker ids rejected by verification
        self.rejected_tids: set = set()
        # Tracker-id → last timestamp it was observed
        self._tid_last_seen: dict = {}

        # Pending async verification futures: tid → {future, class_name}
        self.pending_verify: dict = {}
        # Cache of verify results by tid (prevents re-submission)
        self.verify_cache: dict = {}

        self.rejection_stats: dict = {
            "multi_frame_pending": set(),
            "spatial_duplicate": 0,
            "roi_outside": 0,
            "class_mismatch": 0,
            "vl_mismatch": 0,
            "vl_errors": 0,
        }

        self.verify_stats: dict = {
            "total_verified": 0,
            "verified_success": 0,
            "verified_failed": 0,
            "skipped": 0,
            "vl_overrides": 0,
        }

    # ── Mutations ─────────────────────────────────────────────────────────────

    def confirm_detection(
        self,
        tid: int,
        class_name: str,
        frame_count: int,
        current_time: float,
        conf: float,
        x1: int, y1: int, x2: int, y2: int,
        cx: int, cy: int,
        vl_verified: bool = False,
        vl_confidence=None,
        vl_category=None,
    ):
        """Confirm a detection and update all tracking structures."""
        self.confirmed[tid] = {
            "detection_id": tid,
            "type": class_name,
            "first_detected_frame": frame_count,
            "first_detected_time": round(current_time, 2),
            "confidence": round(float(conf), 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "vl_verified": vl_verified if self.has_verify else None,
            "vl_confidence": vl_confidence,
            "vl_category": vl_category,
        }
        if class_name in self.counted_ids:
            self.counted_ids[class_name].add(tid)
        self.spatial_locations[class_name].append(
            {
                "center": (cx, cy),
                "time": current_time,
                "bbox": (x1, y1, x2, y2),
            }
        )
        self.rejection_stats["multi_frame_pending"].discard(tid)

    def evict_stale(self, current_time: float, time_threshold: float):
        """
        Remove stale entries from verify_cache, rejected_tids,
        tracker_class_lock, and tracker_history whose tid hasn't been
        seen for more than time_threshold seconds.
        """
        stale_tids = [
            tid
            for tid, last_t in self._tid_last_seen.items()
            if current_time - last_t > time_threshold
            and tid not in self.confirmed
            and tid not in self.pending_verify
        ]
        for tid in stale_tids:
            _has_cached_result = tid in self.verify_cache
            self.verify_cache.pop(tid, None)
            if _has_cached_result:
                self.rejected_tids.discard(tid)
            self.tracker_class_lock.pop(tid, None)
            self.tracker_history.pop(tid, None)
            del self._tid_last_seen[tid]
        if stale_tids:
            logger.info(
                f"Evicted {len(stale_tids)} stale tracker IDs | "
                f"verify_cache={len(self.verify_cache)} rejected={len(self.rejected_tids)} "
                f"tracker_lock={len(self.tracker_class_lock)} last_seen={len(self._tid_last_seen)}"
            )

    # ── Read helpers ──────────────────────────────────────────────────────────

    def get_counts_snapshot(self) -> dict:
        """Return a dict of current unique counts per class (for progress)."""
        return {cls: len(ids) for cls, ids in self.counted_ids.items()}
