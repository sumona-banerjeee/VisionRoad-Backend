"""
TrackerState — encapsulates all tracking data structures and mutation logic.

Keeps confirmed detections, spatial deduplication, class locks, rejection/verify
stats, and stale-entry eviction in one cohesive object.
"""

import logging
from collections import defaultdict, deque

from app.detectors.yolo.config import ALL_CLASSES

logger = logging.getLogger(__name__)


class TrackerState:
    """Mutable bag of state that lives for one video processing run."""

    def __init__(self, time_threshold: float, has_verify: bool):
        self.time_threshold = time_threshold
        self.has_verify = has_verify

        # Per-track history: tid -> deque of timestamps
        self.tracker_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        # Confirmed detections: tid -> info dict
        self.confirmed: dict[int, dict] = {}
        # Per-class sets of counted track IDs
        self.counted_ids: dict[str, set] = {cls: set() for cls in ALL_CLASSES}
        # Per-class time-windowed deque for spatial dedup
        self.spatial_locations: dict[str, deque] = defaultdict(deque)
        # tid -> locked class name
        self.tracker_class_lock: dict[int, str] = {}
        # Pending async-verify futures: tid -> {future, class_name}
        self.pending_verify: dict[int, dict] = {}
        # Cached verify results by tid
        self.verify_cache: dict[int, dict] = {}
        # Verify-rejected tids (prevents re-confirmation)
        self.rejected_tids: set[int] = set()
        # tid -> last current_time the tid was observed
        self._tid_last_seen: dict[int, float] = {}

        # ── Stats ────────────────────────────────────────────────────────
        self.rejection_stats = {
            "multi_frame_pending": set(),
            "spatial_duplicate": 0,
            "roi_outside": 0,
            "class_mismatch": 0,
            "vl_mismatch": 0,
            "vl_errors": 0,
        }
        self.verify_stats = {
            "total_verified": 0,
            "verified_success": 0,
            "verified_failed": 0,
            "skipped": 0,
            "vl_overrides": 0,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def is_duplicate_location(
        self,
        cx: int,
        cy: int,
        class_name: str,
        current_time: float,
        min_distance_threshold: int,
    ) -> tuple[bool, str | None]:
        """
        Check if this location/class was already counted recently.

        Entries are appended in time order so expired ones are pruned
        from the front in O(1) before scanning.
        """
        bucket = self.spatial_locations[class_name]

        # Prune entries outside the time window (deque is time-ordered)
        while bucket and (current_time - bucket[0]["time"]) > self.time_threshold:
            bucket.popleft()

        for existing in bucket:
            distance = self.calculate_distance((cx, cy), existing["center"])
            if distance < min_distance_threshold:
                time_gap = current_time - existing["time"]
                return True, f"{distance:.1f}px from existing, {time_gap:.2f}s ago"

        return False, None

    def confirm_detection(
        self,
        tid: int,
        class_name: str,
        frame_count: int,
        current_time: float,
        conf: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        cx: int,
        cy: int,
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
            {"center": (cx, cy), "time": current_time}
        )
        self.rejection_stats["multi_frame_pending"].discard(tid)

    def evict_stale_trackers(self, current_time: float):
        """Remove stale entries whose tid hasn't been seen for > time_threshold."""
        stale_tids = [
            tid
            for tid, last_t in self._tid_last_seen.items()
            if current_time - last_t > self.time_threshold
            and tid not in self.confirmed
            and tid not in self.pending_verify
        ]
        for tid in stale_tids:
            self.verify_cache.pop(tid, None)
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
