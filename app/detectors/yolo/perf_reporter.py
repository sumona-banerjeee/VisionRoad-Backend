"""
Performance timing accumulation and structured report generation.
"""

import logging

from app.core.logging_config import perf_logger

logger = logging.getLogger(__name__)


def create_perf_timings() -> dict:
    """Create a fresh perf_timings accumulator dict."""
    return {
        "frame_decode": {"total": 0.0, "count": 0},
        "yolo_inference": {"total": 0.0, "count": 0},
        "gps_coord": {"total": 0.0, "count": 0},
        "verification": {"total": 0.0, "count": 0},
        "db_gps_match": {"total": 0.0, "count": 0},
        "db_bulk_write": {"total": 0.0, "count": 0},
    }


def record(perf_timings: dict, stage: str, elapsed: float):
    """Record a single timing event for a given stage."""
    perf_timings[stage]["total"] += elapsed
    perf_timings[stage]["count"] += 1


def generate_report(
    perf_timings: dict,
    video_id: str,
    detection_mode: str,
    total_time: float,
    yolo_time: float,
    drain_time: float,
    video_duration: float,
    total_frames: int,
    fps: float,
    frames_processed: int,
    frame_skip: int,
    detections_saved: int,
    total_verified: int,
):
    """Format and log the structured performance report."""

    def _fmt(stage_key):
        d = perf_timings[stage_key]
        total = d["total"]
        count = d["count"]
        avg_ms = (total / count * 1000) if count > 0 else 0.0
        return total, count, avg_ms

    fd_t, fd_c, fd_avg = _fmt("frame_decode")
    yi_t, yi_c, yi_avg = _fmt("yolo_inference")
    gc_t, gc_c, gc_avg = _fmt("gps_coord")
    vl_t, vl_c, vl_avg = _fmt("verification")
    dg_t, dg_c, dg_avg = _fmt("db_gps_match")
    db_t, db_c, db_avg = _fmt("db_bulk_write")

    def _flag(t):
        return " ⚠️" if total_time > 0 and (t / total_time) > 0.20 else ""

    report_lines = [
        f"{'=' * 78}",
        f"  PERF REPORT — [{video_id}]  mode={detection_mode}",
        f"{'=' * 78}",
        f"  {'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg/call (ms)':>15}",
        f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
        f"  {'Frame decode (I/O)':<30} {fd_t:>10.3f} {fd_c:>7d} {fd_avg:>15.2f}{_flag(fd_t)}",
        f"  {'YOLO inference':<30} {yi_t:>10.3f} {yi_c:>7d} {yi_avg:>15.2f}{_flag(yi_t)}",
        f"  {'GPS coord lookup':<30} {gc_t:>10.3f} {gc_c:>7d} {gc_avg:>15.2f}{_flag(gc_t)}",
        f"  {'Verification (async)':<30} {vl_t:>10.3f} {vl_c:>7d} {vl_avg:>15.2f}{_flag(vl_t)}",
        f"  {'Verify drain (post-loop)':<30} {drain_time:>10.3f} {'N/A':>7} {'N/A':>15}",
        f"  {'DB GPS matching':<30} {dg_t:>10.3f} {dg_c:>7d} {dg_avg:>15.2f}{_flag(dg_t)}",
        f"  {'DB bulk write':<30} {db_t:>10.3f} {db_c:>7d} {db_avg:>15.2f}{_flag(db_t)}",
        f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
        f"  {'TOTAL pipeline time':<30} {total_time:>10.3f}s",
        f"  {'Video duration':<30} {video_duration:>10.1f}s  ({total_frames} frames @ {fps:.0f} FPS)",
        f"  {'Frames processed':<30} {frames_processed:>10d}  (FRAME_SKIP={frame_skip})",
        f"  {'Detections saved':<30} {detections_saved:>10d}",
        f"  {'Verifications done':<30} {total_verified:>10d}",
        f"{'=' * 78}",
    ]
    report_str = "\n".join(report_lines)
    perf_logger.info(f"\n{report_str}")
