# Video Processing Concurrency Model

## The Restaurant Analogy

Think of our video processing system like a restaurant:

| Role | System Component | Resource |
|---|---|---|
| **Waiter** (takes orders) | `cap.read()` — frame decode | CPU |
| **Chef** (cooks food) | `model.track()` — YOLO detection | GPU |
| **Food Inspector** (quality check) | `verify_detection_with_vl()` — VL API | Network (cloud) |

In a **bad restaurant** (old blocking code), the waiter, chef, and inspector all work one at a time:

```
Waiter gets order → Chef cooks → Inspector checks → Waiter gets next order
                                  ↑ Everyone waits 3 seconds for this
```

In our **optimized restaurant**, the inspector works independently:

```
Waiter gets order → Chef cooks → Serve immediately → Waiter gets next order
                                  ↓
                          Inspector checks in background
                          (if bad, pull the dish back later)
```

---

## How It Works — Step by Step

### The Players

```
┌─────────────────────────────┐     ┌──────────────────────────────┐
│     MAIN THREAD             │     │   VL THREAD POOL (4 workers) │
│                             │     │                              │
│  CPU: cap.read()            │     │  Worker 1: VL API call       │
│  GPU: model.track()         │     │  Worker 2: VL API call       │
│  CPU: process detections    │     │  Worker 3: VL API call       │
│  CPU: check VL results      │     │  Worker 4: idle              │
└─────────────────────────────┘     └──────────────────────────────┘
```

### Timeline Example (Real Numbers from RTX 5090)

Imagine processing frames 100–108 of a dashcam video:

```
Time    Main Thread                          VL Pool
─────   ──────────────────────────────────   ─────────────────────────
0ms     cap.read(frame 100)     [CPU 25ms]
25ms    model.track(frame 100)  [GPU  3ms]
28ms    Found pothole tid=5 → CONFIRMED ──→  Submit VL for tid=5
28ms    cap.read(frame 102*)    [CPU 25ms]      Worker 1: calling Ollama API...
53ms    model.track(frame 102)  [GPU  3ms]
56ms    Found sign tid=8 → CONFIRMED ─────→  Submit VL for tid=8
56ms    Check VL results → nothing yet         Worker 1: still waiting (2s call)
56ms    cap.read(frame 104*)    [CPU 25ms]      Worker 2: calling Ollama API...
81ms    model.track(frame 104)  [GPU  3ms]
84ms    No new detections
84ms    Check VL results → nothing yet         Worker 1: still waiting...
84ms    cap.read(frame 106*)    [CPU 25ms]      Worker 2: still waiting...
109ms   model.track(frame 106)  [GPU  3ms]
112ms   Found crack tid=12 → CONFIRMED ───→  Submit VL for tid=12
112ms   cap.read(frame 108*)    [CPU 25ms]      Worker 3: calling Ollama API...
        ...
2000ms  (some later frame)                      Worker 1: ✅ VL says tid=5 IS pothole
        Check VL results → tid=5 verified!       → mark confirmed[5].vl_verified = True
        ...
2500ms  (some later frame)                      Worker 2: ❌ VL says tid=8 is NOT a sign
        Check VL results → tid=8 rejected!       → del confirmed[8], add to rejected_tids
```

*\* Odd frames skipped because FRAME_SKIP=2*

### Key Observations

1. **Main thread never waits for VL** — it keeps processing frames at full GPU speed
2. **VL runs in parallel** — 4 API calls can happen simultaneously
3. **Results applied retroactively** — when a VL result arrives (2-4s later), it modifies the existing confirmed detection
4. **Rejected detections are permanent** — `rejected_tids` prevents VL-rejected objects from being re-confirmed

---

## The Two Thread Pools

We use **two separate** thread pools to avoid deadlocks:

```
_async_vl_executor (MAX_VL_CONCURRENT=4 workers)
├── Runs verify_detection_with_vl() as a whole
├── Submitted from the main frame loop
└── Results checked via _process_completed_vl_futures()

_vl_timeout_executor (4 workers)  
├── Runs the raw Ollama API call (vl_client.chat)
├── Submitted from INSIDE verify_detection_with_vl()
└── Used ONLY for VL_TIMEOUT enforcement (30s hard limit)
```

Why two pools? Because `verify_detection_with_vl()` internally submits to a pool for timeout enforcement. If we used the same pool for both the outer async call AND the inner timeout call, all workers would deadlock — each worker would be waiting for an inner future that can never start because all workers are occupied.

---

## What Happens at Video End

When all frames are processed, some VL calls may still be in-flight:

```
Frame loop ends
     ↓
Drain phase: wait up to VL_TIMEOUT (30s) for remaining futures
     ↓
├── Completed futures → apply results (verify/override/reject)
└── Timed-out futures → cancel and count as vl_errors
     ↓
Build final results using updated `confirmed` dict
     ↓
Save to database
```

---

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `FRAME_SKIP` | `2` | Process every Nth frame (reduces CPU decode load) |
| `YOLO_IMGSZ` | `640` | YOLO input resolution (lower = faster, less accurate) |
| `MAX_VL_CONCURRENT` | `4` | Max parallel VL API calls |
| `VL_TIMEOUT_SECONDS` | `30` | Hard timeout per VL call |
| `ENABLE_VL_VERIFICATION` | `true` | Toggle VL entirely on/off |

---

## Performance Impact

Same video (2671 frames, 89s duration) on RTX 5090:

| Configuration | Processing Time | VL Timeouts |
|---|---|---|
| FRAME_SKIP=1, Blocking VL | ~4.5 minutes | 27 |
| FRAME_SKIP=2, Async VL | **~34 seconds** | 0 |

**~8x total speedup** by combining frame skipping + async VL.
