# Location → Chainage + Lane — Full Backend Refactor

Replace the generic [Location](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/location.py#9-66) model with NHAI-style linear infrastructure. The new hierarchy:

```
Project → Package → Chainage → Lane → Video → Detection
```

> [!IMPORTANT]
> **Breaking change.** The [locations](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/location_routes.py#98-125) table, all `/locations` endpoints, and all `location_id` FKs will be removed. The old `visionroad.db` must be deleted and re-seeded.

> [!NOTE]
> **Chainage KM values are absolute NHAI milestones** — e.g. if a package covers KM 100–200, its chainages store `chainage_start_km=100`, `chainage_end_km=120`, etc. never relative offsets.

---

## Proposed Changes

### Models
---

#### [MODIFY] [location.py → chainage.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/location.py)
- Rename file, class ([Location](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/location.py#9-66) → `Chainage`), table ([locations](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/location_routes.py#98-125) → `chainages`)
- Make `chainage_start_km` and `chainage_end_km` **required** (non-nullable)
- Rename relationship `package.locations` → `package.chainages`
- Add `lanes` one-to-many relationship
- Rename [contains_point()](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/location.py#55-66) → [contains_point()](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/location.py#55-66) (keep GPS logic unchanged)

#### [NEW] [lane.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/lane.py)
New model with:
- [id](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/video.py#10-59) (UUID PK), `chainage_id` (FK → chainages, CASCADE)
- `lane_code` (required string: `"LHS"`, `"RHS"`, `"UP"`, `"DOWN"`)
- `lane_type` (optional: `"driving"`, `"shoulder"`), `direction` (optional)
- `TimestampMixin`, belongs to `Chainage`, has many `Videos`

#### [MODIFY] [package.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/package.py)
- Add `chainage_start_km: Float (nullable)` and `chainage_end_km: Float (nullable)`
- Rename relationship [locations](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/location_routes.py#98-125) → `chainages`

#### [MODIFY] [video.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/video.py)
- Replace `location_id` FK (→ [locations](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/location_routes.py#98-125)) with `chainage_id` (→ `chainages`, SET NULL) + `lane_id` (→ `lanes`, SET NULL)
- Update relationship names accordingly

#### [MODIFY] [detection.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/detection.py)
- Replace `location_id` column → `chainage_id` + `lane_id` (both nullable, indexed)
- Update docstring comment

#### [MODIFY] [models/\_\_init\_\_.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/__init__.py)
- Replace [Location](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/location.py#9-66) import → `Chainage` + `Lane`

---

### CRUD Layer
---

#### [MODIFY] [crud_hierarchy.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/db/crud_hierarchy.py)
- Add `chainage_start_km` + `chainage_end_km` params to [create_package()](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/package_routes.py#39-58) (optional)
- Replace all [Location](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/location.py#9-66) CRUD functions with `Chainage` equivalents
- Add full `Lane` CRUD (`create_lane`, `get_lane`, `list_lanes`, `update_lane`, `delete_lane`)
- Rename [find_location_by_gps()](file:///e:/SentientGeeks/VisionRoad-Backend/app/db/crud_hierarchy.py#205-236) → `find_chainage_by_gps()`

#### [MODIFY] [crud.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/db/crud.py)
- [create_video()](file:///e:/SentientGeeks/VisionRoad-Backend/app/db/crud.py#15-40): replace `location_id` param → `chainage_id` + `lane_id`
- [create_detection()](file:///e:/SentientGeeks/VisionRoad-Backend/app/db/crud.py#83-118): replace `location_id` param → `chainage_id` + `lane_id`

---

### API Routes
---

#### [NEW] [chainage_routes.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/chainage_routes.py)
Full CRUD at `/chainages`:
```
POST   /chainages
GET    /chainages?package_id=
GET    /chainages/{id}
PUT    /chainages/{id}
DELETE /chainages/{id}
```
Schemas: `ChainageCreate` (KM fields required), `ChainageUpdate`, `ChainageResponse`
- Validate GPS bounds on create/update (reuse [validate_gps_bounds](file:///e:/SentientGeeks/VisionRoad-Backend/app/services/location_mapper.py#46-83) from `chainage_mapper.py`)

#### [NEW] [lane_routes.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/lane_routes.py)
Full CRUD at `/lanes`:
```
POST   /lanes
GET    /lanes?chainage_id=
GET    /lanes/{id}
PUT    /lanes/{id}
DELETE /lanes/{id}
```
Schemas: `LaneCreate`, `LaneUpdate`, `LaneResponse`

#### [DELETE] [location_routes.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/location_routes.py)
Remove entirely.

#### [MODIFY] [package_routes.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/package_routes.py)
- Add `chainage_start_km` + `chainage_end_km` (Optional[float]) to [PackageCreate](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/package_routes.py#15-19), [PackageUpdate](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/package_routes.py#21-24), [PackageResponse](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/package_routes.py#26-36)
- Pass to `crud_hierarchy.create_package()` and return in responses

#### [MODIFY] [upload_process_routes.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/upload_process_routes.py)
- Add optional `chainage_id: Optional[str] = Form(None)` + `lane_id: Optional[str] = Form(None)` to `/upload`
- Pass to `upload_service.upload_video()`

#### [MODIFY] [summary_routes.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/summary_routes.py)
- Replace all [Location](file:///e:/SentientGeeks/VisionRoad-Backend/app/models/location.py#9-66) imports/references → `Chainage` + `Lane`
- Project summary: restructure `packages → chainages → lanes → detections`
- Package summary: restructure `chainages → lanes → detections`
- Rename `/summary/locations/{id}` → `/summary/chainages/{id}`
- Update statistics endpoint (replace `location_count` → `chainage_count`)

---

### Services
---

#### [MODIFY] [location_mapper.py → chainage_mapper.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/services/location_mapper.py)
- Rename file to `chainage_mapper.py`
- Rename [get_location_hierarchy()](file:///e:/SentientGeeks/VisionRoad-Backend/app/services/location_mapper.py#8-44) → `get_chainage_hierarchy()`; include lane info in returned dict
- Keep [validate_gps_bounds()](file:///e:/SentientGeeks/VisionRoad-Backend/app/services/location_mapper.py#46-83) exactly as-is

#### [MODIFY] [upload_service.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/services/upload_service.py)
- Add `chainage_id` + `lane_id` params to [upload_video()](file:///e:/SentientGeeks/VisionRoad-Backend/app/services/upload_service.py#71-168)
- Pass to `crud.create_video()`

---

### App Initialization
---

#### [MODIFY] [app/\_\_init\_\_.py](file:///e:/SentientGeeks/VisionRoad-Backend/app/__init__.py)
- Replace `location_router` import → `chainage_router` + `lane_router`
- Register both at `/api/v1`
- Update root `"endpoints"` dict: remove `"locations"`, add `"chainages"` + `"lanes"`

---

### Migration & Supporting
---

#### [MODIFY] [alembic/env.py](file:///e:/SentientGeeks/VisionRoad-Backend/alembic/env.py)
- Replace [location](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/location_routes.py#127-147) import → `chainage`, add `lane` import

#### DB Reset
Since this is SQLite and the change is destructive: delete `visionroad.db` and let `init_db()` recreate all tables from scratch.

#### [MODIFY] [seed_data.py](file:///e:/SentientGeeks/VisionRoad-Backend/seed_data.py)
- Replace [create_location()](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/location_routes.py#55-96) helper → `create_chainage()` + `create_lane()`
- Update all seeds to use **absolute NHAI KM values** for chainages
- Add default lanes (`"LHS"`, `"RHS"`) for each chainage
- Update all API URLs from `/locations` → `/chainages`/`/lanes`
- Update final help text

---

## Verification Plan

### Automated
1. Delete DB: `del visionroad.db` (from `e:\SentientGeeks\VisionRoad-Backend`)
2. Start server: `python -m uvicorn main:app --reload`
   - Verify all tables created without errors
3. Run seed: `python seed_data.py`
   - Verify chainages + lanes created successfully
4. Swagger UI at `http://localhost:8000/docs`:
   - Test all `/chainages` and `/lanes` CRUD endpoints
   - Test `/upload` with `chainage_id` + `lane_id`
   - Test `/summary/*` — confirm `chainage → lane` grouping

### Manual
- Confirm no `/locations` endpoints appear in Swagger
- Confirm [PackageResponse](file:///e:/SentientGeeks/VisionRoad-Backend/app/routes/package_routes.py#26-36) includes `chainage_start_km` + `chainage_end_km`
