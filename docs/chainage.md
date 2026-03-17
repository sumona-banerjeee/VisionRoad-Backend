# Location → Chainage + Lane Refactor

Evolve the data model from point-based [Location](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/location.py#9-66) to NHAI-style linear infrastructure: **Chainage** (road stretch by KM) + **Lane** (directional side). The new hierarchy becomes:

```
Project → Package → Chainage → Lane → Video → Detection
```

> [!IMPORTANT]
> This is a **breaking change** to the database schema and all APIs. The existing [locations](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/db/crud_hierarchy.py#170-178) table, `/locations` endpoints, and all `location_id` foreign keys will be replaced. Existing data in [visionroad.db](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/visionroad.db) will need to be re-seeded.

---

## Proposed Changes

### Models (Database Layer)

#### [MODIFY] [location.py → chainage.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/location.py)
Rename [Location](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/location.py#9-66) → `Chainage`. Key changes:
- Table name: [locations](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/db/crud_hierarchy.py#170-178) → `chainages`
- Make `chainage_start_km` and `chainage_end_km` **required** (not nullable)
- Keep `segment_name`, GPS bounds, `package_id` FK
- Relationship: `package.locations` → `package.chainages`
- Add `lanes` relationship (one-to-many)
- Rename [contains_point()](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/location.py#55-66) method, keep GPS logic

#### [NEW] [lane.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/lane.py)
New `Lane` model:
- [id](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/video.py#10-59) (UUID PK)
- `chainage_id` (FK → chainages.id, CASCADE)
- `lane_code` (string, e.g. "UP", "DOWN", "LHS", "RHS")
- `lane_type` (optional string, e.g. "driving", "shoulder")
- `direction` (optional string, normalized)
- Timestamps via [TimestampMixin](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/db/base.py#14-23)
- Relationships: belongs to `Chainage`, has many `Videos`

#### [MODIFY] [video.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/video.py)
- Replace `location_id` FK → `chainage_id` + `lane_id` FKs (both nullable, SET NULL on delete)
- Update relationships accordingly

#### [MODIFY] [detection.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/detection.py)
- Replace denormalized `location_id` → `chainage_id` + `lane_id` (both nullable, indexed)

#### [MODIFY] [package.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/package.py)
- Rename relationship: [locations](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/db/crud_hierarchy.py#170-178) → `chainages`

#### [MODIFY] [\_\_init\_\_.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/__init__.py)
- Replace [Location](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/location.py#9-66) import with `Chainage` + `Lane`

---

### CRUD Layer

#### [MODIFY] [crud_hierarchy.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/db/crud_hierarchy.py)
- Replace all Location CRUD functions with Chainage equivalents (`create_chainage`, `get_chainage`, `list_chainages`, `update_chainage`, `delete_chainage`)
- Add Lane CRUD functions (`create_lane`, `get_lane`, `list_lanes`, `update_lane`, `delete_lane`)
- Replace [find_location_by_gps()](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/db/crud_hierarchy.py#205-236) → `find_chainage_by_gps()` (same bounding-box logic)

#### [MODIFY] [crud.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/db/crud.py)
- [create_video()](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/db/crud.py#15-40): replace `location_id` param → `chainage_id` + `lane_id`
- [create_detection()](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/db/crud.py#83-118): replace `location_id` param → `chainage_id` + `lane_id`

---

### API Routes

#### [NEW] [chainage_routes.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/routes/chainage_routes.py)
Full CRUD at `/chainages`:
```
POST   /chainages              — create chainage in a package
GET    /chainages?package_id=  — list (filterable)
GET    /chainages/{id}         — get by ID
PUT    /chainages/{id}         — update
DELETE /chainages/{id}         — delete (cascades to lanes, videos)
```
Schemas: `ChainageCreate`, `ChainageUpdate`, `ChainageResponse`
- `chainage_start_km` and `chainage_end_km` are **required** in `ChainageCreate`

#### [NEW] [lane_routes.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/routes/lane_routes.py)
Full CRUD at `/lanes`:
```
POST   /lanes              — create lane in a chainage
GET    /lanes?chainage_id= — list (filterable)
GET    /lanes/{id}         — get by ID
PUT    /lanes/{id}         — update
DELETE /lanes/{id}         — delete (cascades to videos)
```
Schemas: `LaneCreate`, `LaneUpdate`, `LaneResponse`

#### [DELETE] [location_routes.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/routes/location_routes.py)
Remove entirely — replaced by `chainage_routes.py` + `lane_routes.py`.

#### [MODIFY] [upload_process_routes.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/routes/upload_process_routes.py)
- Add optional `chainage_id` + `lane_id` form fields to `/upload`
- Pass them through to `upload_service.upload_video()`

#### [MODIFY] [summary_routes.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/routes/summary_routes.py)
- Replace all [Location](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/models/location.py#9-66) references → `Chainage` + `Lane`
- Restructure analytics responses:
  - Project summary: `packages → chainages → lanes → detections`
  - Package summary: `chainages → lanes → detections`
  - Rename `/summary/locations/{id}` → `/summary/chainages/{id}`
- Add lane-level grouping in detection summaries

---

### Services

#### [MODIFY] [location_mapper.py → chainage_mapper.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/services/location_mapper.py)
- Rename file to `chainage_mapper.py`
- Rename [get_location_hierarchy()](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/services/location_mapper.py#8-44) → `get_chainage_hierarchy()`
- Include lane info in hierarchy response
- Keep [validate_gps_bounds()](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/services/location_mapper.py#46-83) unchanged

#### [MODIFY] [upload_service.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/services/upload_service.py)
- Accept `chainage_id` + `lane_id` in [upload_video()](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/routes/upload_process_routes.py#25-49)
- Pass them to `crud.create_video()`

---

### App Initialization

#### [MODIFY] [\_\_init\_\_.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/__init__.py)
- Replace `location_router` import with `chainage_router` + `lane_router`
- Update endpoint listing in root route
- Register new routers: `/api/v1/chainages`, `/api/v1/lanes`

---

### Migration & Supporting

#### [NEW] Alembic migration
- Generate via `alembic revision --autogenerate -m "replace_locations_with_chainages_and_lanes"`
- Since this is SQLite and the change is destructive, we'll delete the old DB and let `init_db()` recreate tables from scratch

#### [MODIFY] [env.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/alembic/env.py)
- Replace [location](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/app/db/crud_hierarchy.py#165-168) import → `chainage`, add `lane` import

#### [MODIFY] [seed_data.py](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/seed_data.py)
- Replace [create_location()](file:///d:/Pothole-RoadSign%20Detection/VisionRoad-Backend/seed_data.py#52-83) → `create_chainage()` + `create_lane()` 
- Each chainage gets default lanes ("LHS", "RHS")
- Update all API calls to use `/chainages` and `/lanes` endpoints

---

## Verification Plan

### Automated (Server Startup + API Testing)
1. **Delete old DB**: `del visionroad.db`
2. **Start server**: `python -m uvicorn main:app --reload`
   - Verify tables are created without errors
3. **Run seed script**: `python seed_data.py`
   - Verify all chainages and lanes are created successfully
4. **Test via Swagger UI**: Open `http://localhost:8000/docs`
   - Test all CRUD endpoints for `/chainages` and `/lanes`
   - Test `/upload` with chainage_id + lane_id
   - Test `/summary/*` endpoints return chainage+lane structure

### Manual Verification
- Browse Swagger docs and verify no `/locations` endpoints remain
- Confirm that the analytics responses now group by chainage → lane
