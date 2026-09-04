# DRISHTI — Frontend tasks

**You own:** `apps/dashboard/`.
**You must not touch:** `apps/api/`, anything under `ml/`, or `docs/API_ENDPOINTS.md`.

**You can start today.** Build against `docs/API_ENDPOINTS.md`, not against whatever the backend
currently returns — it is mid-change. Mock the responses and you are unblocked.

---

## 0 · Setup (once)

```bash
git clone https://github.com/Rehan9599/Sonar-Drishti.git
cd Sonar-Drishti
git checkout -b feat/dashboard-ui

corepack enable
pnpm install                       # installs the whole workspace

pnpm --filter @drishti/dashboard dev
# → http://localhost:5173
```

If the filter name errors, check the `name` field in `apps/dashboard/package.json` and use that.
Fallback: `cd apps/dashboard && pnpm dev`.

---

## What already exists — read before writing

| File | Lines | State |
|---|---|---|
| `vite.config.js` | 27 | ✅ **done** — proxies `/api` → `:8000`, `/ws` → `ws://:8000`, vitest+jsdom configured |
| `package.json` | 27 | ✅ **done** — react 18, react-router-dom, react-leaflet, leaflet, axios, recharts all present |
| `src/websocket.js` | 49 | ✅ **done and contract-correct** — `connectToJob(jobId, handlers)` handles `detection.partial` / `.complete` / `.failed` |
| `src/api.js` | 22 | ⚠️ partial — has `getJobs`, `createJob(inputPath)`, `getJob`. Missing upload/detections/review/export |
| `src/main.jsx` | 18 | ⚠️ check it mounts `<App/>` inside a router |
| `src/styles/globals.css` | 211 | ✅ done |
| `src/App.jsx` | **0** | ✗ empty |
| `src/components/UploadPanel/UploadPanel.jsx` | **0** | ✗ empty |
| `src/components/MapView/MapView.jsx` | **0** | ✗ empty |
| `src/components/ReviewQueue/ReviewQueue.jsx` | **0** | ✗ empty |
| `src/components/ImageOverlay/ImageOverlay.jsx` | **0** | ✗ empty |
| `src/hooks/useDetectionSocket.js` | 3 | ✗ stub |
| `src/state/store.js` | **0** | ✗ empty |
| `src/pages/LiveFeedPage.jsx` | 3 | ✗ stub |
| `src/pages/UploadResultsPage.jsx` | 2 | ✗ stub |
| `src/pages/ExportPage.figma-link.md` | 4 | ✗ placeholder, needs a real `.jsx` |
| `src/pages/GlobalMapPage.figma-link.md` | 6 | ✗ placeholder, needs a real `.jsx` |

**`src/websocket.js` is good — extend it, don't rewrite it.** It already parses the three event
types the backend emits.

---

## The contract you build against

Every detection object, everywhere:

```json
{
  "detection_id": "a138c61c-…",
  "job_id": "176771df-…",
  "ping_id": "DATA0000106.H-PU#1401",
  "timestamp": "2015-08-12T09:08:28.650000+00:00",
  "latitude": 50.3937068,
  "longitude": -7.7132752,
  "class_label": "shipwreck",
  "confidence_score": 64.0,
  "bounding_geometry": {
    "bbox": [475.9, 11.4, 640.0, 153.1],
    "mask_polygon": [],
    "width_m": 55.2,
    "height_m": 47.63
  },
  "across_track_m": 80.29,
  "side": "starboard",
  "review_status": "pending_review",
  "source_file": "DATA0000106.H-PU"
}
```

Routes:

| Method | Route | Returns |
|---|---|---|
| POST | `/api/upload/` | multipart `file` (+ optional `xtf`, `nav`) → `202 {job_id, status}` |
| GET | `/api/jobs/<job_id>/` | job status + `progress` (0–1) + `detection_count` |
| GET | `/api/detections/<job_id>/` | array of the record above |
| PATCH | `/api/detections/<detection_id>/review/` | `{review_status, actor}` → updated record |
| GET | `/api/export/<job_id>/?format=json\|csv\|geojson` | download |
| WS | `/ws/jobs/<job_id>/` | `detection.partial` `{tile_index, detections[]}` · `detection.complete` · `detection.failed` |

---

## 1 · Extend `src/api.js`

Keep what's there; add the missing calls.

```javascript
import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || "/api",
});

export async function uploadLog(file, { xtf, nav } = {}) {
  const fd = new FormData();
  fd.append("file", file);
  if (xtf) fd.append("xtf", xtf);
  if (nav) fd.append("nav", nav);
  const { data } = await api.post("/upload/", fd);
  return data;                                  // { job_id, status }
}

export async function getJobs() {
  const { data } = await api.get("/jobs/");
  return data;
}

export async function getJob(jobId) {
  const { data } = await api.get(`/jobs/${jobId}/`);
  return data;                                  // { status, progress, detection_count, ... }
}

export async function getDetections(jobId) {
  const { data } = await api.get(`/detections/${jobId}/`);
  return data;                                  // Detection[]
}

export async function reviewDetection(detectionId, reviewStatus, actor = "analyst") {
  const { data } = await api.patch(`/detections/${detectionId}/review/`, {
    review_status: reviewStatus,                // analyst_confirmed | analyst_rejected
    actor,
  });
  return data;
}

export function exportUrl(jobId, format = "json") {
  const base = import.meta.env.VITE_API_BASE || "/api";
  return `${base}/export/${jobId}/?format=${format}`;
}
```

Delete `createJob(inputPath)` — that was a dev-only path and the backend is replacing it.

---

## 2 · `src/hooks/useDetectionSocket.js`

Wrap the existing `connectToJob`. Currently a 3-line stub.

```javascript
import { useEffect, useRef, useState } from "react";
import { connectToJob } from "../websocket";

export function useDetectionSocket(jobId) {
  const [detections, setDetections] = useState([]);
  const [tilesDone, setTilesDone] = useState(0);
  const [status, setStatus] = useState("idle");   // idle | live | complete | failed
  const [error, setError] = useState(null);
  const sockRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;
    setDetections([]);
    setTilesDone(0);
    setStatus("live");
    setError(null);

    sockRef.current = connectToJob(jobId, {
      onPartial: (msg) => {
        setDetections((prev) => [...prev, ...(msg.detections || [])]);
        setTilesDone((n) => Math.max(n, (msg.tile_index ?? 0) + 1));
      },
      onComplete: () => setStatus("complete"),
      onFailed: (msg) => { setError(msg.error || "job failed"); setStatus("failed"); },
    });

    return () => sockRef.current?.close();
  }, [jobId]);

  return { detections, tilesDone, status, error };
}
```

---

## 3 · `src/App.jsx` — routing shell

```jsx
import { Navigate, Route, Routes } from "react-router-dom";

import ExportPage from "./pages/ExportPage.jsx";
import GlobalMapPage from "./pages/GlobalMapPage.jsx";
import LiveFeedPage from "./pages/LiveFeedPage.jsx";
import UploadResultsPage from "./pages/UploadResultsPage.jsx";

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <span className="brand">DRISHTI</span>
        <nav>
          <a href="/upload">Upload</a>
          <a href="/map">Map</a>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<Navigate to="/upload" replace />} />
        <Route path="/upload" element={<UploadResultsPage />} />
        <Route path="/jobs/:jobId" element={<LiveFeedPage />} />
        <Route path="/jobs/:jobId/export" element={<ExportPage />} />
        <Route path="/map" element={<GlobalMapPage />} />
      </Routes>
    </div>
  );
}
```

Confirm `src/main.jsx` wraps `<App/>` in `<BrowserRouter>` and imports
`leaflet/dist/leaflet.css` — without that CSS the map renders as grey tiles.

---

## 4 · `UploadPanel.jsx`

```jsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadLog } from "../../api";

export default function UploadPanel() {
  const [file, setFile] = useState(null);
  const [xtf, setXtf] = useState(null);
  const [nav, setNav] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const navigate = useNavigate();

  async function submit(e) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setErr(null);
    try {
      const { job_id } = await uploadLog(file, { xtf, nav });
      navigate(`/jobs/${job_id}`);
    } catch (e2) {
      setErr(e2?.response?.data?.detail ?? "Upload failed. Check the file and try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="upload-panel" onSubmit={submit}>
      <label>Sonar image or log <span className="req">required</span>
        <input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.xtf"
               onChange={(e) => setFile(e.target.files[0])} />
      </label>
      <label>XTF navigation file <span className="opt">optional — needed for coordinates</span>
        <input type="file" accept=".xtf" onChange={(e) => setXtf(e.target.files[0])} />
      </label>
      <label>Navigation CSV <span className="opt">optional</span>
        <input type="file" accept=".csv" onChange={(e) => setNav(e.target.files[0])} />
      </label>
      <button type="submit" disabled={!file || busy}>
        {busy ? "Uploading…" : "Detect"}
      </button>
      {err && <p className="error">{err}</p>}
      <p className="hint">
        Without an XTF or navigation file, detections have no coordinates and will not appear on the map.
      </p>
    </form>
  );
}
```

---

## 5 · `LiveFeedPage.jsx` — the demo moment

```jsx
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import MapView from "../components/MapView/MapView.jsx";
import ReviewQueue from "../components/ReviewQueue/ReviewQueue.jsx";
import { getDetections, getJob } from "../api";
import { useDetectionSocket } from "../hooks/useDetectionSocket";

export default function LiveFeedPage() {
  const { jobId } = useParams();
  const { detections: live, tilesDone, status, error } = useDetectionSocket(jobId);
  const [job, setJob] = useState(null);
  const [settled, setSettled] = useState([]);

  useEffect(() => {
    const t = setInterval(() => getJob(jobId).then(setJob).catch(() => {}), 2000);
    return () => clearInterval(t);
  }, [jobId]);

  // once the job finishes, take the authoritative list from the DB
  useEffect(() => {
    if (status === "complete") getDetections(jobId).then(setSettled).catch(() => {});
  }, [status, jobId]);

  const rows = settled.length ? settled : live;

  return (
    <div className="live-feed">
      <div className="status-bar">
        <span className={`badge ${status}`}>{status}</span>
        <span>{rows.length} detections</span>
        <span>{tilesDone} tiles processed</span>
        {job && <progress value={job.progress ?? 0} max="1" />}
        <Link to={`/jobs/${jobId}/export`}>Export</Link>
      </div>
      {error && <p className="error">{error}</p>}
      <MapView detections={rows} />
      <ReviewQueue detections={rows} onUpdated={(d) =>
        setSettled((prev) => prev.map((x) =>
          x.detection_id === d.detection_id ? d : x))} />
    </div>
  );
}
```

---

## 6 · `MapView.jsx` — 🔴 uncertainty circles, not bare pins

**This is a correctness requirement, not styling.** A detection is a **search area, not a survey
fix** — our two navigation paths placed the same target ~122 m apart. A bare pin implies metre
accuracy we do not have, and someone could take a boat to it.

```jsx
import { Circle, CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";

const COLOR = {
  auto_confirmed: "#146c43",
  pending_review: "#96620c",
  analyst_confirmed: "#0a7a1e",
  analyst_rejected: "#78888e",
};

// positional uncertainty grows with across-track distance
function uncertaintyM(d) {
  const base = 40;
  const across = Math.abs(d.across_track_m ?? 0);
  return Math.max(base, base + across * 1.0);
}

export default function MapView({ detections }) {
  const located = detections.filter((d) => d.latitude != null && d.longitude != null);
  const centre = located.length
    ? [located[0].latitude, located[0].longitude]
    : [50.39, -7.71];

  return (
    <MapContainer center={centre} zoom={13} style={{ height: "480px", width: "100%" }}>
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {located.map((d) => {
        const colour = COLOR[d.review_status] ?? "#0d7490";
        const pos = [d.latitude, d.longitude];
        return (
          <div key={d.detection_id}>
            {/* the honest part: an area, not a point */}
            <Circle center={pos} radius={uncertaintyM(d)}
                    pathOptions={{ color: colour, weight: 1, fillOpacity: 0.12 }} />
            <CircleMarker center={pos} radius={5}
                          pathOptions={{ color: colour, fillColor: colour, fillOpacity: 0.9 }}>
              <Popup>
                <strong>{d.class_label}</strong><br />
                {d.confidence_score?.toFixed(1)}% · {d.review_status}<br />
                {d.side} {d.across_track_m?.toFixed(0)} m across-track<br />
                {d.bounding_geometry?.width_m && (
                  <>~{d.bounding_geometry.width_m.toFixed(1)} × {d.bounding_geometry.height_m?.toFixed(1)} m<br /></>
                )}
                <small>±{uncertaintyM(d).toFixed(0)} m · {d.ping_id}</small>
              </Popup>
            </CircleMarker>
          </div>
        );
      })}
    </MapContainer>
  );
}
```

Add a visible legend caption under the map:

> *Circles show positional uncertainty. Advisory only — not a navigational chart.*

---

## 7 · `ReviewQueue.jsx`

Only `pending_review` rows appear here. That is what makes the queue short enough to work through.

```jsx
import { useState } from "react";
import { reviewDetection } from "../../api";

export default function ReviewQueue({ detections, onUpdated }) {
  const [busy, setBusy] = useState(null);
  const pending = detections.filter((d) => d.review_status === "pending_review");

  async function act(d, verdict) {
    setBusy(d.detection_id);
    try {
      const updated = await reviewDetection(d.detection_id, verdict);
      onUpdated?.(updated);
    } finally {
      setBusy(null);
    }
  }

  if (!pending.length) {
    return <p className="queue-empty">Nothing awaiting review.</p>;
  }

  return (
    <section className="review-queue">
      <h3>Review queue <span className="count">{pending.length}</span></h3>
      <table>
        <thead>
          <tr><th>Class</th><th>Score</th><th>Position</th><th>Ping</th><th /></tr>
        </thead>
        <tbody>
          {pending.map((d) => (
            <tr key={d.detection_id}>
              <td>{d.class_label}</td>
              <td>{d.confidence_score?.toFixed(1)}%</td>
              <td>{d.latitude != null
                    ? `${d.latitude.toFixed(5)}, ${d.longitude.toFixed(5)}`
                    : "—"}</td>
              <td><code>{d.ping_id || "—"}</code></td>
              <td>
                <button disabled={busy === d.detection_id}
                        onClick={() => act(d, "analyst_confirmed")}>Confirm</button>
                <button disabled={busy === d.detection_id}
                        onClick={() => act(d, "analyst_rejected")}>Reject</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
```

---

## 8 · `ExportPage.jsx` (replace the `.figma-link.md` placeholder)

```jsx
import { useParams } from "react-router-dom";
import { exportUrl } from "../api";

export default function ExportPage() {
  const { jobId } = useParams();
  return (
    <section className="export-page">
      <h2>Export report</h2>
      <ul className="export-links">
        <li><a href={exportUrl(jobId, "json")} download>JSON — full report</a></li>
        <li><a href={exportUrl(jobId, "csv")} download>CSV — flat table for spreadsheets</a></li>
        <li><a href={exportUrl(jobId, "geojson")} download>GeoJSON — for GIS / QGIS</a></li>
      </ul>
    </section>
  );
}
```

`GlobalMapPage.jsx`: fetch `getJobs()`, let the user pick one, then render `<MapView/>` with that
job's detections. Same component, no new map code.

Delete both `.figma-link.md` files once the real pages exist.

---

## 9 · Render four classes, not five

Show `submarine_pipeline`, `shipwreck`, `mine_cylinder`, `ghost_net`.

`crab_pot` is trained as a hard negative and **filtered out of the product**. If one ever appears
in a response, that is a backend bug — log it, don't render it.

Suggested labels for the UI:

| `class_label` | Display |
|---|---|
| `submarine_pipeline` | Pipeline / cable |
| `shipwreck` | Shipwreck |
| `mine_cylinder` | Cylinder — suspected ordnance |
| `ghost_net` | Ghost net |

For `mine_cylinder`, add a warning in the popup: *"Suspected object — do not approach. Report to
the maritime authority."*

---

## 10 · Work against mocks until the backend lands

Until the backend merges its five routes, put this in `src/api.js` behind a flag:

```javascript
const MOCK = import.meta.env.VITE_MOCK === "1";

const MOCK_DETECTIONS = [{
  detection_id: "a138c61c-59b4-4eb2-815b-205b9d15b74f",
  job_id: "176771df-7360-4b31-a4ef-c92e3cddf393",
  ping_id: "DATA0000106.H-PU#1401",
  timestamp: "2015-08-12T09:08:28.650000+00:00",
  latitude: 50.3937068, longitude: -7.7132752,
  class_label: "shipwreck", confidence_score: 64.0,
  bounding_geometry: { bbox: [475.9, 11.4, 640.0, 153.1], mask_polygon: [],
                       width_m: 55.2, height_m: 47.63 },
  across_track_m: 80.29, side: "starboard",
  review_status: "pending_review", source_file: "DATA0000106.H-PU",
}];

export async function getDetections(jobId) {
  if (MOCK) return MOCK_DETECTIONS;
  const { data } = await api.get(`/detections/${jobId}/`);
  return data;
}
```

Run with `VITE_MOCK=1 pnpm dev`. Remove the flag before opening the PR.

---

## 11 · Test

```bash
pnpm --filter @drishti/dashboard build      # must compile clean
pnpm --filter @drishti/dashboard test       # vitest
pnpm --filter @drishti/dashboard lint
```

Then against the real backend:

```bash
docker compose up -d                  # backend team's stack
pnpm --filter @drishti/dashboard dev
```

Upload `demo/tiles/synth_ghost_net_00002.jpg` from the UI. Expect: job created, WebSocket goes
live, a `ghost_net` detection at ~95 % appears as **auto_confirmed** (green), and the review queue
stays empty because 95 % is above the 80 % auto bar.

### Checklist

- [ ] Upload accepts a file and navigates to the live feed
- [ ] WebSocket status badge goes `live` → `complete`
- [ ] Detections appear as they stream, before the job finishes
- [ ] Map renders **an uncertainty circle plus a marker** for every located detection
- [ ] Popup shows class, calibrated score, side, across-track distance, `ping_id`
- [ ] Detections with `latitude: null` are skipped on the map (no crash)
- [ ] Review queue lists only `pending_review`; Confirm/Reject persists after refresh
- [ ] Colours follow `review_status`, not a hardcoded score threshold
- [ ] All three export links download
- [ ] `crab_pot` is never rendered
- [ ] `pnpm build` is clean; no `VITE_MOCK` left in the code

---

## 12 · Open the PR

```bash
cd /d/Sonar-Drishti
git status                      # no node_modules, no dist/, no .env
git add -A
git commit -m "feat(dashboard): upload, live feed, map, review queue and export

- api.js: uploadLog (multipart), getDetections, reviewDetection, exportUrl
- useDetectionSocket hook wrapping the existing websocket.js client
- App.jsx routing shell + four real pages
- UploadPanel: sonar file + optional XTF/nav
- LiveFeedPage: WebSocket streaming with progress, settles to DB list on completion
- MapView: uncertainty circles (not bare pins) coloured by review_status
- ReviewQueue: pending_review only, confirm/reject via PATCH
- ExportPage: json/csv/geojson
- renders four shipped classes; crab_pot excluded

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"

git push -u origin feat/dashboard-ui
gh pr create --title "Dashboard: upload, live feed, map, review queue, export" --body "$(cat <<'EOF'
Implements the frontend half of docs/HANDOVER.md, built against the frozen contract in
docs/API_ENDPOINTS.md.

## Done
- Five screens: Upload · Live feed · Map · Review queue · Export
- Live per-tile detections over WebSocket (`detection.partial`), settling to the DB list on `detection.complete`
- **Uncertainty circles rather than bare pins** — a detection is a search area, not a survey fix (our two nav paths placed the same target ~122 m apart)
- Colours driven by `review_status`, not hardcoded thresholds
- Review queue shows only `pending_review` and PATCHes the verdict
- Renders the four shipped classes; `crab_pot` excluded

## Verified
- Uploaded `demo/tiles/synth_ghost_net_00002.jpg` → ghost_net 95.3% auto_confirmed on the map
- WebSocket badge live → complete; detections stream before completion
- Detections with null coordinates are skipped without crashing
- `pnpm build` clean

## Not in scope
- authentication (Phase 2)
- the public mobile debris map (Mode C)
- sonar-tile bounding-box overlay (`ImageOverlay`) — next PR

EOF
)"
```

---

## Rules

1. **`docs/API_ENDPOINTS.md` is frozen.** If you need a field that isn't there, ask — don't invent one or reshape a response client-side.
2. **Never render a bare pin.** Always the uncertainty circle. This is the one non-negotiable UI rule.
3. **Read the review band from `review_status`**, never by comparing `confidence_score` to 80 in the UI. The model layer owns those thresholds.
4. **Quote metrics from `docs/METRICS_PROTOCOL.md`** if any number appears in the UI — mAP@50 is **0.641**.
