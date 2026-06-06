# AIWardrobe 6-Month Body-Fit Virtual Try-On Implementation Plan

## Goal

Build a production path where a user can order clothes online more correctly by combining:

- a mobile phone capture flow,
- a calibrated 3D body representation,
- Apple/iPhone height measurement as the scale anchor,
- clothing photos,
- clothing size and construction parameters,
- virtual try-on rendering,
- and explicit fit recommendations.

The target user experience is:

1. User creates a private digital body profile from a photo and measured height.
2. User selects one or more garments.
3. The app shows the clothes on a personalized mannequin/body.
4. The app explains whether the selected size should fit, where it may be tight/loose/long/short, and what size to choose.

Important product truth: a clothing photo alone is not enough for accurate fit. The photo provides visual appearance. Garment measurements and fabric properties provide physical fit. The body mesh provides shape. Height calibration provides scale.

## Current Project Baseline

Relevant existing files:

- `features/try-on/AITryOnScreen.tsx`
  - Current mobile try-on UI.
  - Sends selected garments to Mobile-VTON.
  - Supports top/layer/pants/shoes slots.

- `features/try-on/utils/mannequin3D.ts`
  - Existing Three.js/procedural mannequin logic.
  - Already supports height, weight, and body type scaling.
  - This should become the first personalized mannequin layer.

- `store/avatarStore.ts`
  - Existing avatar state.
  - Stores height, weight, body type, and gender.

- `api/services/mobileVtonClient.js`
  - Express-side client for Mobile-VTON backend.

- `api/services/strategies/mobileVton.js`
  - Current Mobile-VTON rendering strategy.
  - Supports multi-garment ordering.

- `mobile-vton-service/main.py`
  - FastAPI GPU service.
  - Current render path uses SD1.5 inpainting + IP-Adapter.
  - Multi-garment path is still sequential dressing.

- `docs/MULTI_GARMENT_VTON_ROADMAP.md`
  - Existing roadmap for faster fused multi-garment VTON.
  - This plan should connect to it rather than replace it.

## External Technology Baseline

### SAM 3D Body

SAM 3D Body can reconstruct a full-body 3D human mesh from a single RGB image. It estimates body pose and body shape using Meta's MHR body representation.

Use it for:

- body mesh reconstruction,
- pose/body shape estimation,
- measurement extraction,
- personalized mannequin/body generation.

Do not treat it as:

- a guaranteed exact body scan,
- a medical-grade anthropometric system,
- a source of perfect chest/waist/hip measurements from one casual photo.

Expected production stance:

- use SAM 3D Body as the reconstruction base,
- calibrate scale with measured height,
- add user correction controls,
- store confidence scores for every derived measurement.

### Apple Measure / LiDAR Height

Apple's Measure app can measure a person's height on supported LiDAR iPhone/iPad models. Use that measurement as the high-confidence scale anchor.

Implementation options:

- MVP: user manually enters height measured by Apple Measure.
- Later: build an in-app ARKit height flow or guide the user through Apple's Measure workflow.

## System Architecture

```text
Mobile App
  |
  |-- Body Capture Flow
  |     |-- full-body photo upload
  |     |-- height input / Apple Measure result
  |     |-- weight, gender, body type
  |     |-- manual correction controls
  |
  |-- Try-On Flow
        |-- select garments
        |-- choose size
        |-- view personalized mannequin/body
        |-- view rendered outfit
        |-- view fit recommendation

API Server
  |
  |-- Body Profile API
  |     |-- create profile
  |     |-- analyze photo
  |     |-- store measurements
  |     |-- store mesh refs
  |
  |-- Garment Fit API
  |     |-- normalize garment data
  |     |-- compare garment measurements to body
  |     |-- output fit assessment
  |
  |-- Try-On API
        |-- accepts body profile
        |-- accepts garment photos
        |-- accepts garment parameters
        |-- calls Mobile-VTON

GPU Services
  |
  |-- SAM 3D Body Service
  |     |-- image -> body mesh
  |     |-- mesh -> measurements
  |
  |-- Mobile-VTON Service
        |-- personalized mannequin image/body render
        |-- garment image conditioning
        |-- multi-garment render
```

## Data Models

### BodyProfile

```ts
type BodyProfileSource = 'manual' | 'photo_sam_3d_body' | 'arkit_height' | 'hybrid';

interface BodyMeasurement {
  valueCm: number;
  confidence: 'low' | 'medium' | 'high';
  source: BodyProfileSource;
}

interface BodyProfile {
  id: string;
  userId: string;
  status: 'draft' | 'analyzing' | 'ready' | 'failed';
  gender?: 'male' | 'female' | 'other' | 'prefer_not_to_say';
  height: BodyMeasurement;
  weightKg?: number;
  bodyType?: 'ectomorph' | 'average' | 'mesomorph' | 'endomorph' | 'hourglass' | 'pear';
  measurements: {
    shoulderWidth?: BodyMeasurement;
    chest?: BodyMeasurement;
    waist?: BodyMeasurement;
    hips?: BodyMeasurement;
    torsoLength?: BodyMeasurement;
    armLength?: BodyMeasurement;
    inseam?: BodyMeasurement;
    thigh?: BodyMeasurement;
    calf?: BodyMeasurement;
  };
  mesh?: {
    provider: 'sam_3d_body';
    meshUrl?: string;
    previewImageUrl?: string;
    rawOutputUrl?: string;
    version: string;
  };
  privacy: {
    retainSourcePhoto: boolean;
    retainMesh: boolean;
  };
  createdAt: string;
  updatedAt: string;
}
```

### GarmentPhysicalProfile

```ts
interface GarmentMeasurement {
  valueCm: number;
  unitSource?: 'cm' | 'inch';
  confidence?: 'low' | 'medium' | 'high';
}

interface GarmentPhysicalProfile {
  garmentId: string;
  sizeLabel: string;
  category:
    | 'top'
    | 'shirt'
    | 'jacket'
    | 'coat'
    | 'pants'
    | 'jeans'
    | 'skirt'
    | 'dress'
    | 'shoes';
  fitIntent?: 'compression' | 'slim' | 'regular' | 'relaxed' | 'oversized';
  stretch?: 'none' | 'low' | 'medium' | 'high';
  material?: string[];
  measurements: {
    chest?: GarmentMeasurement;
    waist?: GarmentMeasurement;
    hips?: GarmentMeasurement;
    shoulderWidth?: GarmentMeasurement;
    sleeveLength?: GarmentMeasurement;
    bodyLength?: GarmentMeasurement;
    inseam?: GarmentMeasurement;
    rise?: GarmentMeasurement;
    thigh?: GarmentMeasurement;
    hemOpening?: GarmentMeasurement;
    shoeLength?: GarmentMeasurement;
  };
}
```

### FitAssessment

```ts
interface FitAssessment {
  garmentId: string;
  bodyProfileId: string;
  selectedSize: string;
  overall:
    | 'too_small'
    | 'tight'
    | 'good_fit'
    | 'relaxed'
    | 'oversized'
    | 'too_large'
    | 'unknown';
  confidence: 'low' | 'medium' | 'high';
  sizeRecommendation?: {
    recommendedSize: string;
    reason: string;
  };
  zones: Array<{
    zone:
      | 'shoulders'
      | 'chest'
      | 'waist'
      | 'hips'
      | 'arms'
      | 'sleeves'
      | 'torso_length'
      | 'thigh'
      | 'inseam'
      | 'calf'
      | 'feet';
    status: 'too_tight' | 'snug' | 'good' | 'loose' | 'too_loose' | 'too_short' | 'too_long' | 'unknown';
    deltaCm?: number;
    message: string;
  }>;
}
```

## 6-Month Timeline

## Month 1: Foundation, Data Models, and MVP Body Profile

### Goal

Create the body-fit foundation without depending on SAM 3D Body yet. The app should store body profiles, garment physical parameters, and produce basic fit assessments from manual measurements.

### Engineering Tasks

1. Add `BodyProfile` model and store.
   - Extend `store/avatarStore.ts` or create `store/bodyProfileStore.ts`.
   - Store height, weight, gender, body type, and manual measurements.
   - Support profile status: draft/analyzing/ready/failed.

2. Add backend body profile API.
   - `POST /api/body-profiles`
   - `GET /api/body-profiles/me`
   - `PATCH /api/body-profiles/:id`
   - `DELETE /api/body-profiles/:id`

3. Add garment physical parameters.
   - Extend shop catalog item types.
   - Add optional `physicalProfile`.
   - Add size chart data structure.
   - Add admin/dev seed data for several known garments.

4. Build fit engine v1.
   - Input: body profile + garment physical profile + selected size.
   - Output: `FitAssessment`.
   - Implement simple ease rules:
     - tops: chest, shoulders, sleeve, body length;
     - pants: waist, hips, thigh, inseam;
     - jackets: chest, shoulder, sleeve, layering ease;
     - shoes: foot length if available.

5. Add try-on payload support.
   - Extend Mobile-VTON request object to include:
     - `body_profile_id`,
     - `body_measurements`,
     - `fit_assessment`,
     - `garment_physical_profile`.

6. Add mobile UI for body profile setup.
   - Height input.
   - Weight input.
   - Body type selector.
   - Manual optional measurements.
   - Privacy copy: source photo/mesh retention controls planned for later.

### Acceptance Criteria

- User can create/edit a body profile.
- User can select a garment size.
- App can show a basic fit result before render.
- Existing try-on still works.
- No SAM dependency is required yet.

### Risks

- Product data may not include garment measurements.
- Mitigation: allow unknown measurements and lower confidence instead of blocking.

## Month 2: Personalized Mannequin and Fit-Aware Try-On Payloads

### Goal

Use the user's body profile to generate a more personalized mannequin and pass fit context into the try-on pipeline.

### Engineering Tasks

1. Connect `BodyProfile` to `mannequin3D.ts`.
   - Replace hardcoded/default values with stored profile values.
   - Keep existing procedural/GLB fallback.
   - Add normalized body dimensions object:
     - height,
     - width scale,
     - chest proxy,
     - waist proxy,
     - hip proxy,
     - shoulder proxy.

2. Add a body-profile preview.
   - User can see their personalized mannequin.
   - User can adjust height/weight/body type.
   - Later this preview will be replaced or improved by SAM mesh output.

3. Add fit-aware prompt/context for Mobile-VTON.
   - Include text like:
     - "slightly tight at chest",
     - "relaxed waist",
     - "sleeves slightly long",
     - "pants cropped above ankle".
   - Do not rely only on prompt text; store fit data in response diagnostics.

4. Add fit panel in try-on UI.
   - Overall fit badge.
   - Zone-level notes.
   - Confidence indicator.
   - Size recommendation.

5. Update API schema.
   - Express API receives and forwards body/fit context.
   - Mobile-VTON service can ignore fields initially but should accept them.

### Acceptance Criteria

- Try-on mannequin visually changes with body profile.
- Fit assessment appears next to the render.
- Mobile-VTON endpoints accept new fields without breaking.
- Multi-garment flow still supports top/layer/pants/shoes.

### Risks

- Prompt-only fit conditioning may not visibly affect the render.
- Mitigation: treat rendered image and fit recommendation as separate outputs. Fit recommendation must remain explicit.

## Month 3: SAM 3D Body Service Prototype

### Goal

Integrate SAM 3D Body as a backend prototype that can process a full-body photo and produce mesh/measurement artifacts.

### Engineering Tasks

1. Create `sam-3d-body-service`.
   - Separate Python service from `mobile-vton-service`.
   - Use a GPU-capable deployment target.
   - Add endpoints:
     - `GET /health`
     - `POST /analyze-body`
     - `GET /jobs/:id`

2. Add checkpoint/access management.
   - Confirm Hugging Face access.
   - Confirm SAM License compatibility for intended use.
   - Store model version in every output.

3. Implement body analysis job.
   - Input:
     - source photo,
     - measured height,
     - optional mask/keypoints.
   - Output:
     - mesh file,
     - rendered preview,
     - raw model output,
     - estimated measurements,
     - confidence scores.

4. Implement scale calibration.
   - Use Apple/iPhone height as true height.
   - Scale mesh to match height.
   - Recompute measurements after scaling.

5. Add source photo quality validation.
   - Full body visible.
   - Standing pose preferred.
   - Camera not too close/far.
   - Minimal occlusion.
   - Low-confidence fallback if poor input.

6. Add mobile upload flow.
   - Upload full-body photo.
   - Submit height.
   - Poll job status.
   - Show analyzing/ready/failed state.

### Acceptance Criteria

- A user photo can produce a body analysis job.
- Mesh/preview artifacts are stored.
- Body profile can be populated from SAM-derived measurements.
- Height calibration is applied.
- Failures are clear and recoverable.

### Risks

- SAM output may be inaccurate for loose clothing, occlusion, seated poses, or unusual body shapes.
- Mitigation: use confidence scores and manual correction controls.

## Month 4: Measurement Extraction and Manual Correction

### Goal

Make SAM-derived body data useful for fit by extracting measurements and allowing users to correct uncertain values.

### Engineering Tasks

1. Improve measurement extraction.
   - Height: high confidence from Apple/manual measured height.
   - Shoulder width: mesh-based.
   - Chest/waist/hips: mesh cross-section estimates.
   - Inseam/arm length: skeleton/landmark-based.
   - Thigh/calf: mesh cross-section estimates.

2. Build measurement confidence logic.
   - High: directly measured or geometrically stable.
   - Medium: estimated from visible mesh.
   - Low: inferred from incomplete/occluded input.

3. Add manual correction UI.
   - Chest, waist, hips, inseam, shoulder, sleeve/arm length.
   - Each corrected value updates confidence/source.
   - Store user-corrected values separately from SAM estimates.

4. Add body profile versioning.
   - `bodyProfile.version`
   - `measurementSourceHistory`
   - Keep previous profiles for comparison if user updates body.

5. Add privacy controls.
   - Delete source photo.
   - Delete mesh.
   - Keep only measurements.
   - Re-run analysis if source photo is retained.

6. Build QA dataset.
   - At least 30 internal test profiles:
     - short/tall,
     - slim/average/muscular/curvy/larger bodies,
     - different clothing in source photos,
     - different camera distances.

### Acceptance Criteria

- Body measurements are visible and editable.
- Fit engine uses corrected values over estimated values.
- User can delete source photo/mesh.
- QA dataset exposes failure modes.

### Risks

- Circumference extraction from mesh may be noisy.
- Mitigation: confidence scoring and user correction are required product features, not optional.

## Month 5: Garment Data Pipeline and Fit Recommendation Quality

### Goal

Make clothing parameters reliable enough to support useful purchase decisions.

### Engineering Tasks

1. Build garment parameter ingestion.
   - Manual admin entry.
   - Product size chart parser if available.
   - Unit normalization: inch/cm.
   - Size variant support.

2. Extend catalog schema.
   - Each garment can have multiple size profiles.
   - Store size chart source and confidence.
   - Store material/stretch/fit intent.

3. Add garment photo processing.
   - Keep current product image path for VTON appearance.
   - Add background removal/cutout cache.
   - Add garment category validation.

4. Improve fit engine rules.
   - Category-specific ease ranges.
   - Stretch-aware tolerance.
   - Layering-aware jacket/coat ease.
   - Oversized/slim fit intent.
   - Different rules for men's/women's sizing where needed.

5. Size recommendation engine.
   - Compare selected size and adjacent sizes.
   - Recommend best size based on zone penalties.
   - Explain the recommendation in plain language.

6. Add fit regression tests.
   - Unit tests for fit calculations.
   - Golden examples:
     - shirt too tight at chest,
     - pants too long,
     - jacket tight over hoodie,
     - oversized hoodie expected to be loose.

### Acceptance Criteria

- Garments can store physical size data.
- Fit engine can compare multiple sizes.
- User receives a recommended size.
- Fit output is test-covered.

### Risks

- Many retailers do not provide enough data.
- Mitigation: mark recommendation confidence low and ask user/store/admin for missing values.

## Month 6: Production Integration, Multi-Garment Improvements, and Launch Gate

### Goal

Connect body profiles, garment parameters, fit recommendations, and visual try-on into a production-ready beta.

### Engineering Tasks

1. Integrate body profile into try-on end-to-end.
   - Try-on request includes:
     - body profile,
     - calibrated measurements,
     - selected garment sizes,
     - garment physical profiles,
     - fit assessments.

2. Improve multi-garment fit logic.
   - Top + jacket layering.
   - Pants + shoes length interaction.
   - Dress full-body handling.
   - Accessories later unless needed.

3. Improve Mobile-VTON masks/body regions.
   - Replace fixed rectangular masks where possible.
   - Use body-part regions from personalized mannequin/SAM output.
   - Preserve body silhouette better.

4. Add beta UX.
   - "Visual preview" section.
   - "Fit recommendation" section.
   - "Confidence" section.
   - "What to measure/check" section for low confidence.

5. Add monitoring.
   - Body analysis success rate.
   - Try-on render success rate.
   - Average/p95 latency.
   - Fit confidence distribution.
   - Manual correction frequency.
   - User selected recommended size or ignored it.

6. Add safety/privacy launch checklist.
   - Consent before body photo upload.
   - Delete body data control.
   - No face requirement; recommend faceless/cropped body preview.
   - Secure storage for photos/meshes.
   - Do not claim exact body or guaranteed fit.

7. Launch internal beta.
   - 20-50 users.
   - Compare recommendations to real purchased/owned clothing.
   - Collect feedback on fit accuracy and visual trust.

### Acceptance Criteria

- End-to-end beta works:
  - create body profile,
  - analyze photo,
  - calibrate height,
  - select garment size,
  - render try-on,
  - show fit recommendation.
- User can delete body data.
- Fit recommendations include confidence.
- Beta metrics are instrumented.

### Risks

- Rendered image may visually imply a fit that the measurement engine disagrees with.
- Mitigation: always display explicit fit recommendation and confidence. Do not let the image be the only source of truth.

## Fit Engine Initial Rules

These are starting rules only. They must be validated with real garments.

### Tops

Inputs:

- body chest,
- body shoulder width,
- garment chest,
- garment shoulder width,
- sleeve length,
- body/torso length,
- stretch,
- fit intent.

General ease targets:

- slim: 2-5 cm chest ease,
- regular: 5-10 cm chest ease,
- relaxed: 10-16 cm chest ease,
- oversized: 16+ cm chest ease.

Warnings:

- chest ease below 0 cm: too small,
- chest ease 0-2 cm: tight unless stretch is high,
- shoulder delta below -2 cm: tight shoulders,
- sleeve length more than 4 cm longer than arm target: sleeves long.

### Pants

Inputs:

- body waist,
- body hips,
- thigh,
- inseam,
- garment waist,
- garment hips,
- garment thigh,
- garment inseam,
- rise,
- stretch.

Warnings:

- waist below body waist: too tight unless elastic/stretch,
- hips below body hips: too tight,
- inseam delta below -3 cm: short,
- inseam delta above +4 cm: long,
- thigh below body thigh + ease: tight thigh.

### Jackets / Layers

Inputs:

- body chest,
- inner garment thickness/ease,
- jacket chest,
- shoulder,
- sleeve,
- closure type,
- fit intent.

Extra ease:

- add 2-6 cm over top depending on layer thickness.

Warnings:

- jacket may not close,
- tight over chest,
- restrictive shoulders,
- sleeves long/short.

### Dresses

Inputs:

- chest/bust,
- waist,
- hips,
- torso/body length,
- dress length,
- stretch,
- silhouette.

Warnings:

- tight bust,
- tight waist,
- tight hips,
- length too short/long.

## API Endpoint Plan

### Body Profile

```http
POST /api/body-profiles
GET /api/body-profiles/me
PATCH /api/body-profiles/:id
DELETE /api/body-profiles/:id
```

### Body Analysis

```http
POST /api/body-analysis/jobs
GET /api/body-analysis/jobs/:id
POST /api/body-analysis/jobs/:id/cancel
```

Request:

```json
{
  "sourcePhotoUrl": "https://...",
  "heightCm": 178,
  "heightSource": "apple_measure",
  "weightKg": 76,
  "gender": "male",
  "privacy": {
    "retainSourcePhoto": false,
    "retainMesh": true
  }
}
```

Response:

```json
{
  "jobId": "job_123",
  "status": "queued"
}
```

### Fit Assessment

```http
POST /api/fit/assess
```

Request:

```json
{
  "bodyProfileId": "body_123",
  "garmentId": "garment_456",
  "selectedSize": "M"
}
```

Response:

```json
{
  "overall": "good_fit",
  "confidence": "medium",
  "zones": [
    {
      "zone": "chest",
      "status": "good",
      "deltaCm": 6,
      "message": "Chest has regular ease."
    }
  ]
}
```

### Try-On Render

Extend existing Mobile-VTON payload:

```json
{
  "person_image": "data:image/png;base64,...",
  "body_profile": {
    "id": "body_123",
    "heightCm": 178,
    "measurements": {}
  },
  "garments": [
    {
      "garment_image": "https://...",
      "label": "top",
      "selected_size": "M",
      "physical_profile": {},
      "fit_assessment": {}
    }
  ],
  "pipeline_version": "fused_v3"
}
```

## Developer Implementation Order

Use this exact order to reduce risk:

1. Add body and garment data models.
2. Build manual body profile flow.
3. Build garment physical profile fields.
4. Build fit engine v1.
5. Connect fit result to current try-on UI.
6. Connect body profile to current mannequin scaling.
7. Add SAM 3D Body service prototype.
8. Add body photo upload and analysis job.
9. Extract and calibrate measurements.
10. Add manual correction UI.
11. Improve garment ingestion and size charts.
12. Improve multi-garment/layer fit logic.
13. Improve VTON masks with body-region data.
14. Run internal beta and measure fit accuracy.

## What Not To Build First

Do not start with a full physics cloth simulator. It is too slow and too complex for the first 6 months.

Do not rely only on AI image generation to decide fit. A beautiful generated image can be physically wrong.

Do not promise exact body copying. Promise calibrated digital body approximation and transparent confidence.

Do not require SAM 3D Body before building the data flow. The app can become fit-aware with manual body measurements first, then improve with SAM.

## Success Metrics

### Technical

- Body profile creation success rate: 90%+
- SAM body analysis success rate in beta: 75%+ initially, 90% target
- Try-on render success rate: 90%+
- p95 body analysis latency: under 90 seconds initially
- p95 try-on latency: under 60 seconds initially
- Fit engine unit test coverage: 80%+

### Product

- Users understand fit confidence.
- Users trust the recommendation even when the render is imperfect.
- Users can correct measurements easily.
- Beta users report fewer size mistakes.
- At least 60% of beta users say the fit notes are useful.

## Final Product Principle

The final system should combine four independent signals:

1. Body shape from 3D reconstruction.
2. Scale from Apple/iPhone height measurement.
3. Appearance from the garment photo.
4. Fit truth from garment measurements and fabric data.

Only when all four are combined can AIWardrobe move from "nice virtual try-on image" to "useful online clothing fit decision."
