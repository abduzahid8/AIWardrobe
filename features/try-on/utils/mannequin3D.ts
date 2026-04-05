export const BODY_TYPES = [
  { id: "ectomorph", label: "Slim", icon: "body-outline", desc: "Lean build, narrow frame" },
  { id: "mesomorph", label: "Athletic", icon: "fitness-outline", desc: "Muscular, medium frame" },
  { id: "endomorph", label: "Broad", icon: "shield-outline", desc: "Wider build, larger frame" },
  { id: "average", label: "Average", icon: "person-outline", desc: "Balanced proportions" },
  { id: "hourglass", label: "Hourglass", icon: "flower-outline", desc: "Defined waist, balanced" },
  { id: "pear", label: "Pear", icon: "triangle-outline", desc: "Wider hips, narrower top" },
] as const;

export type BodyTypeId = typeof BODY_TYPES[number]["id"];

/**
 * generate3Dhtml
 *
 * Generates the full HTML/JS for the Three.js WebView mannequin.
 * Supports real-time garment draping via postMessage({ type: 'drape', imageUrl, garmentType }).
 *
 * @param modelUrl  - Optional HTTPS URL to a .glb model (e.g. Supabase Storage).
 *                    When provided the real GLTF model is loaded. When null/undefined,
 *                    the procedural mannequin is used as a fallback.
 */
export function generate3Dhtml(modelUrl?: string | null) {
  const hasModel = !!modelUrl;

  return `
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { overflow: hidden; background: transparent; touch-action: none; }
  canvas { display: block; width: 100vw; height: 100vh; }
  #loader {
    position: fixed; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: rgba(255,255,255,0.92);
    font-family: -apple-system, sans-serif;
    z-index: 999;
    transition: opacity 0.4s ease;
  }
  #loader.hidden { opacity: 0; pointer-events: none; }
  .spinner {
    width: 44px; height: 44px;
    border: 3px solid rgba(0,85,255,0.15);
    border-top-color: #0055FF;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loader-text {
    margin-top: 14px;
    font-size: 13px; font-weight: 600;
    color: #0055FF;
    letter-spacing: 0.3px;
  }
  .loader-sub {
    margin-top: 4px;
    font-size: 11px;
    color: rgba(0,0,0,0.45);
  }
</style>
</head>
<body>
${hasModel ? `
<div id="loader">
  <div class="spinner"></div>
  <div class="loader-text">Loading mannequin…</div>
  <div class="loader-sub">Preparing your 3D model</div>
</div>
` : ''}

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
${hasModel ? `
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/DRACOLoader.js"></script>
` : ''}
<script>
  // ─── Scene ──────────────────────────────────────────
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 5.5);

  var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0xFFFFFF, 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.physicallyCorrectLights = true;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  document.body.appendChild(renderer.domElement);

  // ─── Lighting ───────────────────────────────────────
  var ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  var keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
  keyLight.position.set(3, 8, 5);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 1024;
  keyLight.shadow.mapSize.height = 1024;
  scene.add(keyLight);

  var fillLight = new THREE.DirectionalLight(0xccd5e0, 0.5);
  fillLight.position.set(-4, 4, 3);
  scene.add(fillLight);

  var rimLight = new THREE.DirectionalLight(0x99aacc, 0.4);
  rimLight.position.set(0, 2, -4);
  scene.add(rimLight);

  var bottomFill = new THREE.PointLight(0xddeeff, 0.25, 10);
  bottomFill.position.set(0, -2, 3);
  scene.add(bottomFill);

  // ─── Ground plane ──────────────────────────────────
  var groundGeo = new THREE.PlaneGeometry(8, 8);
  var groundMat = new THREE.ShadowMaterial({ opacity: 0.08 });
  var ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -1.5;
  ground.receiveShadow = true;
  scene.add(ground);

  // ─── Touch and rotation state ──────────────────────
  var userRotationY = 0;
  var autoRotate = true;
  var touchStartX = 0;
  var touchStartRotation = 0;
  var lastTouchTime = 0;

  document.addEventListener('touchstart', function(e) {
    if (e.touches.length > 1) return;
    autoRotate = false;
    touchStartX = e.touches[0].clientX;
    touchStartRotation = userRotationY;
    lastTouchTime = Date.now();
  }, { passive: false });

  document.addEventListener('touchmove', function(e) {
    if (e.touches.length > 1) return;
    var dx = e.touches[0].clientX - touchStartX;
    userRotationY = touchStartRotation + (dx / window.innerWidth) * Math.PI * 2;
    lastTouchTime = Date.now();
  }, { passive: false });

  document.addEventListener('touchend', function() {
    setTimeout(function() {
      if (Date.now() - lastTouchTime >= 2900) { autoRotate = true; }
    }, 3000);
  });

  // ─── Main model group ──────────────────────────────
  var modelGroup = new THREE.Group();
  scene.add(modelGroup);

  // ─── Mannequin material (fiberglass look) ──────────
  var mannequinMat = new THREE.MeshPhysicalMaterial({
    color: 0xF0F4F8,
    metalness: 0.08,
    roughness: 0.25,
    clearcoat: 0.9,
    clearcoatRoughness: 0.15,
  });

  // ─── Current proportions for camera tracking ───────
  var currentCamTargetY = 0;
  var modelLoaded = false;

  // ─── Garment draping state ─────────────────────────
  var clothMesh = null;
  var currentProportions = null;   // set by applyProportions

  /**
   * drapeGarment — places a textured plane on the mannequin,
   * sized and positioned based on saved proportions + garmentType.
   * Works for both the procedural mannequin and the GLB model.
   */
  function drapeGarment(imageUrl, garmentType) {
    // Remove any previous cloth
    if (clothMesh) {
      modelGroup.remove(clothMesh);
      if (clothMesh.material.map) clothMesh.material.map.dispose();
      clothMesh.material.dispose();
      clothMesh.geometry.dispose();
      clothMesh = null;
    }

    if (!imageUrl) return;

    // Determine placement geometry from saved proportions or sensible defaults
    var p = currentProportions;

    var planeW, planeH, planeY, planeZ;

    if (p) {
      if (garmentType === 'lower_body') {
        // Covers hip-to-ankle region
        planeW = p.hipR * 2.8;
        planeH = p.upperLegH + p.lowerLegH + 0.05;
        planeY = p.ankleY + (p.upperLegH + p.lowerLegH) * 0.5;
        planeZ = 0.18;
      } else if (garmentType === 'dresses') {
        // Full body: shoulder to ankle
        planeW = p.shoulderHalfW * 2.6;
        planeH = p.shoulderY - p.ankleY + 0.05;
        planeY = p.ankleY + (p.shoulderY - p.ankleY) * 0.5;
        planeZ = 0.20;
      } else {
        // upper_body default — torso + arms zone
        planeW = p.shoulderHalfW * 2.8;
        planeH = p.torsoH + 0.08;
        planeY = p.hipY + p.torsoH * 0.5;
        planeZ = 0.20;
      }
    } else {
      // GLB model / no proportions yet — use scene-relative defaults
      if (garmentType === 'lower_body') {
        planeW = 0.70; planeH = 1.00; planeY = -0.40; planeZ = 0.22;
      } else if (garmentType === 'dresses') {
        planeW = 0.80; planeH = 1.50; planeY = 0.10; planeZ = 0.22;
      } else {
        planeW = 0.80; planeH = 0.70; planeY = 0.55; planeZ = 0.22;
      }
    }

    // Load texture asynchronously
    var loader = new THREE.TextureLoader();
    loader.crossOrigin = 'anonymous';
    loader.load(imageUrl, function(tex) {
      tex.encoding = THREE.sRGBEncoding;
      tex.minFilter = THREE.LinearMipmapLinearFilter;
      tex.generateMipmaps = true;
      tex.anisotropy = renderer.capabilities.getMaxAnisotropy();

      // Preserve the image's natural aspect ratio
      var imgW = tex.image.naturalWidth  || tex.image.width  || 1;
      var imgH = tex.image.naturalHeight || tex.image.height || 1;
      var aspect = imgW / imgH;

      var finalW = planeW;
      var finalH = planeW / aspect;
      // Clamp height to the intended region
      if (finalH > planeH * 1.4) { finalH = planeH * 1.4; finalW = finalH * aspect; }

      var geo  = new THREE.PlaneGeometry(finalW, finalH);
      var mat  = new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        alphaTest: 0.05,
        side: THREE.FrontSide,
        depthWrite: false,
      });

      clothMesh = new THREE.Mesh(geo, mat);
      clothMesh.position.set(0, planeY, planeZ);
      // Slight renderOrder so it always draws on top of mannequin geometry
      clothMesh.renderOrder = 1;
      modelGroup.add(clothMesh);
    }, undefined, function(err) {
      console.warn('Cloth texture load failed:', err);
    });
  }

  // =======================================================================
  // GLB MODEL LOADER (App Store path — loads from Supabase CDN over HTTPS)
  // =======================================================================
  ${hasModel ? `
  (function loadGLTFModel() {
    try {
      var dracoLoader = new THREE.DRACOLoader();
      dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');

      var loader = new THREE.GLTFLoader();
      loader.setDRACOLoader(dracoLoader);

      loader.load(
        '${modelUrl}',
        function onLoad(gltf) {
          var model = gltf.scene;

          // Auto-scale to fit viewport
          var box = new THREE.Box3().setFromObject(model);
          var size = new THREE.Vector3();
          box.getSize(size);
          var maxDim = Math.max(size.x, size.y, size.z);
          var scale = 2.8 / maxDim;
          model.scale.setScalar(scale);

          // Center the model
          var center = new THREE.Vector3();
          box.getCenter(center);
          model.position.x = -center.x * scale;
          model.position.y = -center.y * scale;
          model.position.z = -center.z * scale;

          // Enable shadows + enhance materials
          model.traverse(function(child) {
            if (child.isMesh) {
              child.castShadow = true;
              child.receiveShadow = true;
              if (child.material) {
                if (Array.isArray(child.material)) {
                  child.material.forEach(function(mat) {
                    if (mat.isMeshStandardMaterial || mat.isMeshPhysicalMaterial) {
                      mat.clearcoat = 0.3;
                      mat.clearcoatRoughness = 0.4;
                    }
                  });
                } else if (child.material.isMeshStandardMaterial || child.material.isMeshPhysicalMaterial) {
                  child.material.clearcoat = 0.3;
                  child.material.clearcoatRoughness = 0.4;
                }
              }
            }
          });

          modelGroup.add(model);

          // Compute camera aim point
          var newBox = new THREE.Box3().setFromObject(model);
          var newSize = new THREE.Vector3();
          newBox.getSize(newSize);
          currentCamTargetY = newBox.min.y + newSize.y * 0.45;

          // Build proportions approximation from GLB bounding box
          // so drapeGarment() has sensible values even for the real GLB model
          var totalH = newSize.y;
          currentProportions = {
            hipR:          newSize.x * 0.18,
            shoulderHalfW: newSize.x * 0.22,
            torsoH:        totalH  * 0.30,
            upperLegH:     totalH  * 0.22,
            lowerLegH:     totalH  * 0.21,
            hipY:          newBox.min.y + totalH * 0.24,
            shoulderY:     newBox.min.y + totalH * 0.78,
            ankleY:        newBox.min.y + totalH * 0.04,
          };

          modelLoaded = true;

          if (window.ReactNativeWebView) {
            window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'model_loaded' }));
          }

          var loaderEl = document.getElementById('loader');
          if (loaderEl) {
            loaderEl.classList.add('hidden');
            setTimeout(function() { loaderEl.remove(); }, 500);
          }
        },
        function onProgress(xhr) {
          var loaderEl = document.getElementById('loader');
          if (loaderEl && xhr.total > 0) {
            var pct = Math.round((xhr.loaded / xhr.total) * 100);
            var subEl = loaderEl.querySelector('.loader-sub');
            if (subEl) subEl.textContent = pct + '% loaded';
          }
        },
        function onError(err) {
          console.warn('GLB load failed, using procedural fallback:', err);
          buildProceduralMannequin();
          var loaderEl = document.getElementById('loader');
          if (loaderEl) { loaderEl.remove(); }
          if (window.ReactNativeWebView) {
            window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'model_fallback' }));
          }
        }
      );
    } catch(e) {
      buildProceduralMannequin();
      var loaderEl = document.getElementById('loader');
      if (loaderEl) { loaderEl.remove(); }
    }
  })();
  ` : `
  // No model URL — use procedural mannequin directly
  buildProceduralMannequin();
  `}

  // =======================================================================
  // PROCEDURAL MANNEQUIN (fallback / default when no GLB URL is provided)
  // =======================================================================

  var proceduralGroup = null;
  var proceduralParts = {};

  function buildProceduralMannequin() {
    proceduralGroup = new THREE.Group();
    modelGroup.add(proceduralGroup);

    function createPart() {
      var mesh = new THREE.Mesh(new THREE.BufferGeometry(), mannequinMat);
      mesh.castShadow = true;
      proceduralGroup.add(mesh);
      return mesh;
    }

    proceduralParts.head         = createPart();
    proceduralParts.neck         = createPart();
    proceduralParts.torso        = createPart();
    proceduralParts.leftUpperArm = createPart();
    proceduralParts.rightUpperArm= createPart();
    proceduralParts.leftForearm  = createPart();
    proceduralParts.rightForearm = createPart();
    proceduralParts.leftHand     = createPart();
    proceduralParts.rightHand    = createPart();
    proceduralParts.leftUpperLeg = createPart();
    proceduralParts.rightUpperLeg= createPart();
    proceduralParts.leftLowerLeg = createPart();
    proceduralParts.rightLowerLeg= createPart();
    proceduralParts.leftFoot     = createPart();
    proceduralParts.rightFoot    = createPart();

    currentProportions = computeProportions(175, 70, 'average');
    applyProportions(currentProportions);
    modelLoaded = true;

    if (window.ReactNativeWebView) {
      window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'model_fallback' }));
    }
  }

  function computeProportions(heightCm, weightKg, bodyType) {
    var h = heightCm / 100;
    var totalHeight = h * 1.1;
    var bmi = weightKg / (h * h);
    var bmiNorm = Math.max(0, Math.min(1, (bmi - 15) / 25));

    var types = {
      ectomorph:  { shoulder: 0.85, chest: 0.82, waist: 0.78, hip: 0.82, armThick: 0.78, legThick: 0.80 },
      mesomorph:  { shoulder: 1.12, chest: 1.10, waist: 0.88, hip: 0.95, armThick: 1.10, legThick: 1.08 },
      endomorph:  { shoulder: 1.0,  chest: 1.08, waist: 1.15, hip: 1.12, armThick: 1.05, legThick: 1.10 },
      average:    { shoulder: 1.0,  chest: 1.0,  waist: 1.0,  hip: 1.0,  armThick: 1.0,  legThick: 1.0  },
      hourglass:  { shoulder: 1.05, chest: 1.08, waist: 0.82, hip: 1.10, armThick: 0.95, legThick: 1.02 },
      pear:       { shoulder: 0.90, chest: 0.92, waist: 0.95, hip: 1.18, armThick: 0.88, legThick: 1.08 },
    };
    var bt = types[bodyType] || types.average;

    var shoulderHalfW = (0.22 * (heightCm / 175)) * bt.shoulder * (1 + bmiNorm * 0.12);
    var chestR  = (0.14 * (heightCm / 175)) * bt.chest  * (1 + bmiNorm * 0.25);
    var waistR  = (0.12 * (heightCm / 175)) * bt.waist  * (1 + bmiNorm * 0.35);
    var hipR    = (0.13 * (heightCm / 175)) * bt.hip    * (1 + bmiNorm * 0.28);

    var headH      = totalHeight * 0.12;
    var neckH      = totalHeight * 0.04;
    var torsoH     = totalHeight * 0.30;
    var upperArmH  = totalHeight * 0.16;
    var forearmH   = totalHeight * 0.14;
    var upperLegH  = totalHeight * 0.22;
    var lowerLegH  = totalHeight * 0.21;

    var armR      = 0.035 * bt.armThick * (1 + bmiNorm * 0.3);
    var forearmR  = 0.030 * bt.armThick * (1 + bmiNorm * 0.25);
    var upperLegR = 0.065 * bt.legThick * (1 + bmiNorm * 0.3);
    var lowerLegR = 0.045 * bt.legThick * (1 + bmiNorm * 0.2);

    var footY    = -totalHeight * 0.48;
    var ankleY   = footY + 0.03;
    var kneeY    = ankleY + lowerLegH;
    var hipY     = kneeY + upperLegH;
    var waistY   = hipY + torsoH * 0.3;
    var chestY   = hipY + torsoH * 0.7;
    var shoulderY= hipY + torsoH;
    var neckY    = shoulderY + neckH * 0.5;
    var headY    = shoulderY + neckH + headH * 0.5;
    var legSep   = hipR * 0.55;

    return {
      totalHeight, headH, neckH, torsoH, upperArmH, forearmH, upperLegH, lowerLegH,
      shoulderHalfW, chestR, waistR, hipR,
      armR, forearmR, upperLegR, lowerLegR,
      headY, neckY, shoulderY, chestY, waistY, hipY, kneeY, ankleY, footY, legSep,
    };
  }

  function updateGeo(mesh, geo) {
    if (mesh.geometry) mesh.geometry.dispose();
    mesh.geometry = geo;
  }

  function applyProportions(p) {
    var pp = proceduralParts;

    updateGeo(pp.head, new THREE.SphereGeometry(p.headH * 0.45, 32, 24));
    pp.head.position.y = p.headY;
    pp.head.scale.set(1, 1.25, 1.1);

    var neckCurve = new THREE.SplineCurve([
      new THREE.Vector2(p.waistR * 0.35, 0),
      new THREE.Vector2(p.waistR * 0.3, p.neckH * 0.5),
      new THREE.Vector2(p.headH * 0.2, p.neckH)
    ]);
    updateGeo(pp.neck, new THREE.LatheGeometry(neckCurve.getPoints(12), 24));
    pp.neck.position.y = p.shoulderY;

    var torsoCurve = new THREE.SplineCurve([
      new THREE.Vector2(p.hipR, 0),
      new THREE.Vector2(p.waistR, p.torsoH * 0.3),
      new THREE.Vector2(p.chestR, p.torsoH * 0.7),
      new THREE.Vector2(p.shoulderHalfW * 0.85, p.torsoH)
    ]);
    updateGeo(pp.torso, new THREE.LatheGeometry(torsoCurve.getPoints(16), 32));
    pp.torso.position.y = p.hipY;
    pp.torso.scale.set(1, 1, 0.65);

    var armRestAngle = 0.08;
    var elbowY = p.shoulderY - p.upperArmH;
    var wristY = elbowY - p.forearmH;

    var upperArmCurve = new THREE.SplineCurve([
      new THREE.Vector2(p.armR * 0.75, 0),
      new THREE.Vector2(p.armR, p.upperArmH * 0.5),
      new THREE.Vector2(p.armR * 0.9, p.upperArmH)
    ]);
    var upperArmGeo = new THREE.LatheGeometry(upperArmCurve.getPoints(12), 24);
    updateGeo(pp.leftUpperArm, upperArmGeo);
    pp.leftUpperArm.position.set(-p.shoulderHalfW - p.armR * 0.5, elbowY, 0);
    pp.leftUpperArm.rotation.z = armRestAngle;
    updateGeo(pp.rightUpperArm, upperArmGeo);
    pp.rightUpperArm.position.set(p.shoulderHalfW + p.armR * 0.5, elbowY, 0);
    pp.rightUpperArm.rotation.z = -armRestAngle;

    var forearmCurve = new THREE.SplineCurve([
      new THREE.Vector2(p.forearmR * 0.5, 0),
      new THREE.Vector2(p.forearmR * 0.9, p.forearmH * 0.7),
      new THREE.Vector2(p.armR * 0.75, p.forearmH)
    ]);
    var forearmGeo = new THREE.LatheGeometry(forearmCurve.getPoints(12), 24);
    updateGeo(pp.leftForearm, forearmGeo);
    pp.leftForearm.position.set(-p.shoulderHalfW - p.armR * 0.5 - p.upperArmH * Math.sin(armRestAngle), wristY, 0);
    pp.leftForearm.rotation.z = armRestAngle * 0.5;
    updateGeo(pp.rightForearm, forearmGeo);
    pp.rightForearm.position.set(p.shoulderHalfW + p.armR * 0.5 + p.upperArmH * Math.sin(armRestAngle), wristY, 0);
    pp.rightForearm.rotation.z = -armRestAngle * 0.5;

    updateGeo(pp.leftHand, new THREE.SphereGeometry(p.forearmR * 0.55, 16, 16));
    pp.leftHand.position.set(-p.shoulderHalfW - p.armR * 0.6 - p.upperArmH * Math.sin(armRestAngle), wristY - p.forearmR, 0);
    pp.leftHand.scale.set(1, 1.4, 0.4);
    updateGeo(pp.rightHand, new THREE.SphereGeometry(p.forearmR * 0.55, 16, 16));
    pp.rightHand.position.set(p.shoulderHalfW + p.armR * 0.6 + p.upperArmH * Math.sin(armRestAngle), wristY - p.forearmR, 0);
    pp.rightHand.scale.set(1, 1.4, 0.4);

    var upperLegCurve = new THREE.SplineCurve([
      new THREE.Vector2(p.lowerLegR * 0.9, 0),
      new THREE.Vector2(p.upperLegR, p.upperLegH * 0.6),
      new THREE.Vector2(p.upperLegR * 1.05, p.upperLegH)
    ]);
    var upperLegGeo = new THREE.LatheGeometry(upperLegCurve.getPoints(12), 24);
    updateGeo(pp.leftUpperLeg, upperLegGeo);
    pp.leftUpperLeg.position.set(-p.legSep, p.kneeY, 0);
    updateGeo(pp.rightUpperLeg, upperLegGeo);
    pp.rightUpperLeg.position.set(p.legSep, p.kneeY, 0);

    var lowerLegCurve = new THREE.SplineCurve([
      new THREE.Vector2(p.lowerLegR * 0.6, 0),
      new THREE.Vector2(p.lowerLegR, p.lowerLegH * 0.6),
      new THREE.Vector2(p.lowerLegR * 0.9, p.lowerLegH)
    ]);
    var lowerLegGeo = new THREE.LatheGeometry(lowerLegCurve.getPoints(12), 24);
    updateGeo(pp.leftLowerLeg, lowerLegGeo);
    pp.leftLowerLeg.position.set(-p.legSep, p.ankleY, 0);
    updateGeo(pp.rightLowerLeg, lowerLegGeo);
    pp.rightLowerLeg.position.set(p.legSep, p.ankleY, 0);

    var footGeo = new THREE.BoxGeometry(p.lowerLegR * 1.5, p.lowerLegR * 0.8, p.lowerLegR * 3.5);
    updateGeo(pp.leftFoot, footGeo);
    pp.leftFoot.position.set(-p.legSep, p.footY + p.lowerLegR * 0.4, p.lowerLegR);
    updateGeo(pp.rightFoot, footGeo);
    pp.rightFoot.position.set(p.legSep, p.footY + p.lowerLegR * 0.4, p.lowerLegR);

    // Aim camera at the vertical center of the mannequin (between feet and head)
    var totalModelHeight = p.headY - p.footY;
    currentCamTargetY = p.footY + totalModelHeight * 0.5;
  }

  // ─── Message handler ──────────────────────────────
  window.addEventListener('message', function(event) {
    try {
      var data = JSON.parse(event.data);

      if (data.type === 'update') {
        // Only applies to procedural mannequin
        if (proceduralGroup && proceduralParts.head) {
          var h = Math.max(100, Math.min(230, data.heightCm || 175));
          var w = Math.max(30,  Math.min(200, data.weightKg || 70));
          var bt = data.bodyType || 'average';
          currentProportions = computeProportions(h, w, bt);
          applyProportions(currentProportions);
          // Re-drape cloth if one is already on the mannequin
          if (clothMesh && clothMesh._lastDrapeData) {
            drapeGarment(clothMesh._lastDrapeData.imageUrl, clothMesh._lastDrapeData.garmentType);
          }
        }

      } else if (data.type === 'drape') {
        // Place garment image directly on the mannequin
        drapeGarment(data.imageUrl, data.garmentType || 'upper_body');
        // Store for re-drape on proportion change
        if (clothMesh) {
          clothMesh._lastDrapeData = { imageUrl: data.imageUrl, garmentType: data.garmentType };
        }
        // Notify React Native
        if (window.ReactNativeWebView) {
          window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'drape_applied' }));
        }

      } else if (data.type === 'remove_cloth') {
        drapeGarment(null, null);

      } else if (data.type === 'capture') {
        renderer.render(scene, camera);
        var base64Image = renderer.domElement.toDataURL('image/jpeg', 0.92);
        if (window.ReactNativeWebView) {
          window.ReactNativeWebView.postMessage(JSON.stringify({
            type: 'snapshot_result',
            base64: base64Image,
          }));
        }
      }
    } catch(e) {}
  });

  // ─── Animation loop ───────────────────────────────
  var time = 0;
  function animate() {
    requestAnimationFrame(animate);
    time += 0.016;

    if (autoRotate) { userRotationY += 0.003; }
    modelGroup.rotation.y = userRotationY;

    // Gentle breathing float — only when model is loaded
    if (modelLoaded) {
      modelGroup.position.y = Math.sin(time * 1.2) * 0.005;
    }

    // Smooth camera pan toward model vertical center
    camera.position.y += (currentCamTargetY - camera.position.y) * 0.05;
    camera.lookAt(0, currentCamTargetY, 0);

    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', function() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
<\/script>
</body>
</html>
`;
}
