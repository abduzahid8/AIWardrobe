export const BODY_TYPES = [
  { id: "ectomorph", label: "Slim", icon: "body-outline", desc: "Lean build" },
  { id: "average", label: "Average", icon: "person-outline", desc: "Balanced proportions" },
  { id: "mesomorph", label: "Muscular", icon: "barbell-outline", desc: "Muscular frame" },
  { id: "endomorph", label: "Fat", icon: "pizza-outline", desc: "Heavy set, wider build" },
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
export function generate3Dhtml(
  modelUrl?: string | null,
  initialH: number = 175,
  initialW: number = 70,
  initialBT: string = 'average',
) {
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

<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
${hasModel ? `
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/DRACOLoader.js"></script>
` : ''}
<script>
  // ─── Scene ──────────────────────────────────────────
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 9.0, 5.5);  // aimed at upper body, closer for bigger mannequin

  // ─── Target scale state — baked in from React Native at render time ────
  // _mDirty = true so the animation loop applies scale as soon as model loads
  window._mH     = ${initialH};
  window._mW     = ${initialW};
  window._mBT    = '${initialBT}';
  window._mDirty = true;

  var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0xFFFFFF, 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.physicallyCorrectLights = true;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.toneMappingExposure = 1.0;
  document.body.appendChild(renderer.domElement);

  // ─── Lighting ───────────────────────────────────────
  var ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
  scene.add(ambientLight);

  var keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
  keyLight.position.set(4, 10, 6);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 2048;
  keyLight.shadow.mapSize.height = 2048;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 25;
  keyLight.shadow.bias = -0.0001;
  scene.add(keyLight);

  var fillLight = new THREE.DirectionalLight(0xddeeff, 0.65);
  fillLight.position.set(-6, 3, 4);
  scene.add(fillLight);

  var rimLight = new THREE.DirectionalLight(0xccddff, 0.5);
  rimLight.position.set(0, 4, -6);
  scene.add(rimLight);

  var bottomFill = new THREE.PointLight(0xabccee, 0.4, 12);
  bottomFill.position.set(0, -3, 4);
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
  var autoRotate = false;
  var touchStartX = 0;
  var touchStartRotation = 0;
  var lastTouchTime = 0;

  document.addEventListener('touchstart', function(e) {
    if (e.touches.length > 1) return;
    e.preventDefault();
    autoRotate = false;
    touchStartX = e.touches[0].clientX;
    touchStartRotation = userRotationY;
    lastTouchTime = Date.now();
  }, { passive: false });

  document.addEventListener('touchmove', function(e) {
    if (e.touches.length > 1) return;
    e.preventDefault();
    var dx = e.touches[0].clientX - touchStartX;
    userRotationY = touchStartRotation + (dx / window.innerWidth) * Math.PI * 2;
    lastTouchTime = Date.now();
  }, { passive: false });

  document.addEventListener('touchend', function() {
    // autoRotate stays false — mannequin only moves when user drags it
  });

  // ─── Main model group ──────────────────────────────
  var modelGroup = new THREE.Group();
  scene.add(modelGroup);

  // ─── Mannequin material (fiberglass look) ──────────
  var mannequinMat = new THREE.MeshStandardMaterial({
    color: 0x808080,
    metalness: 0.12,
    roughness: 0.28,
    envMapIntensity: 1.0,
  });

  // ─── Current proportions for camera tracking ───────
  var currentCamTargetY = 0;
  var modelGroupOffsetY  = 0;   // shifts modelGroup so visual centre stays at world y=5.5
  var targetCamZ = 8.0;         // dynamic camera Z — adjusted per height
  var glbBaseMinY  = 0;         // set when GLB loads
  var glbBaseSizeY = 10;        // set when GLB loads
  var modelLoaded = false;
  var modelFeetWorldY = 0.3;    // world-space Y of model's feet — ground follows this

  // ─── Garment draping state ─────────────────────────
  var clothMeshes = {};             // keyed by garmentType for multi-layer outfits
  var pendingDrapes = [];           // queue of { imageUrl, garmentType } while model is loading
  var currentProportions = null;   // set by applyProportions

  /**
   * drapeGarment — dresses the 3D mannequin in the selected garment.
   *
   * GLB path:  Clones the body mesh, inflates it slightly along normals,
   *            projects the garment texture from the front, and clips to
   *            the garment region.  The cloth follows exact body contours
   *            and deforms with the skeleton (SkinnedMesh).
   *
   * Procedural path (fallback):  Half-cylinder shells around the body.
   */
  var clothGeneration = {}; // prevents stale texture-load callbacks

  // ─── Image loading helper — works for both https:// and data: URIs ──
  function loadImageTexture(imageUrl, onSuccess, onError) {
    var img = new Image();
    img.onload = function() {
      try {
        // Blit into canvas so Three.js CanvasTexture works universally
        var canvas = document.createElement('canvas');
        canvas.width  = img.naturalWidth  || img.width  || 512;
        canvas.height = img.naturalHeight || img.height || 512;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        var tex = new THREE.CanvasTexture(canvas);
        tex.encoding = THREE.sRGBEncoding;
        tex.minFilter = THREE.LinearFilter;
        tex.generateMipmaps = false;
        tex.needsUpdate = true;
        onSuccess(tex, img);
      } catch(e) { onError(e); }
    };
    img.onerror = function(e) { onError(e); };
    // crossOrigin must be set BEFORE src for remote images; harmless for data: URIs
    if (imageUrl && imageUrl.indexOf('https://') === 0) {
      img.crossOrigin = 'anonymous';
    }
    img.src = imageUrl;
  }

  function drapeGarment(imageUrl, garmentType) {
    if (!modelLoaded) {
      if (imageUrl) {
        // Remove any existing queued drape for this garmentType, then append
        pendingDrapes = pendingDrapes.filter(function(d) { return d.garmentType !== (garmentType || 'upper_body'); });
        pendingDrapes.push({ imageUrl: imageUrl, garmentType: garmentType });
      } else {
        // null imageUrl = remove — clear any queued drape for this type
        pendingDrapes = pendingDrapes.filter(function(d) { return d.garmentType !== (garmentType || 'upper_body'); });
      }
      return;
    }
    var gType = garmentType || 'upper_body';
    if (!clothGeneration[gType]) clothGeneration[gType] = 0;
    clothGeneration[gType]++;
    var thisGen = clothGeneration[gType];

    // ── Clean up previous cloth of this garment type ──────────────────
    var existingCloth = clothMeshes[gType];
    if (existingCloth) {
      if (existingCloth.parent) existingCloth.parent.remove(existingCloth);
      existingCloth.traverse(function(child) {
        if (child.isMesh) {
          if (child.material && child.material.map) child.material.map.dispose();
          if (child.material) child.material.dispose();
          if (child.geometry) child.geometry.dispose();
        }
      });
      clothMeshes[gType] = null;
    }

    if (!imageUrl) return;

    // ── Find the main body mesh in the GLB model ───────────────────────
    var glbWrapper = modelGroup.getObjectByName('glb_model');
    var bodyMesh = null;
    if (glbWrapper) {
      var maxVerts = 0;
      glbWrapper.traverse(function(child) {
        if ((child.isSkinnedMesh || child.isMesh) && child.visible) {
          var vc = child.geometry.attributes.position
                 ? child.geometry.attributes.position.count : 0;
          if (vc > maxVerts) { maxVerts = vc; bodyMesh = child; }
        }
      });
    }

    // ════════════════════════════════════════════════════════════════════
    if (bodyMesh) {
    // ── GLB PATH — body-conforming garment mesh ────────────────────────
    // ════════════════════════════════════════════════════════════════════
      var srcGeo = bodyMesh.geometry;
      srcGeo.computeBoundingBox();
      var gBox = srcGeo.boundingBox;
      var gH   = gBox.max.y - gBox.min.y;
      var gMinY = gBox.min.y;
      var gW   = gBox.max.x - gBox.min.x;
      var gMinX = gBox.min.x;

      // Garment region in LOCAL geometry space (bind-pose coordinates)
      var localBot, localTop;
      if (gType === 'lower_body') {
        localBot = gMinY + gH * 0.02;
        localTop = gMinY + gH * 0.52;
      } else if (gType === 'dresses') {
        localBot = gMinY + gH * 0.02;
        localTop = gMinY + gH * 0.80;
      } else if (gType === 'shoes') {
        localBot = gMinY;
        localTop = gMinY + gH * 0.10;
      } else {
        // upper_body
        localBot = gMinY + gH * 0.35;
        localTop = gMinY + gH * 0.82;
      }

      // Clone geometry + inflate along normals (visible shell on top of body)
      var clothGeo = srcGeo.clone();
      var pos  = clothGeo.getAttribute('position');
      var norm = clothGeo.getAttribute('normal');
      var inflate = gH * 0.02;  // 2% of body height — enough to clear z-fighting

      // Store original Y for face culling, then inflate + compute UVs
      var origY = new Float32Array(pos.count);
      var uvData = new Float32Array(pos.count * 2);
      for (var vi = 0; vi < pos.count; vi++) {
        var px = pos.getX(vi), py = pos.getY(vi), pz = pos.getZ(vi);
        var nx = norm.getX(vi), ny = norm.getY(vi), nz = norm.getZ(vi);
        origY[vi] = py;

        // Push outward along surface normal
        pos.setXYZ(vi, px + nx * inflate, py + ny * inflate, pz + nz * inflate);

        // u = left→right across body,  v = top→bottom of garment region
        var u = (px - gMinX) / gW;
        var v = 1.0 - (py - localBot) / (localTop - localBot);
        uvData[vi * 2]     = u;
        uvData[vi * 2 + 1] = v;
      }
      pos.needsUpdate = true;
      clothGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvData, 2));

      // Remove faces entirely outside the garment Y region
      // (replaces clipping planes — immune to transform-chain rotations)
      var keptFaces = 0;
      if (clothGeo.index) {
        var srcIdx = clothGeo.index;
        var kept = [];
        for (var fi = 0; fi < srcIdx.count; fi += 3) {
          var ia = srcIdx.getX(fi), ib = srcIdx.getX(fi+1), ic = srcIdx.getX(fi+2);
          var ya = origY[ia], yb = origY[ib], yc = origY[ic];
          if ((ya >= localBot && ya <= localTop) ||
              (yb >= localBot && yb <= localTop) ||
              (yc >= localBot && yc <= localTop)) {
            kept.push(ia, ib, ic);
          }
        }
        clothGeo.setIndex(kept);
        keptFaces = kept.length / 3;
      }

      var placeholderMat = new THREE.MeshStandardMaterial({
        color: 0xaaaaaa, side: THREE.DoubleSide, roughness: 0.6,
      });

      var clothObj;
      if (bodyMesh.isSkinnedMesh) {
        clothObj = new THREE.SkinnedMesh(clothGeo, placeholderMat);
        clothObj.bind(bodyMesh.skeleton, bodyMesh.bindMatrix);
      } else {
        clothObj = new THREE.Mesh(clothGeo, placeholderMat);
      }

      clothObj.position.copy(bodyMesh.position);
      clothObj.quaternion.copy(bodyMesh.quaternion);
      clothObj.scale.copy(bodyMesh.scale);
      clothObj.frustumCulled = false;
      clothObj.renderOrder = 10;
      bodyMesh.parent.add(clothObj);

      clothMeshes[gType] = clothObj;
      clothObj._lastDrapeData = { imageUrl: imageUrl, garmentType: gType };

      var capturedClothObj = clothObj;
      var capturedGen = thisGen;
      loadImageTexture(imageUrl, function(tex) {
        if (capturedGen !== clothGeneration[gType]) return;
        var texMat = new THREE.MeshStandardMaterial({
          map: tex, side: THREE.DoubleSide, roughness: 0.6, metalness: 0.0,
        });
        capturedClothObj.material = texMat;
        placeholderMat.dispose();
        if (window.ReactNativeWebView) {
          window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'drape_applied' }));
        }
      }, function() {
        if (capturedGen !== clothGeneration[gType]) return;
        capturedClothObj.material = new THREE.MeshStandardMaterial({
          color: 0x888888, side: THREE.DoubleSide, roughness: 0.7,
        });
        placeholderMat.dispose();
        if (window.ReactNativeWebView) {
          window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'drape_applied' }));
        }
      });

    // ════════════════════════════════════════════════════════════════════
    } else {
    // ── PROCEDURAL FALLBACK — half-cylinder shells ─────────────────────
    // ════════════════════════════════════════════════════════════════════
      var p = currentProportions || {};
      var bSize = new THREE.Vector3();
      bSize.set((p.shoulderHalfW || 0.22) * 4,
                (p.shoulderY || 1.1) - (p.ankleY || -1.7),
                (p.shoulderHalfW || 0.22) * 2);
      var bCenter = new THREE.Vector3();
      bCenter.set(0, ((p.shoulderY || 1.1) + (p.ankleY || -1.7)) / 2, 0);

      var bodyW = bSize.x, bodyH = bSize.y;
      var minY = bCenter.y - bodyH / 2;
      var regionBot, regionTop, radTop, radBot;
      if (gType === 'lower_body') {
        regionBot = minY + bodyH * 0.02;
        regionTop = minY + bodyH * 0.50;
        radTop = bodyW * 0.30; radBot = bodyW * 0.18;
      } else if (gType === 'dresses') {
        regionBot = minY + bodyH * 0.02;
        regionTop = minY + bodyH * 0.78;
        radTop = bodyW * 0.30; radBot = bodyW * 0.28;
      } else if (gType === 'shoes') {
        regionBot = minY;
        regionTop = minY + bodyH * 0.12;
        radTop = bodyW * 0.14; radBot = bodyW * 0.12;
      } else {
        regionBot = minY + bodyH * 0.38;
        regionTop = minY + bodyH * 0.80;
        radTop = bodyW * 0.32; radBot = bodyW * 0.28;
      }

      var clothH = regionTop - regionBot;
      var midY   = (regionBot + regionTop) / 2;
      var clothGroup = new THREE.Group();

      loadImageTexture(imageUrl, function(tex, srcImg) {
        if (thisGen !== clothGeneration[gType]) return;

        // Sample dominant color from the image for the back panel
        var fabricHex = 0x909090;
        try {
          var _c = document.createElement('canvas');
          _c.width = 8; _c.height = 8;
          var _ctx = _c.getContext('2d');
          _ctx.drawImage(srcImg, 0, 0, 8, 8);
          var _d = _ctx.getImageData(0, 0, 8, 8).data;
          var _r = 0, _g = 0, _b = 0, _n = 0;
          for (var _i = 0; _i < _d.length; _i += 4) {
            if (_d[_i+3] > 128) { _r += _d[_i]; _g += _d[_i+1]; _b += _d[_i+2]; _n++; }
          }
          if (_n > 0) fabricHex = ((_r/_n|0) << 16) | ((_g/_n|0) << 8) | (_b/_n|0);
        } catch(e) {}

        var frontMat = new THREE.MeshStandardMaterial({
          map: tex, side: THREE.FrontSide, roughness: 0.6, metalness: 0.0,
        });
        var backMat = new THREE.MeshStandardMaterial({
          color: fabricHex, side: THREE.FrontSide, roughness: 0.7, metalness: 0.0,
        });

        var fGeo = new THREE.CylinderGeometry(radTop, radBot, clothH, 32, 2, true, -Math.PI/2, Math.PI);
        var fMesh = new THREE.Mesh(fGeo, frontMat);
        fMesh.position.y = midY; fMesh.castShadow = true;
        clothGroup.add(fMesh);

        var bGeo = new THREE.CylinderGeometry(radTop, radBot, clothH, 20, 1, true, Math.PI/2, Math.PI);
        var bMesh = new THREE.Mesh(bGeo, backMat);
        bMesh.position.y = midY; bMesh.castShadow = true;
        clothGroup.add(bMesh);

        clothMeshes[gType] = clothGroup;
        clothGroup._lastDrapeData = { imageUrl: imageUrl, garmentType: gType };
        modelGroup.add(clothGroup);

        if (window.ReactNativeWebView) {
          window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'drape_applied' }));
        }
      }, function(err) {
        console.warn('Cloth texture failed:', err);
        if (window.ReactNativeWebView) {
          window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'texture_failed', msg: 'Texture load failed' }));
        }
      });
    }
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
          var rawModel = gltf.scene;

          // Compute raw bounds
          var box = new THREE.Box3().setFromObject(rawModel);
          var size = new THREE.Vector3();
          box.getSize(size);
          var center = new THREE.Vector3();
          box.getCenter(center);
          
          // Place model so feet are at local Y=0 (bottom of bounding box at origin)
          rawModel.position.x = -center.x;
          rawModel.position.y = -box.min.y;
          rawModel.position.z = -center.z;

          // Create a wrapper group for scaling so the center remains perfectly anchored
          var scaleWrapper = new THREE.Group();
          scaleWrapper.name = "glb_model";
          scaleWrapper.add(rawModel);

          // Calculate initial base scale
          var maxDim = Math.max(size.x, size.y, size.z);
          var scale = 18.0 / maxDim;
          scaleWrapper.scale.setScalar(scale);
          scaleWrapper.userData.baseScale = scale;

          // Store raw (pre-hScale) Y extents so applyGLBScale can recalculate
          // currentCamTargetY when the height slider changes
          glbBaseMinY  = 0;
          glbBaseSizeY = size.y * scale;
          scaleWrapper.userData.baseMinY  = glbBaseMinY;
          scaleWrapper.userData.baseSizeY = glbBaseSizeY;

          // Enable shadows, override color to grey, and hide hair meshes
          rawModel.traverse(function(child) {
            if (child.isMesh) {
              // Hide hair / scalp / eyebrow / eyelash meshes — mannequin should be bald
              var cName = (child.name || '').toLowerCase();
              var mName = '';
              if (child.material) {
                if (Array.isArray(child.material)) {
                  mName = child.material.map(function(m){ return (m.name||'').toLowerCase(); }).join(' ');
                } else {
                  mName = (child.material.name || '').toLowerCase();
                }
              }
              var combined = cName + ' ' + mName;
              if (combined.indexOf('hair') !== -1 || combined.indexOf('scalp') !== -1 ||
                  combined.indexOf('eyelash') !== -1 || combined.indexOf('eyebrow') !== -1 ||
                  combined.indexOf('brow') !== -1 || combined.indexOf('lash') !== -1) {
                child.visible = false;
                return;
              }

              child.castShadow = true;
              child.receiveShadow = true;
              if (Array.isArray(child.material)) {
                child.material.forEach(function(m) { 
                  m.side = THREE.FrontSide; 
                  m.needsUpdate = true; 
                });
              } else if (child.material) {
                child.material.side = THREE.FrontSide; 
                child.material.needsUpdate = true;
              }
            }
          });

          // ── Close arm–body gap ─────────────────────────────────────
          // 1) Rotate arm bones inward (works on rigged models)
          var armBoneFound = false;
          rawModel.traverse(function(child) {
            if (!child.isBone) return;
            var bn = (child.name || '').toLowerCase();
            // Match common bone names: upperarm, upper_arm, arm.l/.r, shoulder
            var isLeftArm  = (bn.indexOf('left') !== -1 || bn.indexOf('.l') !== -1 || bn.indexOf('_l') !== -1)
                          && (bn.indexOf('arm') !== -1 || bn.indexOf('shoulder') !== -1);
            var isRightArm = (bn.indexOf('right') !== -1 || bn.indexOf('.r') !== -1 || bn.indexOf('_r') !== -1)
                          && (bn.indexOf('arm') !== -1 || bn.indexOf('shoulder') !== -1);
            // Only adjust upper arm / shoulder — not forearm / hand
            if (bn.indexOf('fore') !== -1 || bn.indexOf('hand') !== -1 || bn.indexOf('wrist') !== -1 ||
                bn.indexOf('finger') !== -1 || bn.indexOf('thumb') !== -1) return;
            if (isLeftArm) {
              child.rotation.z = (child.rotation.z || 0) - 0.18;
              armBoneFound = true;
            } else if (isRightArm) {
              child.rotation.z = (child.rotation.z || 0) + 0.18;
              armBoneFound = true;
            }
          });
          // Force skeleton update so skinned mesh follows new bone positions
          rawModel.traverse(function(child) {
            if (child.isSkinnedMesh && child.skeleton) child.skeleton.update();
          });

          // 2) Add armpit bridge spheres (fills any remaining visual gap)
          var bridgeMat = new THREE.MeshStandardMaterial({
            color: 0x808080, metalness: 0.12, roughness: 0.28, side: THREE.FrontSide
          });
          var bridgeR  = size.x * 0.035;
          var armpitY  = box.min.y + size.y * 0.73;  // armpit in original model coords
          var armpitXL = center.x - size.x * 0.23;
          var armpitXR = center.x + size.x * 0.23;
          var armpitZ  = center.z;

          var leftBridge = new THREE.Mesh(new THREE.SphereGeometry(bridgeR, 12, 8), bridgeMat);
          leftBridge.position.set(armpitXL, armpitY, armpitZ);
          leftBridge.scale.set(2.0, 3.0, 1.2);
          leftBridge.castShadow = true;
          rawModel.add(leftBridge);

          var rightBridge = new THREE.Mesh(new THREE.SphereGeometry(bridgeR, 12, 8), bridgeMat);
          rightBridge.position.set(armpitXR, armpitY, armpitZ);
          rightBridge.scale.set(2.0, 3.0, 1.2);
          rightBridge.castShadow = true;
          rawModel.add(rightBridge);

          // Log bone names for debugging (remove before release)
          var boneNames = [];
          rawModel.traverse(function(c) { if (c.isBone) boneNames.push(c.name); });
          console.log('GLB bones:', boneNames.join(', '));

          modelGroup.add(scaleWrapper);

          // Compute camera aim point against the already-scaled wrapper
          var newBox = new THREE.Box3().setFromObject(scaleWrapper);
          var newSize = new THREE.Vector3();
          newBox.getSize(newSize);
          currentCamTargetY  = 0;

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
          // Apply baked-in initial proportions immediately — don't wait for animation loop
          window._mDirty = true;
          if (typeof window.applyGLBScale === 'function') window.applyGLBScale();

          if (pendingDrapes && pendingDrapes.length > 0) {
            for (var di = 0; di < pendingDrapes.length; di++) {
              drapeGarment(pendingDrapes[di].imageUrl, pendingDrapes[di].garmentType);
            }
            pendingDrapes = [];
          }

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
    if (pendingDrapes && pendingDrapes.length > 0) {
      for (var di = 0; di < pendingDrapes.length; di++) {
        drapeGarment(pendingDrapes[di].imageUrl, pendingDrapes[di].garmentType);
      }
      pendingDrapes = [];
    }

    if (window.ReactNativeWebView) {
      window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'model_fallback' }));
    }
  }

  function computeProportions(heightCm, weightKg, bodyType) {
    var h = heightCm / 100;
    var totalHeight = h * 5.7; // Scaled up to match GLB 10.0 scale
    var bmi = weightKg / (h * h);
    var bmiNorm = Math.max(0, Math.min(1, (bmi - 15) / 25));
    var hScale = heightCm / 175;

    // ── Body-type anatomical profiles ─────────────────────────────────────
    // Each type encodes the visual silhouette described in the design spec.
    // Values are multipliers applied to the base 175 cm / average reference.
    var types = {
      // Rectangle/I-shape: narrow everywhere, bones prominent, long limbs
      ectomorph: {
        shoulder: 0.80, chest: 0.76, waist: 0.80, hip: 0.78,
        armThick: 0.72, forearmThick: 0.70, legThick: 0.74, calfThick: 0.72,
        neckThick: 0.82,
        // Torso is almost straight — chest ≈ waist ≈ hip
        waistRelChest: 0.96, hipRelChest: 0.98,
        // Belly protrusion (Z depth)
        bellyZ: 0.72,
        // Head: slightly larger relative to thin body
        headScale: 1.04,
        // Leg separation
        legSepMult: 0.48,
      },
      // Balanced, slight shoulder taper, healthy padding
      average: {
        shoulder: 1.00, chest: 1.00, waist: 1.00, hip: 1.00,
        armThick: 1.00, forearmThick: 1.00, legThick: 1.00, calfThick: 1.00,
        neckThick: 1.00,
        waistRelChest: 0.88, hipRelChest: 0.96,
        bellyZ: 1.00,
        headScale: 1.00,
        legSepMult: 0.55,
      },
      // Inverted triangle V-taper: very broad shoulders, tight waist, peaked muscles
      mesomorph: {
        shoulder: 1.20, chest: 1.15, waist: 0.82, hip: 0.90,
        armThick: 1.22, forearmThick: 1.15, legThick: 1.18, calfThick: 1.14,
        neckThick: 1.14,
        // Strong taper: waist much narrower than chest
        waistRelChest: 0.72, hipRelChest: 0.82,
        bellyZ: 0.85,
        headScale: 0.98,
        legSepMult: 0.52,
      },
      // Oval/O-shape: wide midsection, rounded joints, belly dominates
      endomorph: {
        shoulder: 1.08, chest: 1.20, waist: 1.30, hip: 1.25,
        armThick: 1.18, forearmThick: 1.14, legThick: 1.22, calfThick: 1.16,
        neckThick: 1.18,
        // Waist wider than chest — belly is widest point
        waistRelChest: 1.10, hipRelChest: 1.05,
        bellyZ: 1.40,
        headScale: 1.02,
        legSepMult: 0.62,
      },
      hourglass: {
        shoulder: 1.06, chest: 1.10, waist: 0.78, hip: 1.14,
        armThick: 0.95, forearmThick: 0.92, legThick: 1.04, calfThick: 1.00,
        neckThick: 0.96,
        waistRelChest: 0.72, hipRelChest: 1.04,
        bellyZ: 0.90, headScale: 1.00, legSepMult: 0.56,
      },
      pear: {
        shoulder: 0.88, chest: 0.92, waist: 0.96, hip: 1.22,
        armThick: 0.88, forearmThick: 0.86, legThick: 1.12, calfThick: 1.06,
        neckThick: 0.92,
        waistRelChest: 0.90, hipRelChest: 1.18,
        bellyZ: 0.92, headScale: 1.00, legSepMult: 0.60,
      },
    };
    var bt = types[bodyType] || types.average;

    // Base measurements at reference height — BMI adds soft padding on top
    var shoulderHalfW = (0.22 * hScale) * bt.shoulder * (1 + bmiNorm * 0.08);
    var chestR  = (0.14 * hScale) * bt.chest  * (1 + bmiNorm * 0.20);
    var waistR  = chestR * bt.waistRelChest   * (1 + bmiNorm * 0.32);
    var hipR    = chestR * bt.hipRelChest     * (1 + bmiNorm * 0.26);

    var headH      = totalHeight * 0.12 * bt.headScale;
    var neckH      = totalHeight * 0.04;
    var torsoH     = totalHeight * 0.30;
    var upperArmH  = totalHeight * 0.16;
    var forearmH   = totalHeight * 0.14;
    var upperLegH  = totalHeight * 0.22;
    var lowerLegH  = totalHeight * 0.21;

    var armR      = 0.035 * bt.armThick      * (1 + bmiNorm * 0.28);
    var forearmR  = 0.030 * bt.forearmThick  * (1 + bmiNorm * 0.22);
    var upperLegR = 0.065 * bt.legThick      * (1 + bmiNorm * 0.28);
    var lowerLegR = 0.045 * bt.calfThick     * (1 + bmiNorm * 0.18);
    var neckR     = 0.040 * bt.neckThick     * (1 + bmiNorm * 0.15);

    // Belly / endomorph Z-protrusion: endomorph has a pronounced forward belly
    var bellyProtrusion = bt.bellyZ * (1 + bmiNorm * 0.6);

    // Muscle peak factor: mesomorph gets arm/shoulder bulge
    var musclePeak = (bodyType === 'mesomorph') ? 1.28 : 1.0;

    var footY    = -totalHeight * 0.48;
    var ankleY   = footY + 0.03;
    var kneeY    = ankleY + lowerLegH;
    var hipY     = kneeY + upperLegH;
    var waistY   = hipY + torsoH * 0.28;
    var chestY   = hipY + torsoH * 0.68;
    var shoulderY= hipY + torsoH;
    var neckY    = shoulderY + neckH * 0.5;
    var headY    = shoulderY + neckH + headH * 0.5;
    var legSep   = hipR * bt.legSepMult;

    return {
      totalHeight, headH, neckH, torsoH, upperArmH, forearmH, upperLegH, lowerLegH,
      shoulderHalfW, chestR, waistR, hipR,
      armR, forearmR, upperLegR, lowerLegR, neckR,
      headY, neckY, shoulderY, chestY, waistY, hipY, kneeY, ankleY, footY, legSep,
      bellyProtrusion, musclePeak,
      bodyType: bodyType,
    };
  }

  function updateGeo(mesh, geo) {
    if (mesh.geometry) mesh.geometry.dispose();
    mesh.geometry = geo;
  }

  function applyProportions(p) {
    var pp = proceduralParts;
    var isEcto  = p.bodyType === 'ectomorph';
    var isMeso  = p.bodyType === 'mesomorph';
    var isEndo  = p.bodyType === 'endomorph';

    // ── Head ──────────────────────────────────────────────────────────────
    // Ectomorph: slightly angular; Endomorph: rounder/fuller
    updateGeo(pp.head, new THREE.SphereGeometry(p.headH * 0.45, 32, 24));
    pp.head.position.y = p.headY;
    var headScaleX = isEndo ? 1.06 : isEcto ? 0.96 : 1.0;
    var headScaleZ = isEndo ? 1.08 : isEcto ? 0.98 : 1.05;
    pp.head.scale.set(headScaleX, 1.22, headScaleZ);

    // ── Neck ──────────────────────────────────────────────────────────────
    // Ectomorph: thin long neck; Endomorph: short/wide; Mesomorph: thick/defined
    var neckBaseR = p.neckR || p.waistR * 0.38;
    var neckTopR  = p.headH * (isEcto ? 0.17 : isEndo ? 0.22 : 0.19);
    var neckCurve = new THREE.SplineCurve([
      new THREE.Vector2(neckBaseR, 0),
      new THREE.Vector2(neckBaseR * (isEndo ? 0.98 : 0.88), p.neckH * 0.5),
      new THREE.Vector2(neckTopR,  p.neckH)
    ]);
    updateGeo(pp.neck, new THREE.LatheGeometry(neckCurve.getPoints(12), 24));
    pp.neck.position.y = p.shoulderY;
    pp.neck.scale.set(1, 1, isEndo ? 1.2 : isEcto ? 0.85 : 1.0);

    // ── Torso ─────────────────────────────────────────────────────────────
    // The torso LatheGeometry profile encodes the silhouette of each body type:
    // - Ectomorph: nearly straight column (rectangle)
    // - Average:   gentle S-curve with slight taper
    // - Mesomorph: strong V-taper (wide chest, tight waist), shoulder caps
    // - Endomorph: belly at mid-torso is the widest point (O/oval shape)
    var torsoTopR;
    if (isEcto)  torsoTopR = p.shoulderHalfW * 0.78; // narrow top
    else if (isMeso) torsoTopR = p.shoulderHalfW * 0.88; // wide shoulder cap
    else if (isEndo) torsoTopR = p.shoulderHalfW * 0.82;
    else torsoTopR = p.shoulderHalfW * 0.84;

    var torsoMidWaistR;
    if (isEcto)  torsoMidWaistR = p.waistR * 0.97; // almost same as chest → straight
    else if (isMeso) torsoMidWaistR = p.waistR * 0.78; // deeply pinched waist
    else if (isEndo) torsoMidWaistR = p.waistR * 1.08; // belly pushes out at mid-torso
    else torsoMidWaistR = p.waistR * 0.90;

    // Belly mid-point (endomorph): extra control point for the forward protrusion
    var torsoCurvePoints;
    if (isEndo) {
      // Add extra belly bulge control point between hip and waist
      torsoCurvePoints = [
        new THREE.Vector2(p.hipR,            0),
        new THREE.Vector2(p.waistR * 1.12,   p.torsoH * 0.18),  // lower belly bulge
        new THREE.Vector2(torsoMidWaistR,    p.torsoH * 0.40),  // widest belly point
        new THREE.Vector2(p.chestR * 1.05,   p.torsoH * 0.68),  // chest (slightly wide)
        new THREE.Vector2(torsoTopR,         p.torsoH)
      ];
    } else if (isMeso) {
      torsoCurvePoints = [
        new THREE.Vector2(p.hipR,            0),
        new THREE.Vector2(torsoMidWaistR,    p.torsoH * 0.28),  // tight waist
        new THREE.Vector2(p.chestR,          p.torsoH * 0.65),  // broad chest
        new THREE.Vector2(torsoTopR,         p.torsoH)
      ];
    } else if (isEcto) {
      torsoCurvePoints = [
        new THREE.Vector2(p.hipR,            0),
        new THREE.Vector2(torsoMidWaistR,    p.torsoH * 0.30),  // barely tapers
        new THREE.Vector2(p.chestR,          p.torsoH * 0.68),
        new THREE.Vector2(torsoTopR,         p.torsoH)
      ];
    } else {
      torsoCurvePoints = [
        new THREE.Vector2(p.hipR,            0),
        new THREE.Vector2(torsoMidWaistR,    p.torsoH * 0.30),
        new THREE.Vector2(p.chestR,          p.torsoH * 0.68),
        new THREE.Vector2(torsoTopR,         p.torsoH)
      ];
    }
    var torsoCurve = new THREE.SplineCurve(torsoCurvePoints);
    updateGeo(pp.torso, new THREE.LatheGeometry(torsoCurve.getPoints(18), 32));
    pp.torso.position.y = p.hipY;
    // Z-scale drives belly depth: endomorph protrudes forward, ectomorph is flat
    var torsoZScale = isEndo ? 0.85 * p.bellyProtrusion
                   : isEcto  ? 0.52
                   : isMeso  ? 0.60
                   : 0.65;
    pp.torso.scale.set(1, 1, torsoZScale);

    // ── Arms ──────────────────────────────────────────────────────────────
    // Mesomorph: bicep peak at 40% of upper arm; Ectomorph: thin uniform tube;
    // Endomorph: soft rounded arm without definition
    var armRestAngle = isEndo ? 0.06 : isMeso ? 0.10 : 0.08;
    var elbowY = p.shoulderY - p.upperArmH;
    var wristY = elbowY - p.forearmH;

    var bicepPeakMult = isMeso ? p.musclePeak : isEcto ? 0.88 : isEndo ? 1.05 : 1.0;
    var upperArmCurve = new THREE.SplineCurve([
      new THREE.Vector2(p.armR * 0.70,                    0),
      new THREE.Vector2(p.armR * bicepPeakMult,           p.upperArmH * 0.40), // bicep peak
      new THREE.Vector2(p.armR * 0.92,                    p.upperArmH * 0.72),
      new THREE.Vector2(p.armR * 0.78,                    p.upperArmH)
    ]);
    var upperArmGeo = new THREE.LatheGeometry(upperArmCurve.getPoints(14), 24);
    updateGeo(pp.leftUpperArm,  upperArmGeo);
    pp.leftUpperArm.position.set(-(p.shoulderHalfW + p.armR * 0.5), elbowY, 0);
    pp.leftUpperArm.rotation.z  = armRestAngle;
    updateGeo(pp.rightUpperArm, upperArmGeo);
    pp.rightUpperArm.position.set(p.shoulderHalfW + p.armR * 0.5, elbowY, 0);
    pp.rightUpperArm.rotation.z = -armRestAngle;

    // Forearm: slightly more tapered on ectomorph; fuller on endomorph
    var forearmTopMult = isEcto ? 0.80 : isEndo ? 1.02 : 0.88;
    var forearmCurve = new THREE.SplineCurve([
      new THREE.Vector2(p.forearmR * 0.50, 0),
      new THREE.Vector2(p.forearmR * forearmTopMult, p.forearmH * 0.65),
      new THREE.Vector2(p.armR * 0.72,    p.forearmH)
    ]);
    var forearmGeo = new THREE.LatheGeometry(forearmCurve.getPoints(12), 24);
    var armOffsetX = p.upperArmH * Math.sin(armRestAngle);
    updateGeo(pp.leftForearm,  forearmGeo);
    pp.leftForearm.position.set(-(p.shoulderHalfW + p.armR * 0.5 + armOffsetX), wristY, 0);
    pp.leftForearm.rotation.z  = armRestAngle * 0.5;
    updateGeo(pp.rightForearm, forearmGeo);
    pp.rightForearm.position.set(p.shoulderHalfW + p.armR * 0.5 + armOffsetX, wristY, 0);
    pp.rightForearm.rotation.z = -armRestAngle * 0.5;

    // Hands: endomorph = rounded/puffy; ectomorph = thin
    var handR = p.forearmR * (isEndo ? 0.62 : isEcto ? 0.50 : 0.55);
    updateGeo(pp.leftHand,  new THREE.SphereGeometry(handR, 16, 16));
    pp.leftHand.position.set(-(p.shoulderHalfW + p.armR * 0.6 + armOffsetX), wristY - p.forearmR, 0);
    pp.leftHand.scale.set(1, isEndo ? 1.2 : 1.4, isEndo ? 0.5 : 0.4);
    updateGeo(pp.rightHand, new THREE.SphereGeometry(handR, 16, 16));
    pp.rightHand.position.set(p.shoulderHalfW + p.armR * 0.6 + armOffsetX, wristY - p.forearmR, 0);
    pp.rightHand.scale.set(1, isEndo ? 1.2 : 1.4, isEndo ? 0.5 : 0.4);

    // ── Legs ──────────────────────────────────────────────────────────────
    // Mesomorph: quad sweep/flare at upper thigh; Endomorph: thick/round thighs;
    // Ectomorph: long lean tubes with very little flare
    var quadFlare = isMeso ? 1.14 : isEndo ? 1.10 : isEcto ? 0.92 : 1.0;
    var upperLegCurve = new THREE.SplineCurve([
      new THREE.Vector2(p.lowerLegR * 0.88,             0),
      new THREE.Vector2(p.upperLegR * quadFlare,         p.upperLegH * 0.55),
      new THREE.Vector2(p.upperLegR * (isMeso ? 1.08 : isEndo ? 1.06 : 1.02), p.upperLegH)
    ]);
    var upperLegGeo = new THREE.LatheGeometry(upperLegCurve.getPoints(12), 24);
    updateGeo(pp.leftUpperLeg,  upperLegGeo);
    pp.leftUpperLeg.position.set(-p.legSep, p.kneeY, 0);
    updateGeo(pp.rightUpperLeg, upperLegGeo);
    pp.rightUpperLeg.position.set(p.legSep, p.kneeY, 0);

    // Lower leg / calf: mesomorph = defined calf; ectomorph = thin;
    var calfBulge = isMeso ? 1.06 : isEndo ? 1.04 : isEcto ? 0.88 : 1.0;
    var lowerLegCurve = new THREE.SplineCurve([
      new THREE.Vector2(p.lowerLegR * 0.58, 0),
      new THREE.Vector2(p.lowerLegR * calfBulge, p.lowerLegH * 0.52),
      new THREE.Vector2(p.lowerLegR * 0.88, p.lowerLegH)
    ]);
    var lowerLegGeo = new THREE.LatheGeometry(lowerLegCurve.getPoints(12), 24);
    updateGeo(pp.leftLowerLeg,  lowerLegGeo);
    pp.leftLowerLeg.position.set(-p.legSep, p.ankleY, 0);
    updateGeo(pp.rightLowerLeg, lowerLegGeo);
    pp.rightLowerLeg.position.set(p.legSep, p.ankleY, 0);

    // Feet: endomorph slightly wider/fatter; ectomorph narrow
    var footW = p.lowerLegR * (isEndo ? 1.7 : isEcto ? 1.3 : 1.5);
    var footH = p.lowerLegR * (isEndo ? 1.0 : 0.8);
    var footD = p.lowerLegR * (isEndo ? 3.8 : isEcto ? 3.2 : 3.5);
    var footGeo = new THREE.BoxGeometry(footW, footH, footD);
    updateGeo(pp.leftFoot,  footGeo);
    pp.leftFoot.position.set(-p.legSep, p.footY + p.lowerLegR * 0.4, p.lowerLegR);
    updateGeo(pp.rightFoot, footGeo);
    pp.rightFoot.position.set(p.legSep, p.footY + p.lowerLegR * 0.4, p.lowerLegR);

    // ── Camera & ground offset ────────────────────────────────────────────
    var totalModelHeight = p.headY - p.footY;
    var localCentreY = p.footY + totalModelHeight * 0.5;
    modelGroupOffsetY  = -localCentreY;
    currentCamTargetY  = 0;
    modelFeetWorldY = modelGroupOffsetY + p.footY;

    var fovHalfRad = 27.5 * Math.PI / 180;
    targetCamZ = Math.max(8.0, Math.min(18.0, (totalModelHeight / 2) / Math.tan(fovHalfRad) * 1.12));
    camera.position.z = camera.position.z + (targetCamZ - camera.position.z) * 0.6;
  }

  window.updateMannequinModel = function(data) {
    if (!data) return;
    if (data.type === 'update') {
      window._mH  = Math.max(100, Math.min(230, parseFloat(data.heightCm) || 175));
      window._mW  = Math.max(30,  Math.min(200, parseFloat(data.weightKg) || 70));
      window._mBT = data.bodyType || 'average';
      window._mDirty = true;
      var dbg = document.getElementById('_dbg');
      if (dbg) dbg.textContent = 'H:' + window._mH + ' W:' + window._mW + ' ' + window._mBT;
      // Apply immediately — don't wait for animation loop tick
      if (typeof window.applyGLBScale === 'function') window.applyGLBScale();
    } else if (data.type === 'drape') {
      drapeGarment(data.imageUrl, data.garmentType || 'upper_body');
      // drape_applied is sent by drapeGarment after texture loads
    } else if (data.type === 'remove_cloth') {
      if (data.garmentType) {
        drapeGarment(null, data.garmentType);
      } else {
        var rmKeys = Object.keys(clothMeshes);
        for (var ri = 0; ri < rmKeys.length; ri++) { drapeGarment(null, rmKeys[ri]); }
      }
    } else if (data.type === 'capture') {
      renderer.render(scene, camera);
      var b64 = renderer.domElement.toDataURL('image/png');
      if (window.ReactNativeWebView) window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'snapshot_result', base64: b64 }));
    }
  };

  // Listen on both window and document (covers iOS + Android RN WebView)
  function _onMsg(e) { try { window.updateMannequinModel(JSON.parse(e.data)); } catch(x) {} }
  window.addEventListener('message', _onMsg);
  document.addEventListener('message', _onMsg);

  // ─── GLB scale function — callable from inject AND animation loop ─────────
  // Exposed on window so React Native injectJavaScript can call it direcxtly.
  window.applyGLBScale = function() {
    var h  = window._mH  || 175;
    var w  = window._mW  || 70;
    var bt = window._mBT || 'average';

    var glb = null;
    for (var ci = 0; ci < modelGroup.children.length; ci++) {
      if (modelGroup.children[ci].name === 'glb_model') { glb = modelGroup.children[ci]; break; }
    }
    if (!glb) glb = modelGroup.getObjectByName('glb_model');
    if (!glb) return; // model not loaded yet

    // Height: proportional to 175 cm reference
    var hScale = h / 175.0;
    // BMI-driven width (decoupled from height)
    var bmi    = w / Math.pow(h / 100.0, 2);
    var wScale = Math.pow(bmi / 22.857, 0.55);

    // Body-type X/Z multipliers for the GLB model
    // X = frontal width, Z = depth (belly protrusion)
    // Matches the anatomical profiles in computeProportions:
    //   Ectomorph  → narrow rectangle, thin Z
    //   Average    → balanced baseline
    //   Mesomorph  → broad X (V-taper), lean Z
    //   Endomorph  → wide X + pronounced Z (belly forward)
    var btMap = {
      ectomorph: { x: 0.76, z: 0.74 },  // slim/narrow, minimal depth
      mesomorph: { x: 1.18, z: 0.96 },  // broad shoulders, lean belly
      endomorph: { x: 1.24, z: 1.38 },  // wide midsection + belly protrusion
      average:   { x: 1.00, z: 1.00 },
      hourglass: { x: 1.06, z: 0.98 },
      pear:      { x: 1.16, z: 1.05 },
    };
    var bts = btMap[bt] || btMap['average'];

    var base = glb.userData.baseScale || 1.0;
    glb.scale.set(
      base * wScale * bts.x,  // width  — BMI + body-type
      base * hScale,           // height — real cm
      base * wScale * bts.z   // depth  — BMI + body-type
    );

    // GLB is centred at its local origin — offset stays zero
    currentCamTargetY  = 0;
    modelGroupOffsetY  = 0;

    // Compute where the feet land in world space so the ground plane can follow
    // Feet are at Y=0 since model was repositioned with feet at origin
    modelFeetWorldY = 0;

    // Scale camera Z so the full figure always fits;
    // snap camera instantly (via large lerp factor in animate) to avoid drift
    targetCamZ = Math.max(4.0, Math.min(8.0, 5.5 * hScale));
    camera.position.z = camera.position.z + (targetCamZ - camera.position.z) * 0.6;

    // Keep proportions in sync for correct garment draping at new height
    var bMinY = glbBaseMinY  * hScale;
    var bSizeY = glbBaseSizeY * hScale;
    currentProportions = {
      hipR:          bSizeY * 0.09 * wScale,
      shoulderHalfW: bSizeY * 0.11 * wScale,
      torsoH:        bSizeY * 0.30,
      upperLegH:     bSizeY * 0.22,
      lowerLegH:     bSizeY * 0.21,
      hipY:          bMinY + bSizeY * 0.24,
      shoulderY:     bMinY + bSizeY * 0.78,
      ankleY:        bMinY + bSizeY * 0.04,
    };

    window._mDirty = false;

    var dbg = document.getElementById('_dbg');
    if (dbg) dbg.textContent = h + 'cm/' + w + 'kg/' + bt;
  };

  // ─── Animation loop ───────────────────────────────
  var time = 0;

  function applyTargetScale() {
    if (!window._mDirty) return;
    var h  = window._mH  || 175;
    var w  = window._mW  || 70;
    var bt = window._mBT || 'average';

    if (proceduralGroup && proceduralParts.head) {
      currentProportions = computeProportions(h, w, bt);
      applyProportions(currentProportions);
      var rk1 = Object.keys(clothMeshes);
      for (var ri1 = 0; ri1 < rk1.length; ri1++) {
        var cm1 = clothMeshes[rk1[ri1]];
        if (cm1 && cm1._lastDrapeData) drapeGarment(cm1._lastDrapeData.imageUrl, cm1._lastDrapeData.garmentType);
      }
      window._mDirty = false;
    } else if (modelLoaded && modelGroup) {
      // applyGLBScale is also exposed as window.applyGLBScale for direct calls from injectJavaScript
      if (typeof window.applyGLBScale === 'function') {
        window.applyGLBScale();
      }
      // Re-drape all clothing layers so they fit the updated body proportions
      var rk2 = Object.keys(clothMeshes);
      for (var ri2 = 0; ri2 < rk2.length; ri2++) {
        var cm2 = clothMeshes[rk2[ri2]];
        if (cm2 && cm2._lastDrapeData) drapeGarment(cm2._lastDrapeData.imageUrl, cm2._lastDrapeData.garmentType);
      }
    }
  }

  function animate() {
    requestAnimationFrame(animate);
    time += 0.016;

    applyTargetScale();

    if (autoRotate) { userRotationY += 0.003; }
    modelGroup.rotation.y = userRotationY;

    // Always keep modelGroup centred — modelGroupOffsetY compensates for non-centred local origins
    modelGroup.position.y = modelGroupOffsetY + Math.sin(time * 1.2) * 0.005;

    // Camera Y tracks model centre — only Z zooms in/out
    var glbCamY = glbBaseSizeY * 0.12;
    camera.position.y += (glbCamY - camera.position.y) * 0.15;
    camera.position.z += (targetCamZ - camera.position.z) * 0.15;
    camera.lookAt(0, glbCamY, 0);

    if (ground) ground.position.y = modelFeetWorldY - 0.02;
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', function() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
<\/script>

<!-- Debug overlay: shows active height/weight — remove before release -->
<div id="_dbg" style="position:fixed;bottom:6px;left:8px;font-size:9px;color:rgba(0,0,0,0.28);pointer-events:none;font-family:monospace">175cm/70kg</div>

</body>
</html>
`;
}
