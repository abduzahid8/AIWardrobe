# TestFlight Action Plan - Virtual Try-On

## Quick Summary

✅ **Will mostly work on TestFlight**  
⚠️ **But needs 3 critical fixes first**  
🔴 **Risk: HIGH without fixes, MEDIUM with fixes**

---

## Critical Issues (Must Fix)

### 1. Hardcoded Modal URL ❌

**Problem:** API endpoint is hardcoded, can't be changed for different environments

**Current Code:**
```typescript
// features/try-on/AITryOnScreen.tsx:18
const MODAL_VTON_URL = 'https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run';
```

**Fix (5 minutes):**

**Step 1:** Update `src/config/env.ts`
```typescript
export const Config = {
  // ... existing
  api: {
    url: str(process.env.EXPO_PUBLIC_API_URL, 'https://aiwardrobe-api.onrender.com'),
    alicevisionUrl: str(process.env.EXPO_PUBLIC_ALICEVISION_URL),
    modalUrl: str(process.env.EXPO_PUBLIC_MODAL_VTON_URL, 'https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run'),
  },
}
```

**Step 2:** Update `features/try-on/AITryOnScreen.tsx`
```typescript
import Config from '../../src/config/env';

// Replace line 18:
const MODAL_VTON_URL = Config.api.modalUrl;
```

**Step 3:** Update `eas.json`
```json
{
  "build": {
    "production": {
      "env": {
        "EXPO_PUBLIC_MODAL_VTON_URL": "https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run"
      }
    }
  }
}
```

**Impact:** ✅ Allows environment-specific configuration

---

### 2. Missing Error Boundaries ❌

**Problem:** Network failures might crash the app

**Current Code:**
```typescript
// features/try-on/AITryOnScreen.tsx:365-380
try {
  data = await callModalDirectly();
} catch (err: any) {
  // Only handles specific errors
  const status = err?.response?.status;
  const msg = err?.response?.data?.detail || err?.message;
  console.warn(`[AITryOn] Modal call failed (${status}): ${msg}`);
  // ... retry logic
}
```

**Fix (10 minutes):**

Add network error detection:
```typescript
try {
  data = await callModalDirectly();
} catch (err: any) {
  const status = err?.response?.status;
  const msg = err?.response?.data?.detail || err?.message;
  
  // Add network error detection
  if (!err.response) {
    // Network error (no response from server)
    if (err.code === 'ECONNABORTED') {
      setAiError('Request timed out. Please check your connection and try again.');
    } else if (err.code === 'ENOTFOUND' || err.code === 'ECONNREFUSED') {
      setAiError('Cannot reach the server. Please check your internet connection.');
    } else {
      setAiError('Network connection failed. Please try again.');
    }
    return;
  }
  
  // Server error (has response)
  console.warn(`[AITryOn] Modal call failed (${status}): ${msg}`);
  setAiProgress('Retrying…');
  await new Promise((r) => setTimeout(r, 8_000));
  // ... retry logic
}
```

**Impact:** ✅ Prevents crashes from network errors

---

### 3. No Retry UI ❌

**Problem:** Users don't know if the app is retrying or stuck

**Current Code:**
```typescript
setAiProgress('Retrying…');
await new Promise((r) => setTimeout(r, 8_000));
```

**Fix (5 minutes):**

Add better retry feedback:
```typescript
// Show retry attempt number
let retryAttempt = 1;
setAiProgress(`Retrying (attempt ${retryAttempt}/2)…`);
await new Promise((r) => setTimeout(r, 8_000));

// After retry
if (!data?.success) {
  setAiProgress(null);
  setAiError('Outfit render failed after retry. Please try again.');
  return;
}
```

**Impact:** ✅ Better user experience during retries

---

## Important Issues (Should Fix)

### 4. Three.js CDN Dependencies ⚠️

**Problem:** 3D rendering depends on external CDNs

**Current Code:**
```html
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
```

**Fix (30 minutes):**

Option A: Use local Three.js (recommended)
```typescript
// Already installed in package.json
// Just need to bundle it properly
```

Option B: Add fallback
```typescript
const DRACO_DECODER_PATH = process.env.EXPO_PUBLIC_DRACO_DECODER_URL || 
  'https://www.gstatic.com/draco/versioned/decoders/1.5.6/';
```

**Impact:** ✅ More reliable 3D rendering

---

### 5. Image Size Validation ⚠️

**Problem:** Large images might cause memory issues

**Current Code:**
```typescript
const b64 = await FileSystem.readAsStringAsync(localUri, { encoding: 'base64' })
```

**Fix (10 minutes):**

Add validation:
```typescript
// Check file size before encoding
const fileInfo = await FileSystem.getInfoAsync(localUri);
if (fileInfo.size > 5 * 1024 * 1024) {
  setAiError('Image is too large. Please use a smaller image.');
  return;
}

const b64 = await FileSystem.readAsStringAsync(localUri, { encoding: 'base64' })
```

**Impact:** ✅ Prevents memory issues

---

## Testing Checklist

### Before TestFlight Build

- [ ] Apply all 3 critical fixes
- [ ] Run tests: `npm test -- --no-coverage`
- [ ] Test on simulator (iPhone 14)
- [ ] Test on device (iPhone 12 or older)
- [ ] Test with slow network (3G simulation)
- [ ] Test with no internet
- [ ] Test with VPN enabled

### During TestFlight

- [ ] Monitor crash reports
- [ ] Monitor API performance
- [ ] Collect user feedback
- [ ] Track error rates
- [ ] Monitor network issues

### After TestFlight

- [ ] Fix any issues found
- [ ] Optimize performance
- [ ] Plan Phase 2 improvements
- [ ] Prepare for App Store

---

## Implementation Timeline

### Phase 1: Critical Fixes (1-2 hours)
1. Move Modal URL to config (5 min)
2. Add error boundaries (10 min)
3. Add retry UI (5 min)
4. Test changes (30 min)
5. Commit and push (5 min)

### Phase 2: Important Fixes (2-3 hours)
1. Bundle Three.js locally (30 min)
2. Add image validation (10 min)
3. Add network monitoring (20 min)
4. Test changes (30 min)
5. Commit and push (5 min)

### Phase 3: TestFlight Submission (1 hour)
1. Build for TestFlight (20 min)
2. Submit to App Store (10 min)
3. Wait for review (varies)
4. Distribute to testers (5 min)

---

## Risk Assessment

### Current Risk: 🔴 HIGH
- Hardcoded URLs
- No error boundaries
- Network failures might crash app
- CDN dependencies might fail

### After Critical Fixes: 🟡 MEDIUM
- Environment-specific config
- Error boundaries in place
- Better error messages
- Still has CDN dependencies

### After All Fixes: 🟢 LOW
- Fully configurable
- Robust error handling
- Local dependencies
- Production ready

---

## Success Criteria

### TestFlight Success
- ✅ No crashes on network failures
- ✅ API calls work on slow networks
- ✅ Error messages are clear
- ✅ Retry logic works
- ✅ 3D rendering works on older devices

### Production Success
- ✅ < 1% crash rate
- ✅ < 5% error rate
- ✅ < 30s average response time
- ✅ > 95% user satisfaction
- ✅ < 100MB app size

---

## Rollback Plan

If issues are found on TestFlight:

1. **Minor Issues** (< 5% users affected)
   - Fix in next build
   - Continue testing

2. **Major Issues** (> 5% users affected)
   - Pause TestFlight
   - Fix issues
   - Resubmit

3. **Critical Issues** (crashes, data loss)
   - Immediately pause TestFlight
   - Investigate root cause
   - Fix and retest
   - Resubmit

---

## Communication Plan

### Before TestFlight
- [ ] Notify team of fixes
- [ ] Update documentation
- [ ] Prepare release notes

### During TestFlight
- [ ] Daily monitoring
- [ ] Weekly status updates
- [ ] Respond to tester feedback

### After TestFlight
- [ ] Publish results
- [ ] Plan next phase
- [ ] Prepare for App Store

---

## Questions & Answers

### Q: Will it work on TestFlight without fixes?
**A:** Mostly yes, but network failures might crash the app.

### Q: How long will fixes take?
**A:** 1-2 hours for critical fixes, 2-3 hours for all fixes.

### Q: Can we skip the fixes?
**A:** Not recommended. Risk of crashes and poor user experience.

### Q: What if Modal API goes down?
**A:** App will crash without error boundaries. With fixes, will show error message.

### Q: How do we test on slow networks?
**A:** Use Xcode network throttling or Charles Proxy.

### Q: What about older devices?
**A:** Test on iPhone 11 or older. Monitor memory usage.

---

## Next Steps

1. **Immediately:**
   - [ ] Review this document
   - [ ] Assign fixes to team members
   - [ ] Create GitHub issues

2. **Today:**
   - [ ] Implement critical fixes
   - [ ] Run tests
   - [ ] Test on devices

3. **Tomorrow:**
   - [ ] Implement important fixes
   - [ ] Final testing
   - [ ] Prepare for TestFlight

4. **This Week:**
   - [ ] Submit to TestFlight
   - [ ] Monitor results
   - [ ] Collect feedback

---

## Resources

- **Expo Documentation:** https://docs.expo.dev/
- **TestFlight Guide:** https://developer.apple.com/testflight/
- **Network Testing:** Xcode Network Link Conditioner
- **Error Tracking:** Sentry or Bugsnag

---

## Summary

✅ **Virtual try-on will mostly work on TestFlight**  
⚠️ **But needs 3 critical fixes (1-2 hours)**  
🎯 **Recommended: Apply fixes before TestFlight**  
📊 **Risk: HIGH → MEDIUM → LOW (with fixes)**

**Recommendation:** Apply all fixes before TestFlight submission.

---

**Generated:** May 31, 2026  
**Status:** Ready for implementation
