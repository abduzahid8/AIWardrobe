# TestFlight Compatibility Analysis - Virtual Try-On

**Date:** May 31, 2026  
**Status:** ⚠️ **MOSTLY WORKING - WITH IMPORTANT CAVEATS**

---

## Executive Summary

The virtual try-on solution **will mostly work on TestFlight**, but there are **critical issues** that need to be addressed before production deployment:

### ✅ What Will Work
- UI rendering and interactions
- Mannequin 3D display
- Garment selection and preview
- Subscription gating
- Error handling

### ⚠️ What Might Have Issues
- Modal GPU API calls (hardcoded URL)
- Three.js library loading (CDN dependencies)
- CORS and network policies
- Certificate pinning (if enabled)
- App Transport Security (ATS)

### ❌ Critical Issues to Fix
1. **Hardcoded Modal URL** - Not configurable for different environments
2. **CDN Dependencies** - Three.js loaded from unpkg/CDN
3. **No Environment Switching** - Can't easily test different API endpoints
4. **Missing Error Boundaries** - Network failures could crash the app

---

## Detailed Analysis

### 1. Hardcoded URLs ⚠️

**Location:** `features/try-on/AITryOnScreen.tsx:18`

```typescript
const MODAL_VTON_URL = 'https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run';
```

**Issue:** 
- Hardcoded production URL
- No way to test with staging/development endpoints
- If Modal service goes down, no fallback

**TestFlight Impact:** 🔴 **HIGH**
- TestFlight testers will hit production API
- Can't test with staging environment
- No way to debug API issues without changing code

**Solution:**
```typescript
// ✅ RECOMMENDED FIX
const MODAL_VTON_URL = Config.api.modalUrl || 'https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run';

// In env.ts:
modal: {
  url: str(process.env.EXPO_PUBLIC_MODAL_VTON_URL, 'https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run'),
}
```

---

### 2. Three.js CDN Dependencies ⚠️

**Location:** `features/try-on/utils/mannequin3D.ts:77-80`

```html
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/DRACOLoader.js"></script>
```

**Issue:**
- Depends on external CDNs (unpkg, CDN.jsdelivr.net)
- Network failures will break 3D rendering
- No offline fallback
- CDN might be blocked in some regions

**TestFlight Impact:** 🟡 **MEDIUM**
- If CDN is slow, 3D loading will be slow
- If CDN is down, 3D rendering fails
- Users on restricted networks might not see mannequin

**Solution:**
```typescript
// ✅ RECOMMENDED FIX
// Bundle Three.js locally instead of loading from CDN
// In package.json: three is already installed
// Use a bundler to include it in the WebView

// Or use a fallback:
const DRACO_DECODER_PATH = process.env.EXPO_PUBLIC_DRACO_DECODER_URL || 
  'https://www.gstatic.com/draco/versioned/decoders/1.5.6/';
```

---

### 3. App Transport Security (ATS) ⚠️

**Location:** `app.json` (iOS configuration)

**Current Status:** Not explicitly configured

**Issue:**
- iOS requires HTTPS for all network requests
- Modal URL is HTTPS ✅ (good)
- Supabase URL is HTTPS ✅ (good)
- Three.js CDN is HTTPS ✅ (good)
- But ATS might block some requests

**TestFlight Impact:** 🟡 **MEDIUM**
- Might block requests to certain domains
- Need to verify all domains are HTTPS
- May need ATS exceptions for development

**Solution:**
```json
{
  "ios": {
    "infoPlist": {
      "NSAppTransportSecurity": {
        "NSAllowsArbitraryLoads": false,
        "NSExceptionDomains": {
          "modal.run": {
            "NSIncludesSubdomains": true,
            "NSTemporaryExceptionAllowsInsecureHTTPLoads": false,
            "NSTemporaryExceptionMinimumTLSVersion": "TLSv1.2"
          },
          "supabase.co": {
            "NSIncludesSubdomains": true,
            "NSTemporaryExceptionAllowsInsecureHTTPLoads": false
          }
        }
      }
    }
  }
}
```

---

### 4. CORS and Network Policies ⚠️

**Issue:**
- Modal API might have CORS restrictions
- WebView might have different CORS behavior than browser
- Network requests from WebView might be blocked

**TestFlight Impact:** 🟡 **MEDIUM**
- API calls might fail with CORS errors
- WebView might not send proper headers
- Cross-origin image loading might fail

**Solution:**
```typescript
// ✅ RECOMMENDED FIX
// Ensure proper headers in API calls
const resp = await axios.post(
  `${MODAL_VTON_URL}${endpoint}`,
  modalPayload,
  { 
    timeout: 240_000,
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': 'AIWardrobe-Mobile/1.0.8',
    }
  },
);
```

---

### 5. Certificate Pinning ⚠️

**Issue:**
- If certificate pinning is enabled, TestFlight might fail
- Modal service certificate might change
- Supabase certificate might change

**TestFlight Impact:** 🟡 **MEDIUM**
- Requests might fail if certificates don't match
- Need to update pinned certificates regularly

**Solution:**
```typescript
// ✅ RECOMMENDED FIX
// Don't use certificate pinning for external APIs
// Only use for your own backend if needed
// Keep certificates updated in eas.json
```

---

### 6. WebView Limitations ⚠️

**Issue:**
- WebView might have different JavaScript engine
- WebView might not support all Three.js features
- WebView might have memory limitations

**TestFlight Impact:** 🟡 **MEDIUM**
- 3D rendering might be slower
- Complex models might not load
- Memory issues on older devices

**Solution:**
```typescript
// ✅ RECOMMENDED FIX
// Add memory monitoring
// Reduce model complexity for older devices
// Add fallback for WebView limitations

if (Platform.OS === 'ios') {
  // iOS WebView specific optimizations
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
}
```

---

### 7. Network Timeout Issues ⚠️

**Current:** 240 seconds (4 minutes)

**Issue:**
- TestFlight might have stricter network policies
- Cellular networks might timeout faster
- WiFi might be unstable during testing

**TestFlight Impact:** 🟡 **MEDIUM**
- API calls might timeout on slow networks
- Retry logic might not be enough
- Users on 3G might experience failures

**Solution:**
```typescript
// ✅ RECOMMENDED FIX
// Implement adaptive timeout based on network type
import NetInfo from '@react-native-community/netinfo';

const getTimeout = async () => {
  const state = await NetInfo.fetch();
  if (state.type === 'cellular') {
    return 300_000; // 5 minutes for cellular
  }
  return 240_000; // 4 minutes for WiFi
};
```

---

### 8. Image Processing on TestFlight ⚠️

**Issue:**
- Base64 encoding might be slow on older devices
- Large images might cause memory issues
- File system access might be restricted

**TestFlight Impact:** 🟡 **MEDIUM**
- Image encoding might timeout
- Memory pressure on older devices
- File system permissions might be denied

**Solution:**
```typescript
// ✅ RECOMMENDED FIX
// Add image size validation
// Compress images before encoding
// Add progress feedback

const validateImageSize = (uri: string) => {
  const maxSize = 5 * 1024 * 1024; // 5MB
  // Check file size before processing
};

const compressImage = async (uri: string) => {
  // Use expo-image-manipulator to compress
  // Reduce quality/resolution for faster encoding
};
```

---

### 9. Subscription Verification ⚠️

**Issue:**
- RevenueCat might have network issues
- Subscription status might not sync
- Feature gating might fail

**TestFlight Impact:** 🟡 **MEDIUM**
- Users might not be able to access try-on
- Subscription status might be incorrect
- Quota might not be tracked properly

**Solution:**
```typescript
// ✅ RECOMMENDED FIX
// Add offline subscription caching
// Implement retry logic for subscription checks
// Add fallback for network failures

const checkSubscription = async () => {
  try {
    const status = await getSubscriptionStatus();
    cacheSubscriptionStatus(status);
    return status;
  } catch (error) {
    // Fall back to cached status
    return getCachedSubscriptionStatus();
  }
};
```

---

### 10. Error Handling & Crash Prevention ⚠️

**Issue:**
- Network errors might crash the app
- Missing error boundaries
- No graceful degradation

**TestFlight Impact:** 🔴 **HIGH**
- App might crash on network failures
- Users might see black screen
- No way to recover from errors

**Solution:**
```typescript
// ✅ RECOMMENDED FIX
// Add error boundary
// Implement graceful error handling
// Add retry UI

try {
  const result = await handleAITryOn();
} catch (error) {
  if (error.code === 'NETWORK_ERROR') {
    setAiError('Network connection failed. Please check your connection and try again.');
  } else if (error.code === 'TIMEOUT') {
    setAiError('Request timed out. Please try again.');
  } else {
    setAiError('An unexpected error occurred. Please try again.');
  }
}
```

---

## TestFlight Specific Issues

### Build Configuration

**Current:** ✅ Good
```json
{
  "ios": {
    "buildNumber": "45",
    "bundleIdentifier": "com.aiwardrobe"
  }
}
```

### Permissions

**Current:** ✅ Good
```json
{
  "NSCameraUsageDescription": "...",
  "NSPhotoLibraryUsageDescription": "...",
  "NSLocationWhenInUseUsageDescription": "..."
}
```

### Updates Configuration

**Current:** ⚠️ Needs Review
```json
{
  "updates": {
    "enabled": true,
    "checkAutomatically": "NEVER",
    "fallbackToCacheTimeout": 10000
  }
}
```

**Issue:** 
- Updates are enabled but checking is disabled
- Might cause version mismatch issues
- TestFlight might have different update behavior

**Solution:**
```json
{
  "updates": {
    "enabled": false,
    "checkAutomatically": "NEVER"
  }
}
```

---

## Pre-TestFlight Checklist

### ✅ Must Fix Before TestFlight

- [ ] Move hardcoded Modal URL to environment config
- [ ] Add error boundaries for network failures
- [ ] Implement retry UI for failed requests
- [ ] Add network status monitoring
- [ ] Test on slow networks (3G simulation)
- [ ] Test on older devices (iPhone 11 or older)
- [ ] Verify all HTTPS URLs
- [ ] Test with VPN enabled
- [ ] Test with restricted networks

### ⚠️ Should Fix Before TestFlight

- [ ] Bundle Three.js locally instead of CDN
- [ ] Add image size validation
- [ ] Implement adaptive timeouts
- [ ] Add subscription caching
- [ ] Add offline support
- [ ] Implement progress indicators
- [ ] Add detailed error messages

### 🟡 Nice to Have

- [ ] Add analytics for API performance
- [ ] Add crash reporting
- [ ] Add performance monitoring
- [ ] Add user feedback mechanism
- [ ] Add debug mode for TestFlight

---

## Recommended Changes

### 1. Update Environment Configuration

**File:** `src/config/env.ts`

```typescript
export const Config = {
  // ... existing config
  api: {
    url: str(process.env.EXPO_PUBLIC_API_URL, 'https://aiwardrobe-api.onrender.com'),
    alicevisionUrl: str(process.env.EXPO_PUBLIC_ALICEVISION_URL),
    modalUrl: str(process.env.EXPO_PUBLIC_MODAL_VTON_URL, 'https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run'),
  },
  // ... rest of config
}
```

### 2. Update EAS Configuration

**File:** `eas.json`

```json
{
  "build": {
    "production": {
      "env": {
        "EXPO_PUBLIC_MODAL_VTON_URL": "https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run"
      }
    },
    "testflight": {
      "distribution": "internal",
      "ios": {
        "image": "latest"
      },
      "env": {
        "EXPO_PUBLIC_MODAL_VTON_URL": "https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run"
      }
    }
  }
}
```

### 3. Update Try-On Component

**File:** `features/try-on/AITryOnScreen.tsx`

```typescript
import Config from '../../src/config/env';

// Replace hardcoded URL
const MODAL_VTON_URL = Config.api.modalUrl;
```

### 4. Add Error Boundary

**File:** `features/try-on/AITryOnScreen.tsx`

```typescript
// Add try-catch around API calls
try {
  data = await callModalDirectly();
} catch (err: any) {
  const status = err?.response?.status;
  const msg = err?.response?.data?.detail || err?.message;
  
  if (status === 0 || err.code === 'ECONNABORTED') {
    setAiError('Network connection failed. Please check your connection and try again.');
  } else if (err.code === 'ENOTFOUND') {
    setAiError('Cannot reach the server. Please check your internet connection.');
  } else {
    setAiError(msg || 'Outfit render failed.');
  }
}
```

---

## Testing Recommendations

### Before Submitting to TestFlight

1. **Network Testing**
   - Test on 3G network
   - Test with VPN enabled
   - Test with network throttling
   - Test with airplane mode + WiFi

2. **Device Testing**
   - Test on iPhone 11 (older device)
   - Test on iPhone 15 (newer device)
   - Test on iPad
   - Test on different iOS versions

3. **Scenario Testing**
   - Test with no internet
   - Test with slow internet
   - Test with interrupted connection
   - Test with subscription expired
   - Test with quota exceeded

4. **Performance Testing**
   - Monitor memory usage
   - Monitor CPU usage
   - Monitor battery drain
   - Monitor network bandwidth

---

## Conclusion

### Will It Work on TestFlight?

**Short Answer:** ⚠️ **Mostly yes, but with caveats**

**Detailed Answer:**
- ✅ UI and interactions will work
- ✅ 3D rendering will work (mostly)
- ✅ Subscription gating will work
- ⚠️ API calls might have issues
- ⚠️ Network failures might crash app
- ⚠️ CDN dependencies might fail

### Recommendation

**Before TestFlight:**
1. ✅ Move hardcoded URLs to environment config
2. ✅ Add error boundaries for network failures
3. ✅ Implement retry UI
4. ✅ Test on slow networks
5. ✅ Test on older devices

**After TestFlight:**
1. Monitor crash reports
2. Monitor API performance
3. Collect user feedback
4. Fix any issues found
5. Plan Phase 2 improvements

### Risk Level

**Current:** 🔴 **HIGH** (hardcoded URLs, no error boundaries)  
**After Fixes:** 🟡 **MEDIUM** (normal for new feature)  
**Production:** 🟢 **LOW** (after monitoring and fixes)

---

## Action Items

### Immediate (Before TestFlight)
- [ ] Move Modal URL to environment config
- [ ] Add error boundaries
- [ ] Test on slow networks
- [ ] Test on older devices

### Short Term (After TestFlight)
- [ ] Monitor crash reports
- [ ] Collect user feedback
- [ ] Fix any issues found
- [ ] Optimize performance

### Long Term (Phase 2)
- [ ] Bundle Three.js locally
- [ ] Add offline support
- [ ] Implement analytics
- [ ] Add more features

---

## Support

For questions or issues:
1. Check console logs
2. Review error messages
3. Check network status
4. Verify subscription status
5. Contact support team

---

**Generated:** May 31, 2026  
**Status:** Ready for TestFlight with recommended fixes
