package com.aiwardrobe

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle

import com.facebook.react.ReactActivity
import com.facebook.react.ReactActivityDelegate
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint.fabricEnabled
import com.facebook.react.defaults.DefaultReactActivityDelegate

import expo.modules.ReactActivityDelegateWrapper

class MainActivity : ReactActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    // Set the theme to AppTheme BEFORE onCreate to support
    // coloring the background, status bar, and navigation bar.
    // This is required for expo-splash-screen.
    setTheme(R.style.AppTheme);
    intent = purchaseShareIntent(intent)
    super.onCreate(null)
  }

  // Convert bounded ACTION_SEND text into the existing deep-link pathway.
  // No logging, merchant fetch, image import or background notification.
  private fun purchaseShareIntent(original: Intent): Intent {
    if (original.action != Intent.ACTION_SEND || original.type != "text/plain") return original
    val text = original.getCharSequenceExtra(Intent.EXTRA_TEXT)?.toString()?.take(4096) ?: return original
    return Intent(original).apply {
      action = Intent.ACTION_VIEW
      data = Uri.Builder().scheme("com.aiwardrobe").authority("buy-check").appendQueryParameter("input", text).build()
      removeExtra(Intent.EXTRA_TEXT)
    }
  }

  override fun onNewIntent(intent: Intent) {
    val received = purchaseShareIntent(intent)
    setIntent(received)
    super.onNewIntent(received)
  }

  /**
   * Returns the name of the main component registered from JavaScript. This is used to schedule
   * rendering of the component.
   */
  override fun getMainComponentName(): String = "main"

  /**
   * Returns the instance of the [ReactActivityDelegate]. We use [DefaultReactActivityDelegate]
   * which allows you to enable New Architecture with a single boolean flags [fabricEnabled]
   */
  override fun createReactActivityDelegate(): ReactActivityDelegate {
    return ReactActivityDelegateWrapper(
          this,
          BuildConfig.IS_NEW_ARCHITECTURE_ENABLED,
          object : DefaultReactActivityDelegate(
              this,
              mainComponentName,
              fabricEnabled
          ){})
  }

  /**
    * Align the back button behavior with Android S
    * where moving root activities to background instead of finishing activities.
    * @see <a href="https://developer.android.com/reference/android/app/Activity#onBackPressed()">onBackPressed</a>
    */
  override fun invokeDefaultOnBackPressed() {
      if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.R) {
          if (!moveTaskToBack(false)) {
              // For non-root activities, use the default implementation to finish them.
              super.invokeDefaultOnBackPressed()
          }
          return
      }

      // Use the default back button implementation on Android S
      // because it's doing more than [Activity.moveTaskToBack] in fact.
      super.invokeDefaultOnBackPressed()
  }
}
