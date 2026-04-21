# Native Testing Guide (Espresso & XCUITest)

For deep native testing that specifically targets platform-level behavior (like Three.js GLB loading performance or native camera modules), you can use the native tools provided by Google and Apple.

## Android: Espresso

Espresso tests live in your `android/` directory.

### Location
`android/app/src/androidTest/java/com/aiwardrobe/`

### How to use
1. Open the `android` folder in **Android Studio**.
2. Navigate to the `androidTest` folder.
3. Create a new Test Class.
4. Use the `ActivityScenarioRule` to launch `MainActivity`.

**Example Snippet (Kotlin):**
```kotlin
@RunWith(AndroidJUnit4::class)
class ClosetScreenTest {
    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)

    @Test
    fun verifyClosetTitle() {
        onView(withText("My Closet")).check(matches(isDisplayed()))
    }
}
```

---

## iOS: XCUITest

XCUITest tests live in your `ios/` directory under a Test Target.

### Location
`ios/AIWardrobeTests/` or `ios/AIWardrobeUITests/`

### How to use
1. Open `ios/AIWardrobe.xcworkspace` in **Xcode**.
2. Select **File > New > Target... > iOS UI Testing Bundle**.
3. Name it `AIWardrobeUITests`.
4. Command+U to run tests.

**Example Snippet (Swift):**
```swift
func testVerifyClosetTitle() throws {
    let app = XCUIApplication()
    app.launch()
    
    let closetHeader = app.staticTexts["My Closet"]
    XCTAssertTrue(closetHeader.exists)
}
```

---

## When to use Native vs Maestro?
- **Maestro**: 90% of your E2E needs (UI flows, navigation, logic).
- **Native**: 10% of needs (High-performance measuring, testing native modules you've written in Java/Swift, complex gesture handling deep in the OS).
