# AIWardrobe App Stabilization & Quality Improvement PRD

## 1. Executive Summary
- AIWardrobe is currently experiencing significant stability, usability, and compliance issues across core features (Virtual Try-On, Admin Panel, Paywall, App Store compliance). This initiative focuses on a comprehensive stabilization effort to resolve critical bugs, improve the reliability of the Virtual Try-On pipeline, ensure smooth post-purchase navigation, and meet all Apple App Store review guidelines for successful submission.

## 2. Problem Statement
- **Who has this problem?** End-users experiencing app crashes, failed try-ons, and broken navigation. App reviewers rejecting the app due to compliance issues. Administrators unable to reliably manage catalog images.
- **What is the problem?** "Many issues, not all functions usable, not all working correctly." Specifically: the FLUX.1 Virtual Try-On pipeline fails sequentially, paywall routing post-purchase is broken, Admin image uploads fail to update the UI, and App Store rejections persist (ATT, EULA, Subscription flows).
- **Why is it painful?** It degrades user trust, blocks revenue generation (broken paywall routing), prevents app updates (App Store rejections), and creates operational overhead for administrators.
- **Evidence:** Recent attempts to fix Virtual Try-On black screens, App Store rejections (Guideline 3.1.2(c), 3.1.1, 2.1(a)), Admin image upload failures, and post-purchase navigation crashes (`navigation.reset` errors).

## 3. Target Users & Personas
- **Primary persona(s):** Daily app users who rely on the Virtual Try-On and Outfit Calendar features.
- **Secondary persona(s):** App Administrators (managing the catalog/inspo images) and Apple App Store Reviewers.
- **Jobs-to-be-done:** 
  - **Users:** Reliably generate layered virtual try-on images without errors. Easily navigate the app after subscribing.
  - **Admins:** Seamlessly upload and manage catalog images with immediate UI updates.

## 4. Strategic Context
- **Business goals (OKRs):** Successfully launch on the iOS App Store. Improve App stability and crash-free sessions to >99%. Increase subscription conversion by fixing post-purchase flows.
- **Market opportunity:** A stable, high-fidelity AI virtual try-on app can capture significant market share in the digital fashion space.
- **Competitive landscape:** Competitors offer stable apps; AIWardrobe must match baseline stability while providing superior FLUX.1-powered AI generations.
- **Why now?** The app is currently blocked from App Store approval and existing features are causing user friction, directly impacting growth and revenue.

## 5. Solution Overview
- **High-level description:** A dedicated sprint/phase focused purely on technical debt reduction, bug fixing, and compliance.
- **Key features (Fixes):**
  - **Virtual Try-On Stabilization:** Refine FLUX.1-kontext-dev prompt engineering and edge functions to ensure sequential multi-layer garment retention (top -> layer -> pants -> shoes) without black screens or state loss.
  - **Paywall & Navigation:** Fix `The action 'RESET' with payload...` error in `PaywallScreen.tsx` to route users correctly to the Main App tab post-purchase.
  - **Admin Panel Image Uploads:** Bypass React Native network limitations using `expo-file-system/legacy` for direct binary uploads to Supabase and implement cache-busting for immediate UI updates.
  - **App Store Compliance:** Implement App Tracking Transparency (ATT), add a functional EULA link, integrate native offer code redemption sheets, and ensure the free tier is accessible without a forced subscription flow.
  - **Performance/Memory:** Address memory leaks in React Native components identified in the project audit.

## 6. Success Metrics
- **Primary metric:** App Store Approval (Status: Approved).
- **Secondary metrics:** 
  - Virtual Try-On success rate (>95% without black screens).
  - Post-purchase successful navigation rate (100%).
  - Admin image upload success rate with immediate UI update (100%).

## 7. User Stories & Requirements
- **Epic hypothesis:** By focusing on core stability and compliance, we will unlock App Store distribution and significantly improve user retention.
- **User stories with acceptance criteria:**
  - *As a user, I want the Virtual Try-On to correctly layer multiple items so I can see a complete outfit.* (AC: Top, layer, pants, and shoes can be added sequentially without erasing previous layers).
  - *As a subscriber, I want to be redirected to the home screen immediately after purchasing so I can use the app.* (AC: Paywall success correctly resets navigation stack to the main app navigator).
  - *As an admin, I want uploaded images to appear immediately in the Inspo/Add/Guide tabs.* (AC: Uploads use file system binary upload, UI refreshes immediately without manual reload).
  - *As an App Store Reviewer, I need to see a functional EULA and not be forced into a subscription.* (AC: EULA link works, skip button exists on paywall, ATT prompt appears correctly).

## 8. Out of Scope
- Building net-new features (e.g., social sharing, new AI models other than stabilizing FLUX.1).
- Major UI redesigns (unless required for App Store compliance).

## 9. Dependencies & Risks
- **Technical dependencies:** Supabase Edge Functions, Replicate API (FLUX.1), Expo File System, RevenueCat (In-App Purchases).
- **External dependencies:** Apple App Store Review team approval timelines.
- **Risks and mitigations:** 
  - *Risk:* Fixing VTON prompts degrades image quality. *Mitigation:* Extensive testing of the sequential pipeline.
  - *Risk:* Admin upload fixes cause memory issues. *Mitigation:* Ensure proper file stream cleanup after upload.

## 10. Open Questions
- What is the exact route name for the post-purchase screen in `App.tsx` (e.g., `HomeTab`, `Main`)?
- Are there specific devices or OS versions where the Virtual Try-On black screen occurs more frequently?
- Do we have all the required URLs (Privacy Policy, EULA, Support) finalized for App Store Connect?
