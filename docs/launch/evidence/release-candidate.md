# Local release candidate report

RELEASE CANDIDATE BLOCKED

Version 3.0.2; iOS 65; Android 63. Base commit 5784fc5cff24381cdae982e937d841fca611c7cd; dirty=true; worktree SHA-256 3df0c34ea3530b2edc13d60aa48811e116d8e02f2eb99b0599dba41f1ce974d6.

| Check | Result | Evidence |
|---|---|---|
| typecheck | PASS | docs/launch/evidence/typecheck.log |
| lint | PASS | docs/launch/evidence/lint.log |
| edge_typecheck | PASS | docs/launch/evidence/edge_typecheck.log |
| ios_bundle | PASS | docs/launch/evidence/ios_bundle.log |
| android_bundle | PASS | docs/launch/evidence/android_bundle.log |
| critical_tests | PASS | docs/launch/evidence/critical_tests.log |
| full_regressions | PASS | docs/launch/evidence/full_regressions.log |
| migration_tests | PASS | docs/launch/evidence/migration_tests.log |
| security_checks | PASS | docs/launch/evidence/security_checks.log |
| secrets_scan | PASS | docs/launch/evidence/secrets_scan.log |
| feature_flags_localization | PASS | docs/launch/evidence/feature_flags_localization.log |
| compatibility_contracts | PASS | docs/launch/evidence/compatibility_contracts.log |
| accessibility_feedback | PASS | docs/launch/evidence/accessibility_feedback.log |
| beta_review_tools | PASS | docs/launch/evidence/beta_review_tools.log |
| accessibility_device | MANUAL REQUIRED | VoiceOver/TalkBack, text scaling, screen-reader result order and contrast on four required devices pending |
| compatibility_installed_binaries | MANUAL REQUIRED | A/B/C/D backend + installed old/new client matrix not proven by contract tests |
| migration_hosted_baseline | MANUAL REQUIRED | Pending set tested against bounded fixtures, not hosted schema. Approve schema comparison/backup rehearsal separately. |
| device_qa_performance | MANUAL REQUIRED | Four required devices, native image imports/scroll/memory/background, signed release build and sandbox billing pending |
| remote_activation | MANUAL REQUIRED | Ordered packet: docs/launch/REMOTE_ACTIVATION.md; all remote mutations require explicit approval |
| secret_history_rotation | MANUAL REQUIRED | Current text worktree scanned. History/bundles and rotation of historical credential require review; see blockers. |
| source_unchanged_during_check | PASS | Worktree fingerprint excludes generated evidence only |

REMOTE ACTIVATION PENDING EXPLICIT APPROVAL
