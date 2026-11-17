# Launch blocker register

`blockers.json` is the authoritative structured register, consumed by release:check. Open validation gaps are distinguished from reproduced product defects. Every owner role needs a named person before activation.

| ID | Area / severity | Reproducible? / flow | Impact / workaround | Owner | Fix status | Regression evidence | Launch decision |
|---|---|---|---|---|---|---|---|
| RC-001 | Devices/billing / BLOCKER | Validation gap / startup, upgrade, offline, restore | Critical safety unproven / developer-only | Release + mobile QA | OPEN | Q1–Q8 × device matrix | NO-GO |
| RC-002 | Remote safety / BLOCKER | Validation gap / migrations, compatibility, rollback | Schema drift and remote rollback unproven / do not activate | Backend/on-call | OPEN | SQL log is local only; target drills pending approval | NO-GO |
| RC-003 | Credential hygiene / HIGH | Key-shaped value found in old document | Historical copies may contain credential / redacted locally | Credential owner | Local redaction done; classification/history/rotation pending | Secret scan | Review before distribution |
| RC-004 | Migration grants / HIGH | SQL inspection and isolated probe / private persistence, reward ledgers | Missing explicit grants, client ledger forging / remote changes withheld | Backend | CLOSED locally | releaseIsolation.sql | Hosted proof still pending |
| RC-005 | Analytics / HIGH | Mocked Edge error and concurrent enqueue / evidence capture | Evidence erased / local voluntary export | Mobile | CLOSED locally | telemetry.test.ts | Full rerun required |
| RC-006 | Regression harness / MEDIUM | 14 failed assertions in 5 suites | Stale four-slot/callback/seed expectations / fixtures corrected | Mobile QA | CLOSED locally | triage suites, full regression log | Require fresh full run |
| RC-007 | Flags / HIGH | Source search / advanced providers | Six metered Edge controls prepared locally; deployed/legacy coverage unverified / advanced activation withheld | Backend/mobile | OPEN | FEATURES.md wiring disclosure | Prove effective provider deny and budget control |

| RC-008 | Visual ownership / HIGH | Cross-owner FK probe | Private links could reference another owner / withheld remotely | Backend | CLOSED locally | Composite owner FKs + SQL deny probes | Hosted proof pending |

| RC-009 | Tooling dependency / HIGH | Nested shell-quote 1.10.0 failed injection regression | Unsafe quoting in Appium toolchain / repaired resolution | Mobile tooling | CLOSED locally | All copies resolve 1.11.0; security suite | Fresh full gate required |

| RC-010 | Feedback keyboard / HIGH | Reproduced on native iOS Release | Correction fields hidden by keyboard / fixed locally | Mobile | CLOSED locally | Native UI + before/after screenshots | Full app/Android QA still required |

Do not relabel an untested gate as PASS. Any discovered cross-user read/write, startup crash, data loss, forged server entitlement, hard-constraint violation, destructive migration uncertainty or unbounded paid path becomes a new BLOCKER immediately.
