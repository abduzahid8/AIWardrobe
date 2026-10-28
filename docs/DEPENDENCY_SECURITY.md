# Dependency security checks

Install with the project Node version and peer checks enabled:

```sh
npm ci --legacy-peer-deps=false
npm run test:dependency-security
npm run test:webdriver-toolchain
```

The project `.npmrc` enables peer checks too. EAS profiles additionally pass `NPM_CONFIG_LEGACY_PEER_DEPS=false` in the environment so the isolated Git preparation subprocess receives the setting. A fresh local EAS production install failed to resolve `appium/support.js` and `appium/driver.js` with only the project configuration; an explicit environment setting allowed the pinned source install to complete. A cold-cache production build is being verified. Do not use legacy peer mode for this dependency graph: Git dependency preparation needs its Appium peer to compile the driver.

The XCUITest driver is built from the official v12.18.1 source at commit `1be253af959bbd29427166a910a78a38fdd66185`. The published shrinkwrapped/bundled packages retained vulnerable transitive copies that npm overrides could not replace. The pinned source has no bundle/shrinkwrap, so normal npm resolution applies the existing patched `proxy-addr` and `shell-quote` overrides. No third-party source patch or runtime loader replacement is used.

The tested tooling versions are Appium 3.8.0 and expect-webdriverio 6.0.9. The latter satisfies the installed WebDriver globals peer requirement. The visual service's matcher registration was verified against that assertion instance; controlled pass/fail results exercise compatibility without taking screenshots.

Updating the source pin requires a clean peer-enabled install, the two commands above, the full application suite, TypeScript and a driver runtime check. Git and access to the public upstream repository are required. Keep the source commit pinned and review lockfile changes.

These regression checks cover specific advisory attack shapes, not every vulnerability. The latest audit still contains high and moderate findings. See [the readiness report](reports/READINESS_10000_DAU.md) for current evidence and remaining gates. A loopback Appium status check is not proof of a WebDriverAgent build, simulator compatibility or a successful UI session.


## 25 October2026 targeted patches

Pinned fast-uri3.1.9 and serialize-javascript7.1.3. Only these two installed package versions changed. Attack-shape tests reproduced encoded-scheme URL confusion and a trailing closing-script serializer escape before the update (2/2failed); patched security/toolchain checks pass15/15. See the maintainers’ [fast-uri advisory](https://github.com/fastify/fast-uri/security/advisories/GHSA-w74x-v6f9-j3xg) and [serialize-javascript advisory](https://github.com/yahoo/serialize-javascript/security/advisories/GHSA-m5mm-p8h7-m29v).

All2,379app tests/186suites, TypeScript and iOS Hermes export pass. The fresh npm audit changed from80findings(65high/13moderate/2low) to76(65high/11moderate),0critical. Neither patched package remains flagged. These scoped toolchain changes do not establish full app security or production capacity. [Evidence](reports/readiness-evidence/oct25-targeted-dependency-patches.json).


## 25 October2026 HTTP dependency patches

Pinned follow-redirects1.16.1 and both compatible Undici branches (6.29.1/7.30.1). Local loopback fixtures reproduced credential-header forwarding and crashes on truncated excessive-content-encoding responses before patching. The patched attack cases pass; all18security/toolchain checks,2,379app tests/186suites, TypeScript and iOS Hermes export pass. See [follow-redirects](https://github.com/follow-redirects/follow-redirects/security/advisories/GHSA-q3jh-wrc6-jgq8) and [Undici](https://github.com/nodejs/undici/security/advisories/GHSA-xjj6-f33q-929c) maintainer advisories.

Audit76→74findings; neither updated family remains flagged.63high/11moderate remain. These patches were not deployed as a backend or installed native app update. [Evidence](reports/readiness-evidence/oct25-http-dependency-patches.json).

## 26 October 2026 build-install investigation

The published XCUITest 12.20.0 package installed but its bundled shell-quote 1.10.0 failed the existing post-comment newline regression case. The update was rejected and the official 12.18.1 source pin restored. Its restored installed graph passes all 17 dependency security checks and the WebDriver assertion integration check. The audit remains 74 findings (63 high / 11 moderate / 0 critical). No security assertions were relaxed. EAS profile environment propagation is the scoped build configuration change; a cold-cache release build remains the acceptance check.
