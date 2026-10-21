# Dependency security checks

Install with the project Node version and peer checks enabled:

```sh
npm ci --legacy-peer-deps=false
npm run test:dependency-security
npm run test:webdriver-toolchain
```

The project `.npmrc` enables peer checks too. Do not use legacy peer mode for this dependency graph: Git dependency preparation needs its Appium peer to compile the driver.

The XCUITest driver is built from the official v12.18.1 source at commit `1be253af959bbd29427166a910a78a38fdd66185`. The published shrinkwrapped/bundled packages retained vulnerable transitive copies that npm overrides could not replace. The pinned source has no bundle/shrinkwrap, so normal npm resolution applies the existing patched `proxy-addr` and `shell-quote` overrides. No third-party source patch or runtime loader replacement is used.

The tested tooling versions are Appium 3.8.0 and expect-webdriverio 6.0.9. The latter satisfies the installed WebDriver globals peer requirement. The visual service's matcher registration was verified against that assertion instance; controlled pass/fail results exercise compatibility without taking screenshots.

Updating the source pin requires a clean peer-enabled install, the two commands above, the full application suite, TypeScript and a driver runtime check. Git and access to the public upstream repository are required. Keep the source commit pinned and review lockfile changes.

These regression checks cover specific advisory attack shapes, not every vulnerability. The latest audit still contains high and moderate findings. See [the readiness report](reports/READINESS_10000_DAU.md) for current evidence and remaining gates. A loopback Appium status check is not proof of a WebDriverAgent build, simulator compatibility or a successful UI session.
