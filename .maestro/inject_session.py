#!/usr/bin/env python3
"""Inject a Supabase session into Expo Go's AsyncStorage manifest.

Usage: inject_session.py <session.json> [udid]

The app MUST be terminated first — React Native keeps AsyncStorage's
manifest.json in memory and would overwrite an external edit on its next
write.  run_save_checks.sh terminates before calling this.

The stored value is the GoTrue session object (access_token, refresh_token,
expires_at, expires_in, token_type, user) under the key
`sb-<project-ref>-auth-token`, which is what @supabase/supabase-js reads on
boot.
"""
import glob
import json
import os
import subprocess
import sys

SB_KEY_PREFIX = "sb-"


def udid_or_booted() -> str:
    if len(sys.argv) > 2:
        return sys.argv[2]
    out = subprocess.run(
        ["xcrun", "simctl", "list", "devices", "booted", "-j"],
        capture_output=True, text=True, check=True,
    )
    devices = json.loads(out.stdout)["devices"]
    for devs in devices.values():
        for dev in devs:
            if dev["state"] == "Booted":
                return dev["udid"]
    raise SystemExit("no booted simulator")


def manifest_path(udid: str) -> str:
    data = subprocess.run(
        ["xcrun", "simctl", "get_app_container", udid, "host.exp.Exponent", "data"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    hits = glob.glob(
        os.path.join(data, "Documents", "ExponentExperienceData",
                     "*", "*", "RCTAsyncLocalStorage", "manifest.json")
    ) or glob.glob(
        os.path.join(data, "Documents", "ExponentExperienceData",
                     "**", "manifest.json"), recursive=True
    )
    if not hits:
        raise SystemExit(f"no AsyncStorage manifest under {data}")
    return hits[0]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    session = json.load(open(sys.argv[1]))
    ref = session.get("aud")  # not present on the token response
    # The project ref is the subdomain of the Supabase URL; fall back to
    # scanning the manifest for an existing sb-* key (avoids needing .env).
    udid = udid_or_booted()
    path = manifest_path(udid)
    manifest = json.load(open(path))

    key = next((k for k in manifest if k.startswith(SB_KEY_PREFIX)), None)
    if key is None:
        env = {}
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", ".env")) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
        host = env.get("EXPO_PUBLIC_SUPABASE_URL", "").split("//")[-1].split(".")[0]
        key = f"sb-{host}-auth-token"

    manifest[key] = json.dumps(session)
    with open(path, "w") as fh:
        json.dump(manifest, fh)
    print(f"injected {key} into {path} (expires_at={session.get('expires_at')})")


if __name__ == "__main__":
    main()
