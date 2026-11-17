# Restore verified v193 generate-outfits source

`v193-source.tar.gz` contains the original self-contained `index.ts` downloaded
from ACTIVE v193 before this patch. SHA-256:
`05ab555c747f9b5b65f99da24fc2bcb99ca16b95f19675eb48b8842e1c46c2a7`.
It exactly matched the v193 source manifest and a fresh bundle of the v193 branch.
Archive and input hashes are recorded in `manifest.json`. Existing v192 artifacts
and the full v193 Git history are preserved.

From this worktree, with a management credential supplied securely:

```
SUPABASE_ACCESS_TOKEN_FILE=/secure/path/to/token node scripts/validation/styling-v194-release.cjs rollback
```

The command checks the active deployment version, fingerprint and original source,
verifies and extracts the archive, and deploys only generate-outfits with the prior
`verify_jwt=false` configuration (the function enforces authentication internally).
It refuses to overwrite a newer release. It verifies ACTIVE status, downloaded
source integrity, other functions unchanged, OPTIONS 200, and unauthenticated and
invalid-token POST 401. Supabase assigns a new monotonically increasing version;
the restored source is v193. No schema or user-data changes are made.
