# Simulator evidence

The final client source is `033a21d41d3ce729f40f11367696afa0673da9c8`, served by Metro on port 8082 from the isolated running-repair worktree. The native host is Expo Go. Hosted `generate-outfits` was verified as v179; no deployment occurred.

- [Original duplicated Uzbek copy](before-duplicated-copy.png) was reproduced on the older dirty worktree.
- [Final Work screenshot](work.png) shows the client SHA and a real four-piece result; all three displayed looks and product IDs are in [simulator events](simulator-events.json).
- [Earlier Casual screenshot](casual.png) and the recorded Travel/Formal runs belong to earlier client revisions; they must not be presented as final-build verification.
- Two consecutive Work decks on `b66f76460d72` had 24 placements, 24 distinct normalized product IDs and zero overlap. Account-wide clothing recency was added subsequently.
- [Native persisted history](persisted-history.json) records the final Work request before the explicit process restart.

**Verification is incomplete:** the Mac locked after that restart. Computer use reported it could not unlock the Mac automatically. Final-build Casual, Travel, Classic + Formal and Casual + Formal checks remain pending manual unlock. Expo Go also exhibited a native Hermes crash during development reloads; cold launches were used for verification.

[Versioned Edge artifact and rollback plan](ROLLOUT.md) remain unexecuted. Client and Edge TypeScript checks and focused regressions passed.
