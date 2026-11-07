STOP the current unrelated quality expansion for now.

The new task is narrowly:

```text
FIX SHOE ROTATION
```

Do not redesign the rest of Personal Stylist.

# SOURCE

Continue on:

```text
branch:
stylist-four-slot-recovery
```

Last pushed GitHub HEAD:

```text
26a940f
```

Before changing anything:

```bash
git status
git rev-parse HEAD
```

There may be additional uncommitted quality work after `26a940f`.

DO NOT discard it.

If the working tree is dirty, checkpoint those changes safely before starting the shoe-rotation fix.

---

# STABLE CONTRACT — DO NOT REGRESS

Every successful generation must still produce:

```text
3 cards
×
4 items:

outerwear
top
bottom
shoes
```

Keep existing:

```text
style quality
occasion quality
weather handling
budget handling
physical product identity
history semantics
rotation scope
request IDs
```

Do not weaken quality simply to force different shoes.

---

# ACTUAL BUG

Clothing changes between repeated Generate presses, but shoes are not rotating enough / appear to stay the same.

We need the shoe slot to obey the same recency logic as the other slots.

Example current problem:

```text
G1 shoes:
A B C

G2 shoes:
A B C

G3 shoes:
A B C
```

This is WRONG if other valid shoes exist.

---

# FIRST: ESTABLISH THE REAL SHOE POOL

For the exact request:

```text
New clothing
Old Money
Date Night
No anchor
```

capture:

```text
all catalog shoes fetched
↓
shoes rejected before style/quality
↓
shoes rejected by refined-date rules
↓
final admissible physical shoes
```

For EVERY rejected shoe report:

```text
product:
physical product key:
rejection reason:
```

We currently believe approximately:

```text
5 physical shoes
```

survive the refined-date rules.

PROVE this.

Do not assume it.

---

# CHECK FOR FALSE NEGATIVES

If the catalog contains more genuinely appropriate Old Money / Date Night shoes but they are rejected by an overly broad rule:

fix that classification/filter.

Examples that may be appropriate depending on actual product:

```text
loafers
derbies
oxfords
refined leather shoes
monk straps
dress boots where appropriate
```

Do NOT blindly add these categories.

Inspect actual catalog items.

Do NOT admit rugged, sporty, technical or obviously inappropriate shoes just to increase inventory.

---

# MATHEMATICAL ROTATION RULE

If:

```text
N = number of valid physical shoes
K = 3 shoes required per deck
```

then the theoretical minimum consecutive overlap is:

```text
minimumOverlap = max(0, 2*K - N)
```

Examples:

```text
N >= 6
→ expected previous-deck shoe overlap = 0

N = 5
→ minimum unavoidable overlap = 1

N = 4
→ minimum unavoidable overlap = 2

N = 3
→ overlap = 3
```

The application should achieve this minimum whenever valid outfit combinations allow it.

---

# CURRENT CASE — 5 VALID SHOES

If there are exactly five valid shoes:

```text
A B C D E
```

and every generation needs three shoes:

Correct possible sequence:

```text
G1:
A B C

G2:
D E A

G3:
B C D

G4:
E A B

G5:
C D E
```

The exact order does not matter.

The properties DO matter:

```text
every next deck uses the two shoes absent from the previous deck

only ONE previous-deck shoe repeats

the repeated shoe itself rotates over time
```

WRONG:

```text
G1 A B C
G2 A B C
G3 A B C
```

Also wrong:

```text
G1 A B C
G2 A D E
G3 A B C
G4 A D E
```

because shoe A is unfairly pinned and repeated every generation.

---

# LRU / FAIRNESS RULE

Once reuse becomes mathematically necessary:

choose the eligible shoe with the OLDEST last-use.

Do not simply choose:

```text
highest style score every time
first DB row
lowest ID
same favorite shoe
```

Required priority among shoes that already pass all hard quality rules:

```text
1. shoe not in immediately previous deck
2. shoe least recently used in older history
3. shoe with fewer total recent appearances
4. outfit quality/ranking as tie-breaker
```

Quality is a prerequisite.

Do not select a bad shoe just because it is fresh.

---

# IMPORTANT: QUALITY FILTER FIRST, ROTATION SECOND

Correct pipeline:

```text
catalog shoes
↓
hard style/occasion/weather/quality validation
↓
ADMISSIBLE SHOE POOL
↓
rotation scheduling
↓
outfit combination quality
```

Do not let rotation resurrect a shoe that failed the hard quality gate.

---

# PREVIOUS DECK MUST HAVE STRONGEST PENALTY

Given:

```text
previousDeckShoes = A B C

validPool = A B C D E
```

G2 must first take:

```text
D
E
```

Then only one shoe from:

```text
A B C
```

may be restored because three slots are required.

Do NOT restore A/B/C before D/E are used.

---

# OLDER HISTORY MUST BE BETTER THAN PREVIOUS DECK

Suppose:

```text
G1 = A B C
G2 = D E A
```

Before G3:

```text
previous deck:
D E A

older history:
B C
```

G3 should strongly prefer:

```text
B
C
```

before reusing:

```text
D
E
A
```

Then choose only the minimum required previous-deck shoe.

This is the key behavior.

---

# AUDIT FINAL OUTFIT SELECTION

Do not only inspect the catalog trimming stage.

The bug may occur later because complete-outfit ranking repeatedly favors the same shoes.

Trace:

```text
admissible shoes
↓
candidate combinations
↓
quality scoring
↓
diversity selection
↓
final 3 outfits
↓
displayed shoes
```

At each stage log shoe IDs/product keys.

We need to identify exactly where alternative shoes disappear.

---

# LEXICOGRAPHIC DECK OBJECTIVE

For candidates that ALREADY pass the hard quality threshold, choose the final deck using this priority:

```text
1. minimize previous-deck shoe overlap
2. minimize previous-deck overlap in all other slots
3. minimize older-history reuse / prefer LRU
4. maximize outfit quality
5. maximize visual variety
```

Do NOT let a tiny quality-score advantage cause:

```text
same shoes every generation
```

when another candidate is still above the required quality threshold.

---

# FINAL SHOE AUDIT

Before accepting a deck calculate:

```text
validPhysicalShoesAvailable
previousDeckShoes
selectedShoes
freshShoesAvailable
previousShoeOverlap
minimumPossibleOverlap
```

Then enforce:

```ts
previousShoeOverlap <= minimumPossibleOverlap
```

whenever the available valid combinations permit it.

For the current five-shoe case:

```text
valid shoes = 5
selected per deck = 3

minimumPossibleOverlap = 1
```

If final deck has:

```text
2 or 3 previous shoes
```

while all five are valid:

REJECT/RESELECT the deck.

---

# DO NOT HARD-CODE CURRENT FIVE SHOE IDS

The algorithm must work for:

```text
3 shoes
4 shoes
5 shoes
6 shoes
10 shoes
50 shoes
```

Do not special-case current product names.

---

# TESTS

Add focused tests.

## Case A — six shoes

```text
valid shoes:
A B C D E F

G1:
A B C

G2:
D E F
```

Expected overlap:

```text
0
```

---

## Case B — five shoes

```text
A B C D E
```

Every generation uses 3.

Expected consecutive overlap:

```text
exactly 1
```

Never 2 or 3.

Over multiple generations all five shoes must receive usage.

No shoe should remain permanently pinned.

---

## Case C — four shoes

Expected minimum consecutive overlap:

```text
2
```

but the non-previous shoe must always be used.

---

## Case D — three shoes

Expected:

```text
all 3 repeat
```

because unavoidable.

---

## Case E — duplicate catalog rows

Two DB rows representing the same physical shoe must count as ONE shoe.

Use:

```text
ID
normalized name
normalized image
```

physical identity.

---

# REAL ACCEPTANCE TEST

Use the real app:

```text
Old Money
Date Night
New clothing
No anchor
```

Press Generate five times.

Capture:

```text
G1 shoes:
G2 shoes:
G3 shoes:
G4 shoes:
G5 shoes:
```

For each transition report:

```text
G1→G2 overlap:
G2→G3 overlap:
G3→G4 overlap:
G4→G5 overlap:
```

Also report:

```text
valid shoe pool size:
theoretical minimum overlap:
actual overlap:
```

---

# EXPECTED RESULT IF POOL = 5

For every consecutive transition:

```text
actual shoe overlap = 1
```

not:

```text
2
3
```

And across the run:

```text
all 5 shoes used
```

with fair/LRU reuse.

---

# CLOTHING ROTATION MUST KEEP WORKING

Regression-check:

```text
outerwear
top
bottom
```

They should continue rotating as they do now.

Do not solve shoes by breaking clothing rotation.

---

# STOP THE IMAGE-QUALITY RABBIT HOLE FOR THIS TASK

Do not continue building a large visual-classification system right now.

The current task is:

```text
shoe rotation
```

Only inspect image evidence if it directly determines whether a SHOE should belong to the valid shoe pool.

Do not widen this patch into general shirt/jacket image understanding.

We can continue that separately after rotation is correct.

---

# FINAL REPORT

Return:

```text
SOURCE
- starting HEAD:
- final HEAD:
- Edge version:

SHOE POOL
- catalog shoes:
- physical unique shoes:
- admitted refined-date shoes:
- rejected shoes + reasons:

ROOT CAUSE
- where alternative shoes were lost:
- why same shoes kept winning:

ROTATION ALGORITHM
- minimum-overlap rule:
- LRU rule:
- previous-deck handling:
- older-history handling:

REAL TEST

G1 shoes:
1.
2.
3.

G2 shoes:
1.
2.
3.
overlap G1→G2:

G3 shoes:
1.
2.
3.
overlap G2→G3:

G4 shoes:
1.
2.
3.
overlap G3→G4:

G5 shoes:
1.
2.
3.
overlap G4→G5:

- valid shoe pool:
- theoretical minimum:
- actual maximum observed overlap:
- all valid shoes used: yes/no

REGRESSION
- 3 cards: PASS/FAIL
- 4 items: PASS/FAIL
- outerwear rotation: PASS/FAIL
- top rotation: PASS/FAIL
- bottom rotation: PASS/FAIL
- quality preserved: PASS/FAIL

FINAL
- shoe rotation: PASS/FAIL
```

Do not claim PASS merely because shoes changed once.

PASS means the real app consistently achieves the mathematically minimum previous-deck shoe overlap without admitting lower-quality shoes.