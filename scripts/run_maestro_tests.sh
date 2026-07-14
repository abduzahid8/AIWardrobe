#!/bin/bash
# AIWardrobe Maestro E2E Test Runner
# Usage: ./run_maestro_tests.sh [--no-build]
set -euo pipefail

JAVA_HOME=/opt/homebrew/opt/openjdk@17
export JAVA_HOME PATH=$JAVA_HOME/bin:$PATH
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MAESTRO="$HOME/.maestro/bin/maestro"
BUNDLE_ID="org.name.AIWardrobe"
TEST_OUTPUT_DIR="$PROJECT_DIR/maestro_test_output"

# Parse args
NO_BUILD=false
for arg in "$@"; do
  case "$arg" in
    --no-build) NO_BUILD=true ;;
  esac
done

echo "========================================"
echo " AIWardrobe Maestro E2E Test Runner"
echo "========================================"
echo ""

# 1. Build if needed
if [ "$NO_BUILD" = false ]; then
  echo "[1/5] Building app..."
  cd "$PROJECT_DIR"
  npx expo run:ios --configuration Debug 2>&1 | tail -5
  echo "  Build complete."
else
  echo "[1/5] Skipping build."
fi

# 2. Kill stale processes
echo "[2/5] Cleaning up..."
pkill -f maestro 2>/dev/null || true
sleep 3
echo "  Clean."

# 3. Start Metro
echo "[3/5] Starting Metro..."
cd "$PROJECT_DIR"
npx expo start --port 8081 2>&1 &
METRO_PID=$!
sleep 15
echo "  Metro running (PID=$METRO_PID)."

# 4. Launch app on simulator
echo "[4/5] Launching app on simulator..."
xcrun simctl terminate booted "$BUNDLE_ID" 2>/dev/null || true
sleep 2
PID=$(xcrun simctl launch booted "$BUNDLE_ID" 2>&1 | grep -oE '[0-9]+$' || echo "0")
echo "  App launched (PID=$PID)."
sleep 10

# 5. Create output directory
mkdir -p "$TEST_OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$TEST_OUTPUT_DIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"
echo "  Output: $RUN_DIR"

# 6. Run Maestro test flows
echo "[5/5] Running Maestro tests..."
PASSED=0
FAILED=0
for flow in "$PROJECT_DIR/.maestro"/*.yaml; do
  fname=$(basename "$flow" .yaml)
  # Skip internal/helper files
  [[ "$fname" == e2e_full_flow* ]] && continue

  echo "  -> $fname..."
  if "$MAESTRO" test "$flow" --no-reinstall-driver 2>&1 | tee "$RUN_DIR/${fname}.log" | grep -q "Passed"; then
    echo "    PASSED"
    PASSED=$((PASSED+1))
  else
    echo "    FAILED (see log)"
    FAILED=$((FAILED+1))
  fi

  # Collect screenshots from project root
  for img in "$PROJECT_DIR"/*_tab.png "$PROJECT_DIR"/screenshot-*.png; do
    [ -f "$img" ] && mv "$img" "$RUN_DIR/" 2>/dev/null || true
  done
done

# Summary
echo ""
echo "========================================"
echo " Results: $PASSED passed, $FAILED failed"
echo " Output: $RUN_DIR"
echo "========================================"

# Cleanup
kill $METRO_PID 2>/dev/null || true

exit $FAILED
