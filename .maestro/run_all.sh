#!/bin/bash
# Run all Maestro E2E flows
# Individual flows each start from Sign In screen.

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH=$JAVA_HOME/bin:$PATH
MAESTRO="$HOME/.maestro/bin/maestro"

echo "=== Ensuring app is running ==="
xcrun simctl terminate booted org.name.AIWardrobe 2>/dev/null || true
sleep 2

echo "=== Fresh launch ==="
xcrun simctl launch booted org.name.AIWardrobe > /dev/null 2>&1
sleep 30

echo "=== Shake device to reload Metro bundle ==="
xcrun simctl shake booted 2>/dev/null || true
sleep 5

echo "=== E2E Full Flow (runs first while app is fresh) ==="
"$MAESTRO" test "$DIR/e2e_full_flow.yaml" --format junit 2>&1

echo ""
echo "=== Individual Flows ==="
"$MAESTRO" test \
  "$DIR/forgot_password_flow.yaml" \
  "$DIR/auth_signin_validation.yaml" \
  "$DIR/auth_signin_wrong_credentials.yaml" \
  "$DIR/smoke.yaml" \
  "$DIR/closet_flow.yaml" \
  "$DIR/navigation_flow.yaml" \
  --format junit 2>&1
