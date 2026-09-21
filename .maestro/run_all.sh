#!/bin/bash
# Run all Maestro E2E flows
# Individual flows each start from Sign In screen.

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
# Maestro 2.8.0 works fine with unset JAVA_HOME. Only set it when a Homebrew
# OpenJDK is actually installed (this machine has openjdk/openjdk@26, not @17).
if [ -d "/opt/homebrew/opt/openjdk@17" ]; then
  export JAVA_HOME=/opt/homebrew/opt/openjdk@17
  export PATH="$JAVA_HOME/bin:$PATH"
elif [ -d "/opt/homebrew/opt/openjdk" ]; then
  export JAVA_HOME=/opt/homebrew/opt/openjdk
  export PATH="$JAVA_HOME/bin:$PATH"
elif [ -d "/opt/homebrew/opt/openjdk@26" ]; then
  export JAVA_HOME=/opt/homebrew/opt/openjdk@26
  export PATH="$JAVA_HOME/bin:$PATH"
else
  echo "(info) No Homebrew OpenJDK found — leaving JAVA_HOME unset (Maestro 2.8.0 is fine without it)"
fi
if [ -n "${JAVA_HOME:-}" ] && [ ! -x "$JAVA_HOME/bin/java" ]; then
  echo "(warn) JAVA_HOME=$JAVA_HOME has no bin/java — unsetting"
  unset JAVA_HOME
fi
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
"$MAESTRO" test "$DIR/full_app_e2e.yaml" --format junit 2>&1

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
