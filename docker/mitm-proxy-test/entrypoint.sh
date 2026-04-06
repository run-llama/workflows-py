#!/usr/bin/env bash
set -euo pipefail

# Generate a local CA if not present and install it into the OS trust store
# mitmproxy provides its own CA in ~/.mitmproxy
mkdir -p /root/.mitmproxy

# Start and stop mitmproxy once to ensure CA exists
if [ ! -f /root/.mitmproxy/mitmproxy-ca-cert.pem ]; then
  # Run mitmdump briefly to generate the CA, then kill it
  timeout 2s mitmdump >/dev/null 2>&1 || true
fi

CA_PEM=/root/.mitmproxy/mitmproxy-ca-cert.pem

if [ -f "$CA_PEM" ]; then
  echo "Installing mitmproxy CA into system trust store..."
  cp "$CA_PEM" /usr/local/share/ca-certificates/mitmproxy-ca.crt
  update-ca-certificates
else
  echo "mitmproxy CA not found; continuing without installing"
fi

PORT=${PORT:-8080}

# Run mitmdump (non-interactive) in the background
# mitmdump logs to stdout by default
if [ -n "${TARGET:-}" ]; then
  echo "Starting mitmdump in reverse proxy mode to $TARGET on port $PORT..."
  mitmdump --mode reverse:"$TARGET" --listen-port "$PORT" &
else
  echo "Starting mitmdump on port $PORT..."
  mitmdump --listen-port "$PORT" &
fi

MITM_PID=$!
echo "mitmdump started with PID $MITM_PID"

export HTTPS_PROXY=http://localhost:$PORT
# Drop into a shell, or run any command passed as arguments
if [ $# -eq 0 ]; then
  exec bash
else
  exec "$@"
fi
