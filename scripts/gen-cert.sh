#!/usr/bin/env sh
set -eu

CERT_DIR=".cert"
KEY_FILE="$CERT_DIR/localhost-key.pem"
CERT_FILE="$CERT_DIR/localhost-cert.pem"

mkdir -p "$CERT_DIR"

if [ -f "$KEY_FILE" ] && [ -f "$CERT_FILE" ]; then
  exit 0
fi

openssl req \
  -x509 \
  -newkey rsa:2048 \
  -sha256 \
  -nodes \
  -days 3650 \
  -keyout "$KEY_FILE" \
  -out "$CERT_FILE" \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
