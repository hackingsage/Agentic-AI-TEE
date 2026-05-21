#!/usr/bin/env bash
# verify-attestation.sh — Verify an Enclave task's attestation receipt
#
# Usage: ./scripts/verify-attestation.sh <host_url> <task_id> [expected_pcr0]
#
# Example:
#   ./scripts/verify-attestation.sh http://localhost:8000 abc123 expected_pcr0_hex
#
set -euo pipefail

HOST_URL="${1:?Usage: $0 <host_url> <task_id> [expected_pcr0]}"
TASK_ID="${2:?Usage: $0 <host_url> <task_id> [expected_pcr0]}"
EXPECTED_PCR0="${3:-}"

echo "🔒 Enclave Attestation Verifier"
echo "================================"
echo "Host:    $HOST_URL"
echo "Task ID: $TASK_ID"
echo ""

# Fetch attestation receipt
echo "📥 Fetching attestation receipt..."
RECEIPT=$(curl -sf "$HOST_URL/tasks/$TASK_ID/attestation" 2>/dev/null) || {
    echo "❌ Failed to fetch attestation receipt"
    echo "   Make sure the task exists and is completed."
    exit 1
}

echo "$RECEIPT" | python3 -m json.tool 2>/dev/null || echo "$RECEIPT"
echo ""

# Extract PCR values
PCR0=$(echo "$RECEIPT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pcrs',{}).get('0','MISSING'))")
PCR1=$(echo "$RECEIPT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pcrs',{}).get('1','MISSING'))")
PCR2=$(echo "$RECEIPT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pcrs',{}).get('2','MISSING'))")

echo "📋 PCR Values:"
echo "   PCR[0] (image):  ${PCR0:0:32}..."
echo "   PCR[1] (kernel): ${PCR1:0:32}..."
echo "   PCR[2] (app):    ${PCR2:0:32}..."
echo ""

# Check for debug mode (all-zero PCRs)
ZERO_96="000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
if [ "$PCR0" = "$ZERO_96" ]; then
    echo "⚠️  WARNING: PCR[0] is all zeros — enclave is in DEBUG MODE"
    echo "   Data protection is NOT active in debug mode!"
    exit 2
fi

# Verify PCR[0] if expected value provided
if [ -n "$EXPECTED_PCR0" ]; then
    if [ "$PCR0" = "$EXPECTED_PCR0" ]; then
        echo "✅ PCR[0] matches expected image hash"
    else
        echo "❌ PCR[0] MISMATCH"
        echo "   Expected: ${EXPECTED_PCR0:0:32}..."
        echo "   Actual:   ${PCR0:0:32}..."
        echo "   The enclave image may have been tampered with!"
        exit 3
    fi
else
    echo "ℹ️  No expected PCR[0] provided — skipping image verification"
    echo "   Provide the expected hash as the third argument for full verification."
fi

echo ""
echo "✅ Attestation receipt retrieved successfully"
echo "   Task hash: $(echo "$RECEIPT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('task_hash','N/A')[:32])")..."
echo ""
echo "🔗 For full verification, compare PCR values against:"
echo "   https://github.com/your-org/enclave/releases/latest/pcrs.json"
