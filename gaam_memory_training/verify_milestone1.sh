#!/bin/bash
set -e

echo "=================================="
echo "Milestone 1 Verification Script"
echo "=================================="
echo ""

echo "Test 1: Load safe graph with strict mode"
echo "---"
python scripts/inspect_oracle_graph.py \
  --graph outputs/no_leak_smoke_test/e47becba.graph.json \
  --evidence_pack outputs/no_leak_smoke_test/e47becba.evidence_pack.json \
  --strict > /tmp/milestone1_test1.json
echo "✅ Exit code: $?"
grep -q '"leakage_checked": true' /tmp/milestone1_test1.json && echo "✅ Leakage checked"
grep -q '"integrity_checked": true' /tmp/milestone1_test1.json && echo "✅ Integrity checked"
echo ""

echo "Test 2: Reject old leaky graph"
echo "---"
set +e
python scripts/inspect_oracle_graph.py \
  --graph outputs/longmemeval_s_graph/e47becba.graph.json \
  --strict 2>&1 | grep -q "leakage_detected"
if [ $? -eq 0 ]; then
    echo "✅ Leaky graph correctly rejected"
else
    echo "❌ Leaky graph not rejected"
    exit 1
fi
set -e
echo ""

echo "Test 3: Run all tests"
echo "---"
python -m pytest tests/test_oracle_graph_loader.py tests/test_memory_schema.py -v --tb=line | tail -1
echo ""

echo "=================================="
echo "✅ Milestone 1 Verification Complete"
echo "=================================="
