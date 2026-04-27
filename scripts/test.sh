#!/bin/bash

# Run a smoke test: serve a model, check golden output.
run_smoke_test() {
  local model="$1"
  local revision="$2"
  local prompt="$3"
  local expected="$4"
  shift 4
  local extra_args=("$@")

  section "Smoke test: $model"

  VLLM_VULKAN_MEMORY_FRACTION=0.8 \
    vllm serve "$model" --revision "$revision" --max-model-len 512 --max-num-batched-tokens 64 ${extra_args[@]+"${extra_args[@]}"} &

  local vllm_pid=$!

  echo "Waiting for vLLM to start..."
  local health_url="http://localhost:8000/health"
  if ! curl --retry 30 --retry-delay 10 --retry-all-errors -s "$health_url" > /dev/null; then
    echo "vLLM failed to start."
    kill $vllm_pid
    exit 1
  fi

  echo "Model loaded successfully!"

  local response
  response=$(curl -s -X POST "http://localhost:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$model\",
      \"prompt\": \"$prompt\",
      \"temperature\": 0,
      \"max_tokens\": 10
    }")

  if ! echo "$response" | grep -q '"choices"'; then
    echo "Completions test failed. Response:"
    echo "$response"
    kill $vllm_pid
    exit 1
  fi

  local actual
  actual=$(echo "$response" | python3 -c "import sys,json; print(json.loads(sys.stdin.read(), strict=False)['choices'][0]['text'])")

  if [ "$actual" != "$expected" ]; then
    echo "Golden comparison FAILED"
    echo "  expected: '$expected'"
    echo "  actual:   '$actual'"
    kill $vllm_pid
    exit 1
  fi

  echo "Smoke test passed!"

  kill $vllm_pid 2>/dev/null
  wait $vllm_pid 2>/dev/null || true
}

smoke_tests() {
  # Basic smoke test with Qwen3-0.6B
  run_smoke_test \
    "Qwen/Qwen3-0.6B" \
    "c1899de289a04d12100db370d81485cdf75e47ca" \
    "The capital of France is" \
    " Paris. The capital of Italy is Rome. The"
}

installs() {
  section "Installing vllm"
  ./install.sh
}

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  setup_dev_env

  installs
  # shellcheck source=/dev/null
  source .venv-vllm-vulkan/bin/activate

  section "Verifying package import"
  python -c "import vllm_vulkan; print('vllm_vulkan imported successfully')"

  section "Checking Vulkan availability"
  python -c "
from vllm_vulkan.platform import VulkanPlatform
available = VulkanPlatform.is_available()
print(f'Vulkan available: {available}')
if available:
    print(f'Device count: {VulkanPlatform.get_device_count()}')
    print(f'Device name: {VulkanPlatform.get_device_name()}')
"

  if python -c "from vllm_vulkan.platform import VulkanPlatform; exit(0 if VulkanPlatform.is_available() else 1)" 2>/dev/null; then
    smoke_tests
  else
    echo "No Vulkan device found; skipping smoke tests."
  fi

  section "Running unit tests"
  pytest -m "not slow" tests/python/ -v --tb=short
}

main "$@"
