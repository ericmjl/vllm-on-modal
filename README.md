# vLLM on Modal

vLLM inference server deployed on [Modal](https://modal.com) with GPU memory snapshot support for fast cold starts.

## Why vLLM instead of Ollama?

Ollama runs as a subprocess, so its GPU state isn't captured by Modal's memory snapshots. vLLM supports [sleep mode](https://docs.vllm.ai/en/stable/features/sleep_mode/) which works with Modal's GPU snapshots, reducing cold starts from minutes to seconds.

## Deploy

```bash
# Deploy to production
pixi run deploy

# Deploy to test environment (used by CI on PRs)
pixi run deploy-test
```

## Test locally

```bash
# Run ephemeral server and test it
pixi run test
```

## Use as OpenAI-compatible API

The deployed endpoint is fully OpenAI-compatible. Point any OpenAI SDK client at the Modal URL:

```bash
curl https://ericmjl--vllm-service-vllmserver-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## GPU Snapshot Architecture

1. **Snapshot build** (`@modal.enter(snap=True)`): vLLM starts, model loads into GPU, warmup runs, then server enters sleep mode (weights offloaded to CPU, KV cache cleared). Modal snapshots this state.
2. **Restore** (`@modal.enter(snap=False)`): Server wakes from sleep, model weights moved back to GPU. Much faster than a full cold start because JIT compilation artifacts and the process state are already in memory.

## Model

Currently serving **Qwen3.6-35B-A3B GPTQ-Int4** (`palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`) on an **L40S GPU** (48GB).

To change the model, update `MODEL_NAME` in `endpoint.py` and redeploy.
