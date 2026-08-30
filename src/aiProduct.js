// The AI product under test. This is the ONLY place you swap Ollama <-> your real API.
// It also measures the per-call operational data (latency, tokens, retries, failure).

const config = require("../config");

function estimateTokens(text) {
  // Rough fallback when the provider doesn't return token counts (~4 chars/token).
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

// --- Ollama product implementation ---------------------------------------
async function callOllama(inputText) {
  const res = await fetch(`${config.ollama.baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: config.ollama.model,
      stream: false,
      options: { temperature: 0 },
      messages: [
        { role: "system", content: config.systemPrompt },
        { role: "user", content: inputText },
      ],
    }),
  });
  if (!res.ok) {
    throw new Error(`Ollama HTTP ${res.status}: ${await res.text()}`);
  }
  const data = await res.json();
  const output = data?.message?.content ?? "";
  return {
    output,
    // Ollama returns real token counts on the chat response.
    inputTokens: data?.prompt_eval_count ?? estimateTokens(config.systemPrompt + inputText),
    outputTokens: data?.eval_count ?? estimateTokens(output),
  };
}

// --- Your real product API implementation --------------------------------
// EDIT the request body and response mapping to match your API contract.
async function callRealApi(inputText) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.api.timeoutMs);
  try {
    const res = await fetch(config.api.url, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(config.api.apiKey ? { Authorization: `Bearer ${config.api.apiKey}` } : {}),
      },
      // >>> Adjust this payload to your product's expected request shape <<<
      body: JSON.stringify({ model: config.api.model, input: inputText }),
    });
    if (!res.ok) {
      throw new Error(`API HTTP ${res.status}: ${await res.text()}`);
    }
    const data = await res.json();
    // >>> Adjust these paths to your product's response shape <<<
    const output = data?.output ?? data?.text ?? data?.message?.content ?? "";
    return {
      output,
      inputTokens: data?.usage?.input_tokens ?? estimateTokens(inputText),
      outputTokens: data?.usage?.output_tokens ?? estimateTokens(output),
    };
  } finally {
    clearTimeout(timer);
  }
}

function invokeProduct(inputText) {
  return config.AI_SOURCE === "api" ? callRealApi(inputText) : callOllama(inputText);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Public: call the product with retry + operational instrumentation.
async function callAIProduct(inputText) {
  const start = Date.now();
  let retries = 0;
  let lastError = null;

  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    try {
      const { output, inputTokens, outputTokens } = await invokeProduct(inputText);
      const latencyMs = Date.now() - start;
      const inputCost = (inputTokens / 1000) * config.pricing.inputPer1kTokens;
      const outputCost = (outputTokens / 1000) * config.pricing.outputPer1kTokens;
      return {
        output,
        failed: false,
        latencyMs,
        inputTokens,
        outputTokens,
        totalTokens: inputTokens + outputTokens,
        costPerQuery: Number((inputCost + outputCost).toFixed(6)),
        retries,
        error: null,
      };
    } catch (err) {
      lastError = err;
      if (attempt < config.maxRetries) {
        retries++;
        await sleep(config.retryBackoffMs * (attempt + 1));
      }
    }
  }

  // All attempts failed.
  return {
    output: "",
    failed: true,
    latencyMs: Date.now() - start,
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    costPerQuery: 0,
    retries,
    error: String(lastError && lastError.message ? lastError.message : lastError),
  };
}

module.exports = { callAIProduct };
