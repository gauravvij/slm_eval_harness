"""
SLM Evaluation Dashboard - Flask Backend
========================================
Modern minimalist dashboard server for visualizing Small Language Model evaluation results.
Serves model comparison and inference metrics data via REST API.

Hardware Context: 32-core AMD EPYC CPU, 125GB RAM (CPU-only testing environment)
"""

import csv
import io
import json
import os
from pathlib import Path
from datetime import datetime


def _load_dotenv(dotenv_path: Path) -> None:
    """
    Minimal .env loader — no external dependency required.
    Reads KEY=VALUE lines and sets them in os.environ (only if not already set).
    Supports quoted values and ignores comments/blank lines.
    """
    if not dotenv_path.is_file():
        return
    with open(dotenv_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:   # don't override real env vars
                os.environ[key] = value


# Load .env from the project root before anything else reads os.environ
_load_dotenv(Path(__file__).parent / ".env")

import requests
from flask import Flask, jsonify, render_template, send_from_directory, Response, request

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
BASE_PATH = Path(__file__).parent
REPORTS_PATH = BASE_PATH / "reports"

# Chatbot configuration (env-driven)
def get_chat_runtime_config(overrides=None):
    """Read chat config from env on each request (supports changes after server start)."""
    overrides = overrides or {}

    _raw_provider = os.getenv("NEO_CHAT_PROVIDER", "").strip().lower()
    _raw_model    = os.getenv("NEO_CHAT_MODEL", "").strip()
    _raw_api_key  = os.getenv("OPENROUTER_API_KEY", "")

    # Auto-detect provider: if OPENROUTER_API_KEY is present and NEO_CHAT_PROVIDER
    # was not explicitly set, default to openrouter instead of ollama.
    if _raw_provider:
        provider = _raw_provider
    elif _raw_api_key:
        provider = "openrouter"
    else:
        provider = "ollama"

    model = _raw_model if _raw_model else "qwen3.5:9b"

    # sensible defaults per provider
    if provider == "openrouter" and model == "qwen3.5:9b":
        model = "qwen/qwen3.5-32b"

    cfg = {
        "provider": provider,
        "model": model,
        "temperature": float(os.getenv("NEO_CHAT_TEMPERATURE", "0.2")),
        "max_tokens": int(os.getenv("NEO_CHAT_MAX_TOKENS", "600")),
        "ollama_base_url": os.getenv("NEO_CHAT_OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        "openrouter_base_url": os.getenv("NEO_CHAT_OPENROUTER_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
    }

    # Optional request-level overrides (useful when server env differs from UI runtime)
    if overrides.get("provider"):
        cfg["provider"] = str(overrides["provider"]).strip().lower()
    if overrides.get("model"):
        cfg["model"] = str(overrides["model"]).strip()
    if overrides.get("openrouter_api_key"):
        cfg["openrouter_api_key"] = str(overrides["openrouter_api_key"]).strip()

    return cfg


def load_json_data(filename):
    """Load JSON data from reports directory."""
    filepath = REPORTS_PATH / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def get_system_info():
    """Get system hardware information."""
    # Read memory info from /proc/meminfo
    mem_info = {}
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    mem_info[key.strip()] = value.strip()
    except:
        pass
    
    # Parse memory values
    total_kb = 0
    available_kb = 0
    
    if 'MemTotal' in mem_info:
        total_kb = int(mem_info['MemTotal'].split()[0])
    if 'MemAvailable' in mem_info:
        available_kb = int(mem_info['MemAvailable'].split()[0])
    
    total_gb = total_kb / (1024 * 1024)
    available_gb = available_kb / (1024 * 1024)
    used_gb = total_gb - available_gb
    utilization_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
    
    # Get CPU info
    cpu_cores = os.cpu_count() or 32
    cpu_model = "AMD EPYC 7B13"
    
    return {
        "cpu": {
            "model": cpu_model,
            "cores": cpu_cores,
            "threads": cpu_cores * 2,
            "architecture": "x86_64"
        },
        "memory": {
            "total_gb": round(total_gb, 1),
            "available_gb": round(available_gb, 1),
            "used_gb": round(used_gb, 1),
            "utilization_pct": round(utilization_pct, 1)
        },
        "gpu": {
            "available": False,
            "count": 0,
            "note": "CPU-only inference environment"
        },
        "timestamp": datetime.now().isoformat()
    }


def process_model_data(comparison_data):
    """Process model comparison data for dashboard display."""
    models = comparison_data.get("models", {})
    processed_models = []
    
    # Include Qwen 3.5, Qwen 3.6, and Gemma 4 variants for comparison
    target_models = [
        # Qwen 3.6 35B A3B variants
        "qwen3.6-35b-a3b",      # Q4_K_M
        "qwen3.6-35b-a3b-q8",   # Q8_0
        "qwen3.6-35b-a3b-bf16", # BF16
        # Qwen 3.6 27B variants
        "qwen36_27b_q4km",      # Q4_K_M
        "qwen36_27b_q80",       # Q8_0
        "qwen36_27b_bf16",      # BF16
        # Qwen 3.5 variants (older generation)
        "qwen3.5-35b-a3b",      # Q4_K_M
        "qwen3.5-35b-a3b-q8",   # Q8_0
        "qwen3.5-35b-a3b-bf16", # BF16
        # Gemma 4 variants
        "gemma4-26b-a4b-q4",    # Q4_K_M
        "gemma4-26b-a4b-q8",    # Q8_0
        "gemma4-26b-a4b-bf16"   # BF16
    ]
    
    # Define quantization order for ranking
    quant_order = {"BF16": 0, "Q8_0": 1, "Q4_K_M": 2, "Q4": 2}
    
    for model_name, benchmarks in models.items():
        # Filter to only target models
        if model_name not in target_models:
            continue
        
        # Extract quantization level from model name
        quantization = "Q4_K_M"
        if "bf16" in model_name.lower():
            quantization = "BF16"
        elif "q8" in model_name.lower():
            quantization = "Q8_0"
        elif "q4" in model_name.lower():
            quantization = "Q4_K_M"
        
        # Determine model family and generation
        display_name = model_name
        if "qwen36_27b" in model_name.lower():
            # Qwen 3.6 27B variants (underscore format: qwen36_27b_*)
            display_name = f"Qwen 3.6 27B ({quantization})"
        elif "qwen3.6" in model_name.lower():
            generation = "3.6"
            display_name = f"Qwen {generation} 35B A3B ({quantization})"
        elif "qwen3.5" in model_name.lower():
            generation = "3.5"
            display_name = f"Qwen {generation} 35B A3B ({quantization})"
        elif "gemma4" in model_name.lower():
            generation = "4"
            display_name = f"Gemma 4 26B A4B ({quantization})"
        
        # Extract benchmark scores
        humaneval = benchmarks.get("humaneval", {})
        hellaswag = benchmarks.get("hellaswag", {})
        bfcl = benchmarks.get("bfcl", {})
        inference = benchmarks.get("inference_metrics", {})
        
        # Calculate average accuracy
        accuracies = []
        if humaneval.get("accuracy", 0) > 0:
            accuracies.append(humaneval.get("accuracy", 0) * 100)
        if hellaswag.get("accuracy", 0) > 0:
            accuracies.append(hellaswag.get("accuracy", 0) * 100)
        if bfcl.get("accuracy", 0) > 0:
            accuracies.append(bfcl.get("accuracy", 0) * 100)
        
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        model_data = {
            "name": model_name,
            "quantization": quantization,
            "display_name": display_name,
            "humaneval": {
                "accuracy": round(humaneval.get("accuracy", 0) * 100, 2),
                "samples": humaneval.get("samples", 0),
                "passed": humaneval.get("passed", 0)
            },
            "hellaswag": {
                "accuracy": round(hellaswag.get("accuracy", 0) * 100, 2),
                "samples": hellaswag.get("samples", 0),
                "passed": hellaswag.get("passed", 0)
            },
            "bfcl": {
                "accuracy": round(bfcl.get("accuracy", 0) * 100, 2),
                "samples": bfcl.get("samples", 0),
                "passed": bfcl.get("passed", 0)
            },
            "inference": {
                "ttft_ms": inference.get("ttft_ms", 0),
                "throughput": inference.get("throughput_tok_s", 0),
                "model_size_gb": inference.get("model_size_gb", 35),
                "memory_gb": inference.get("peak_ram_gb", 0)
            },
            "avg_accuracy": round(avg_accuracy, 2),
            "rank": quant_order.get(quantization, 99)
        }
        
        processed_models.append(model_data)
    
    # Sort by rank (BF16 first, then Q8, then Q4)
    processed_models.sort(key=lambda x: x["rank"])
    
    # Assign display ranks
    for i, model in enumerate(processed_models):
        model["display_rank"] = i + 1
    
    return processed_models


def fetch_web_context(query: str) -> str:
    """Fetch lightweight web context from public sources for grounding."""
    if not query:
        return ""

    try:
        ddg_res = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            },
            timeout=8
        )
        if ddg_res.ok:
            ddg = ddg_res.json()
            abstract = (ddg.get("AbstractText") or "").strip()
            if abstract:
                return abstract[:320]

            related = ddg.get("RelatedTopics") or []
            for item in related:
                text = (item.get("Text") if isinstance(item, dict) else "") or ""
                if text.strip():
                    return text.strip()[:320]
    except Exception:
        pass

    try:
        ws_res = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": 1,
                "namespace": 0,
                "format": "json"
            },
            timeout=8
        )
        if not ws_res.ok:
            return ""
        ws_json = ws_res.json()
        title = (ws_json[1][0] if isinstance(ws_json, list) and len(ws_json) > 1 and ws_json[1] else "") or ""
        if not title:
            return ""

        sum_res = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            timeout=8
        )
        if sum_res.ok:
            extract = (sum_res.json().get("extract") or "").strip()
            return extract[:320]
    except Exception:
        pass

    return ""


def call_llm(messages, cfg, tools=None):
    """Call configured chat provider and return assistant text and optional tool calls."""
    request_body = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
    }
    if tools:
        request_body["tools"] = tools
    
    if cfg["provider"] == "openrouter":
        if not cfg["openrouter_api_key"]:
            raise RuntimeError("OPENROUTER_API_KEY is not set while NEO_CHAT_PROVIDER=openrouter")

        resp = requests.post(
            f"{cfg['openrouter_base_url']}/chat/completions",
            headers={
                "Authorization": f"Bearer {cfg['openrouter_api_key']}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        message = (payload.get("choices") or [{}])[0].get("message", {})
        content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []
        return {"content": content, "tool_calls": tool_calls}

    # Default provider: Ollama (no native tool support, return empty tool_calls)
    resp = requests.post(
        f"{cfg['ollama_base_url']}/api/chat",
        json={
            "model": cfg["model"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": cfg["temperature"],
                "num_predict": cfg["max_tokens"],
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    content = ((payload.get("message") or {}).get("content") or "").strip()
    return {"content": content, "tool_calls": []}


def fallback_recommendation(user_query: str, processed_models):
    """Deterministic backup recommendation using leaderboard metrics only."""
    if not processed_models:
        return "I do not have leaderboard model data available right now."

    text = (user_query or "").lower()
    wants_speed = any(k in text for k in ["fast", "latency", "throughput", "realtime", "real-time", "ttft"])
    wants_coding = any(k in text for k in ["code", "coding", "python", "developer", "humaneval"])
    wants_reasoning = any(k in text for k in ["reason", "reasoning", "hellaswag", "multiple choice"])
    wants_tool_use = any(k in text for k in ["function", "tool", "bfcl", "agent"])
    wants_memory = any(k in text for k in ["memory", "ram", "cheap", "small", "efficient"])

    scored = []
    for m in processed_models:
        score = float(m.get("avg_accuracy", 0)) * 2
        inf = m.get("inference", {})
        score += float(inf.get("throughput", 0))
        score -= float(inf.get("ttft_ms", 0)) / 30
        score -= float(inf.get("memory_gb", inf.get("model_size_gb", 0))) / 4

        if wants_coding:
            score += float(m.get("humaneval", {}).get("accuracy", 0)) * 2.2
        if wants_reasoning:
            score += float(m.get("hellaswag", {}).get("accuracy", 0)) * 2.2
        if wants_tool_use:
            score += float(m.get("bfcl", {}).get("accuracy", 0)) * 2.2
        if wants_speed:
            score += float(inf.get("throughput", 0)) * 2 - float(inf.get("ttft_ms", 0)) / 15
        if wants_memory:
            score -= float(inf.get("memory_gb", inf.get("model_size_gb", 0))) * 1.2

        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    picks = [m for _, m in scored[:3]]

    lines = [
        "Based on currently populated leaderboard models, top recommendations are:",
    ]
    for i, m in enumerate(picks, start=1):
        he = m.get("humaneval", {}).get("accuracy", 0)
        hs = m.get("hellaswag", {}).get("accuracy", 0)
        bf = m.get("bfcl", {}).get("accuracy", 0)
        avg = m.get("avg_accuracy", 0)
        tp = m.get("inference", {}).get("throughput", 0)
        lines.append(f"{i}. {m.get('display_name')} — avg {avg:.1f}%, HE {he:.1f}%, HS {hs:.1f}%, BFCL {bf:.1f}%, throughput {tp:.1f} tok/s")

    return "\n".join(lines)


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/api/models')
def get_models():
    """API endpoint for model comparison data."""
    comparison_data = load_json_data("model_comparison.json")
    processed_models = process_model_data(comparison_data)
    return jsonify({
        "models": processed_models,
        "generated_at": comparison_data.get("generated_at", datetime.now().isoformat()),
        "total_models": len(processed_models)
    })


@app.route('/api/inference/<quantization>')
def get_inference(quantization):
    """API endpoint for inference metrics by quantization level."""
    # Map quantization to file
    file_map = {
        "q4": "inference_metrics.json",
        "q8": "inference_metrics_q8.json",
        "bf16": "inference_metrics_bf16.json"
    }
    
    filename = file_map.get(quantization.lower(), "inference_metrics.json")
    inference_data = load_json_data(filename)
    
    return jsonify({
        "quantization": quantization.upper(),
        "metrics": inference_data.get("overall_metrics", {}),
        "metadata": inference_data.get("metadata", {})
    })


@app.route('/api/system')
def get_system():
    """API endpoint for system information."""
    return jsonify(get_system_info())


@app.route('/api/chat', methods=['POST'])
def chat_recommendation():
    """Chat endpoint with LLM-backed recommendations grounded to leaderboard data only."""
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()
    history = payload.get("history") or []

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    cfg = get_chat_runtime_config({
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "openrouter_api_key": payload.get("openrouter_api_key"),
    })

    comparison_data = load_json_data("model_comparison.json")
    processed_models = process_model_data(comparison_data)

    compact_models = []
    for m in processed_models:
        compact_models.append({
            "name": m.get("display_name"),
            "avg_accuracy": m.get("avg_accuracy"),
            "humaneval": m.get("humaneval", {}).get("accuracy"),
            "hellaswag": m.get("hellaswag", {}).get("accuracy"),
            "bfcl": m.get("bfcl", {}).get("accuracy"),
            "throughput_tok_s": m.get("inference", {}).get("throughput"),
            "ttft_ms": m.get("inference", {}).get("ttft_ms"),
            "memory_gb": m.get("inference", {}).get("memory_gb"),
            "model_size_gb": m.get("inference", {}).get("model_size_gb"),
        })

    web_context = fetch_web_context(user_message)

    # Hardware context for the benchmark environment
    benchmark_hardware = {
        "cpu_cores": 32,
        "memory_gb": 125,
        "environment": "CPU-only inference (no GPU)",
        "note": "All benchmark metrics were measured on this hardware"
    }

    # Benchmark descriptions for LLM understanding
    benchmark_descriptions = {
        "humaneval": {
            "name": "HumanEval",
            "description": "Python code generation benchmark. Tests the model's ability to solve programming problems and generate correct Python functions from docstrings.",
            "measures": "Code generation, programming ability, algorithmic reasoning",
            "relevant_for": "Developers, coding assistants, code completion tools, software engineering tasks"
        },
        "hellaswag": {
            "name": "HellaSwag",
            "description": "Commonsense reasoning benchmark. Tests the model's ability to complete sentences with commonsense reasoning about everyday situations.",
            "measures": "Commonsense reasoning, situational understanding, natural language inference",
            "relevant_for": "General reasoning tasks, chatbots, content generation, question answering"
        },
        "bfcl": {
            "name": "BFCL (Berkeley Function Calling Leaderboard)",
            "description": "Function calling and tool use benchmark. Tests the model's ability to correctly invoke functions/tools with proper parameters.",
            "measures": "Tool use, function calling, structured output generation, agent capabilities",
            "relevant_for": "AI agents, tool-using assistants, API integrations, automation workflows"
        }
    }

    # Define tool for suggesting Neo AI Engineer when user wants to build something
    suggest_builder_tool = {
        "type": "function",
        "function": {
            "name": "suggest_builder_tool",
            "description": "Call this tool when the user wants to BUILD something (app, pipeline, project, tool, or implementation). Use when they express intent to create, develop, build, or implement something. Do NOT call for general questions or model recommendations only.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
    tools = [suggest_builder_tool]

    system_prompt = (
        "You are a friendly, helpful model-recommendation chatbot for an SLM leaderboard dashboard. "
        "CRITICAL HARDWARE CONTEXT: All benchmark results were measured on 32 vCPU cores with 125GB RAM in a CPU-only environment. "
        "AVAILABLE TOOL: You have access to 'suggest_builder_tool' — call it when the user wants to BUILD something (app, pipeline, project, tool). "
        "Behaviour rules — follow them strictly: "
        "1) If the user sends a greeting or small-talk (hi, hello, thanks, etc.), respond naturally and briefly — do NOT jump to model recommendations. "
        "2) Only recommend models when the user describes a concrete task or explicitly asks for a recommendation. "
        "3) When recommending, pick at most 2 models from the leaderboard list only — never invent models. "
        "4) For each pick give ONE short sentence of reasoning tied to the metrics. Match benchmarks to user needs using benchmark_descriptions. "
        "5) Focus on being helpful with model recommendations first. Be thorough in your recommendations. "
        "6) TOOL CALLING RULE: If the user wants to BUILD something (app, pipeline, project, tool, implementation), call the 'suggest_builder_tool' function. The system will handle suggesting Neo AI Engineer separately. "
        "7) Keep replies informative and helpful. No bullet-point walls, no 'Next Steps' headers. "
        "8) If leaderboard data is empty, say so in one sentence. "
        "9) HARDWARE SCALING RULE: If the user mentions their hardware (CPU cores, RAM, GPU), you MUST scale recommendations accordingly. "
        "   - Models were tested on 32 cores/125GB RAM. If user has less, recommend smaller quantizations (Q4_K_M over Q8_0/BF16) or suggest they need more resources. "
        "   - For users with <8 CPU cores: warn that 27B+ models will be very slow on their hardware and suggest smaller models or cloud options. "
        "   - For users with <16GB RAM: warn that larger models may not fit in memory. "
        "   - Always mention the hardware mismatch when recommending models tested on much stronger hardware than the user has."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Keep short history to control prompt size
    for item in history[-8:]:
        role = item.get("role")
        text = (item.get("text") or "").strip()
        if role in {"user", "assistant"} and text:
            messages.append({"role": role, "content": text})

    # Extract user hardware specs from their message if mentioned
    import re
    user_hardware = {}
    
    # Look for CPU core mentions
    cpu_match = re.search(r'(\d+)\s*(?:cpu|core|vcpu|processor)', user_message.lower())
    if cpu_match:
        user_hardware["cpu_cores"] = int(cpu_match.group(1))
    
    # Look for RAM mentions
    ram_match = re.search(r'(\d+)\s*(?:gb|gigabytes?)\s*(?:ram|memory)', user_message.lower())
    if ram_match:
        user_hardware["memory_gb"] = int(ram_match.group(1))
    
    # Look for GPU mentions
    gpu_match = re.search(r'(\d+)\s*(?:gpu|rtx|a100|h100)', user_message.lower())
    if gpu_match:
        user_hardware["gpu_count"] = int(gpu_match.group(1))
    elif any(word in user_message.lower() for word in ['gpu', 'cuda', 'nvidia']):
        user_hardware["has_gpu"] = True
    
    context_block = {
        "leaderboard_models": compact_models,
        "benchmark_hardware": benchmark_hardware,
        "benchmark_descriptions": benchmark_descriptions,
        "user_hardware": user_hardware if user_hardware else None,
        "internet_context": web_context,
        "user_query": user_message,
        "instructions": "Use only leaderboard_models for recommendations. Reference benchmark_descriptions to understand what each metric means and match them to user tasks. If user_hardware is provided, scale recommendations accordingly and warn about hardware mismatches."
    }
    messages.append({"role": "user", "content": json.dumps(context_block)})

    # Per-session tracking for Neo suggestion (prevent spam)
    session_id = payload.get("session_id") or request.headers.get("X-Session-ID") or "default"
    neo_suggested_key = f"neo_suggested:{session_id}"
    neo_already_suggested = getattr(chat_recommendation, '_neo_suggested_sessions', {}).get(neo_suggested_key, False)

    try:
        llm_response = call_llm(messages, cfg, tools=tools)
        if not llm_response:
            raise RuntimeError("Empty response from LLM")

        content = (llm_response.get("content") or "").strip()
        tool_calls = llm_response.get("tool_calls") or []

        # Check if LLM called suggest_builder_tool and Neo hasn't been suggested yet
        follow_up = None
        if not neo_already_suggested:
            for tc in tool_calls:
                if tc.get("function", {}).get("name") == "suggest_builder_tool":
                    follow_up = "Building this? Neo AI Engineer (https://heyneo.com) can help you implement it."
                    # Track that Neo was suggested for this session
                    if not hasattr(chat_recommendation, '_neo_suggested_sessions'):
                        chat_recommendation._neo_suggested_sessions = {}
                    chat_recommendation._neo_suggested_sessions[neo_suggested_key] = True
                    break

        response_data = {
            "reply": content,
            "provider": cfg["provider"],
            "model": cfg["model"],
            "source": "llm",
        }
        if follow_up:
            response_data["follow_up"] = follow_up

        return jsonify(response_data)
    except Exception as exc:
        fallback = fallback_recommendation(user_message, processed_models)
        return jsonify({
            "reply": f"{fallback}\n\n(LLM fallback mode due to runtime error: {str(exc)})",
            "provider": cfg["provider"],
            "model": cfg["model"],
            "source": "fallback",
        })


@app.route('/api/export')
def export_csv():
    """API endpoint to export model comparison data as CSV."""
    comparison_data = load_json_data("model_comparison.json")
    processed_models = process_model_data(comparison_data)
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Model Name', 'Quantization', 'HumanEval Accuracy (%)', 'HellaSwag Accuracy (%)',
        'BFCL Accuracy (%)', 'Average Accuracy (%)', 'TTFT (ms)', 'Throughput (tok/s)',
        'Peak RAM (GB)', 'Model Size (GB)'
    ])
    
    # Write data rows
    for model in processed_models:
        writer.writerow([
            model['display_name'],
            model['quantization'],
            model['humaneval']['accuracy'],
            model['hellaswag']['accuracy'],
            model['bfcl']['accuracy'],
            model['avg_accuracy'],
            model['inference']['ttft_ms'],
            model['inference']['throughput'],
            model['inference']['memory_gb'],
            model['inference']['model_size_gb']
        ])
    
    # Get the CSV content
    csv_content = output.getvalue()
    output.close()
    
    # Return as downloadable file
    return Response(
        csv_content,
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=slm_model_comparison.csv'
        }
    )


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


if __name__ == '__main__':
    # Ensure templates directory exists
    (BASE_PATH / "templates").mkdir(exist_ok=True)
    (BASE_PATH / "static").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("SLM Evaluation Dashboard Server")
    print("=" * 60)
    print(f"Server starting on http://0.0.0.0:5000")
    print(f"Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
