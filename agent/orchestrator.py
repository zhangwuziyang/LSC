"""LLM orchestrator: the model plans the analysis and calls the session steps as tools.
Claude (Anthropic API) is the default; DeepSeek runs through its OpenAI-compatible API
and Qwen (千问 AI 平台) through its Anthropic-compatible API.

The model never computes anything. Every number in the final report is produced by the
engine and checked by Gate G5; a gate that blocks is reported back as the tool result,
and the model has to fix the input (for example drop a post-treatment covariate) or stop.

    from agent.orchestrator import run_agent
    run = run_agent(df, "妥投天数对 GMV 有什么影响？12 天以上的用户缩短到 10 天能带来多少增益？")
    print(run.final_text)
"""
import json
from dataclasses import dataclass, field

from .session import AnalysisSession, StepError

MODEL = "claude-opus-5-5"

SYSTEM = """你是一名因果推断分析助手，负责回答"物流妥投天数对用户 GMV 的影响"这类问题。
你只能通过工具完成分析，所有数字都由工具计算，你不做任何计算，也不估算。

固定流程：
1. 用 list_metrics / lookup_metric 了解可用变量，确认处理变量、结果变量和候选混淆变量。
2. build_dataset：选择混淆变量。只能用干预前（period = pre）测量的变量。如果 G1 阻断，按提示删掉违规变量后重新调用。
3. fit_gps → estimate_dose_response → run_placebo，按顺序调用。
4. render_report：传入一段简短的中文解读（narrative）。解读里只能出现工具返回过的数字；G5 不通过时，删掉或改正出错的数字后重新调用。

解读规则：
- 某个 Gate 降级时，必须在解读里说明这一点，以及它对结论的影响。
- 可信度为"外推"的剂量点和增益格子不能作为结论。
- placebo 未通过时，以敏感性区间为准，不报点估计结论。
- 拐点 CI 宽于 3 天时，不对拐点下结论。
- 用户问到数据无法回答的问题（例如超出支持范围的剂量），直接说明无法回答及原因。

render_report 通过后，用两三句话向用户总结结论，并附上报告。"""


def _schema(props, required=None):
    return {"type": "object", "properties": props, "required": required if required is not None else list(props),
            "additionalProperties": False}


TOOLS = [
    {"name": "list_metrics", "description": "列出语义层中的所有变量（列名、名称、角色、测量时期）。",
     "input_schema": _schema({})},
    {"name": "lookup_metric", "description": "按关键词或列名查询一个变量的定义、角色和测量时期。",
     "input_schema": _schema({"query": {"type": "string"}})},
    {"name": "build_dataset",
     "description": "用选定的混淆变量构建分析表，并运行 G1 时序检查。干预期或之后测量的变量会被阻断。",
     "input_schema": _schema({"covariates": {"type": "array", "items": {"type": "string"}}})},
    {"name": "fit_gps", "description": "估计 GPS 和双稳健剂量响应（耗时数分钟），运行 G2 平衡检查。",
     "input_schema": _schema({})},
    {"name": "estimate_dose_response",
     "description": "返回相对参照天数的剂量响应曲线、拐点和增益矩阵，运行 G3 重叠检查。"
                    "from_days 是当前妥投天数，to_days 是目标天数。",
     "input_schema": _schema({"from_days": {"type": "array", "items": {"type": "integer"}},
                              "to_days": {"type": "array", "items": {"type": "integer"}}})},
    {"name": "run_placebo", "description": "运行 placebo 平行趋势检验（G4）和敏感性分析。",
     "input_schema": _schema({})},
    {"name": "render_report",
     "description": "生成最终报告，并对报告和 narrative 中的每个数字做 G5 核对。",
     "input_schema": _schema({"narrative": {"type": "string"}})},
]
for t in TOOLS:
    t["strict"] = True


@dataclass
class AgentRun:
    final_text: str
    trace: list = field(default_factory=list)
    session: AnalysisSession = None
    stop_reason: str = None
    turns: int = 0
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


def _execute(session, name, args):
    fn = getattr(session, name, None)
    if name not in {t["name"] for t in TOOLS} or fn is None:
        return {"error": f"unknown tool {name}"}, True
    try:
        return fn(**args), False
    except StepError as e:
        return {"error": str(e)}, True
    except Exception as e:  # engine failure: report it to the model instead of crashing the run
        return {"error": f"{type(e).__name__}: {e}"}, True


def _call_tool(session, run, turn, name, args):
    out, is_error = _execute(session, name, args)
    shown = out.get("gate") if isinstance(out, dict) else None
    true = session.gates.get(shown["gate"]) if shown else None
    run.trace.append({"turn": turn, "tool": name, "input": args, "is_error": is_error, "gate": shown,
                      # the real verdict, which differs from `gate` when gates are switched off
                      "gate_true": true.to_dict() if true is not None else None})
    return json.dumps(out, ensure_ascii=False), is_error


def run_agent(df, question, client=None, session_kw=None, max_turns=25, effort="high",
              provider="anthropic", model=None):
    """provider: "anthropic" (Claude, default) or a key of PROVIDERS ("deepseek", "qwen")."""
    session = AnalysisSession(df, **(session_kw or {}))
    run = AgentRun(final_text="", session=session)
    if provider == "anthropic":
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        return _loop_anthropic(run, question, client, max_turns, model or MODEL, claude=True, effort=effort)
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider}")
    cfg = provider_config(provider)
    client = client or make_client(provider)
    if cfg["style"] == "anthropic":
        return _loop_anthropic(run, question, client, max_turns, model or cfg["model"], claude=False)
    return _loop_openai_compat(run, question, client, max_turns, model or cfg["model"])


# Third-party Anthropic-compatible endpoints get plain tool definitions and none of the
# Claude-only request fields (beta headers, server-side fallback, effort).
PLAIN_TOOLS = [{k: v for k, v in t.items() if k != "strict"} for t in TOOLS]


def _openai_tools(tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


def _default_execute(run):
    return lambda turn, name, args: _call_tool(run.session, run, turn, name, args)


FINAL_PROMPT = "已达到工具调用上限。请不要再调用工具，基于以上结果直接用中文给出最终结论。"


def _loop_anthropic(run, question, client, max_turns, model, claude, effort="high",
                    system=SYSTEM, tools=None, execute=None, force_final=False):
    """Messages-API tool loop. `system`, `tools` and `execute(turn, name, args) -> (content,
    is_error)` default to the analysis agent; the code baseline passes its own.
    With force_final, the last turn forbids tool calls and asks for the final answer."""
    tools = tools or TOOLS
    execute = execute or _default_execute(run)
    plain = [{k: v for k, v in t.items() if k != "strict"} for t in tools]
    messages = [{"role": "user", "content": question}]
    for turn in range(max_turns):
        if claude:
            response = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": effort},
                system=system,
                tools=tools,
                messages=messages,
            )
        else:
            extra = {"tool_choice": {"type": "none"}} if force_final and turn == max_turns - 1 else {}
            response = client.messages.create(model=model, max_tokens=16000, system=system,
                                              tools=plain, messages=messages, **extra)
        run.turns = turn + 1
        run.stop_reason = response.stop_reason
        usage = getattr(response, "usage", None)
        for k in run.usage:
            run.usage[k] += getattr(usage, k, 0) or 0
        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason != "tool_use":
            run.final_text = text
            break
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            content, is_error = execute(turn, block.name, dict(block.input))
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": content, "is_error": is_error})
        if force_final and turn == max_turns - 2:
            results.append({"type": "text", "text": FINAL_PROMPT})
        messages.append({"role": "user", "content": results})
    return run


# ---- third-party providers -------------------------------------------------------------
# style "openai": chat completions with function calling; "anthropic": Messages API.
# Base URL, key, model and style can be overridden from the environment (see .env.example):
# <NAME>_BASE_URL, <NAME>_MODEL, <NAME>_API_STYLE, and the key variable(s) below.
# Check with `python -m agent.check_providers`.
PROVIDERS = {
    "deepseek": dict(style="openai", base_url="https://api.deepseek.com", key_envs=("DEEPSEEK_API_KEY",),
                     default_model="deepseek-v4-pro"),
    # 千问 AI 平台 (platform.qianwenai.com), Anthropic-compatible endpoint.
    "qwen": dict(style="anthropic", base_url="https://maas.qianwenaiapi.com/apps/anthropic",
                 key_envs=("QWEN_API_KEY", "DASHSCOPE_API_KEY"), default_model="qwen3.8-max"),
}
OPENAI_TOOLS = [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}} for t in TOOLS]


def provider_config(provider):
    import os

    from .env import load_env
    load_env()
    cfg = PROVIDERS[provider]
    name = provider.upper()
    key_env = next((k for k in cfg["key_envs"] if os.environ.get(k)), cfg["key_envs"][0])
    return dict(style=os.environ.get(f"{name}_API_STYLE", cfg["style"]),
                base_url=os.environ.get(f"{name}_BASE_URL", cfg["base_url"]),
                api_key=os.environ.get(key_env), key_env=key_env,
                model=os.environ.get(f"{name}_MODEL", cfg["default_model"]))


def make_client(provider):
    cfg = provider_config(provider)
    if not cfg["api_key"]:
        raise RuntimeError(f"{cfg['key_env']} is not set (environment settings or .env)")
    if cfg["style"] == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=cfg["api_key"], base_url=cfg["base_url"])
    import openai
    return openai.OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def _loop_openai_compat(run, question, client, max_turns, model, system=SYSTEM, tools=None, execute=None,
                        force_final=False):
    """Chat-completions tool loop; same parameters as _loop_anthropic."""
    oa_tools = _openai_tools(tools) if tools else OPENAI_TOOLS
    execute = execute or _default_execute(run)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    for turn in range(max_turns):
        extra = {"tool_choice": "none"} if force_final and turn == max_turns - 1 else {}
        response = client.chat.completions.create(model=model, messages=messages, tools=oa_tools,
                                                  max_tokens=16000, **extra)
        choice = response.choices[0]
        msg = choice.message
        run.turns = turn + 1
        run.stop_reason = choice.finish_reason
        usage = getattr(response, "usage", None)
        run.usage["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        run.usage["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        calls = msg.tool_calls or []
        if not calls:
            run.final_text = msg.content or ""
            break
        assistant = {"role": "assistant", "content": msg.content or "",
                     "tool_calls": [{"id": c.id, "type": "function",
                                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                    for c in calls]}
        # Thinking-mode models return reasoning_content; it is passed back within a tool-use turn.
        if getattr(msg, "reasoning_content", None):
            assistant["reasoning_content"] = msg.reasoning_content
        messages.append(assistant)
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                content = json.dumps({"error": "tool arguments were not valid JSON"})
                run.trace.append({"turn": turn, "tool": c.function.name, "input": c.function.arguments,
                                  "is_error": True, "gate": None})
            else:
                content, _ = execute(turn, c.function.name, args)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": content})
        if force_final and turn == max_turns - 2:
            messages.append({"role": "user", "content": FINAL_PROMPT})
    return run
