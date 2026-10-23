"""Check that each provider is reachable, the key works and the configured model answers
one tiny tool call. Costs a few hundred tokens per provider.

    python -m agent.check_providers            # all providers
    python -m agent.check_providers qwen
"""
import json
import sys

from .orchestrator import OPENAI_TOOLS, PLAIN_TOOLS, PROVIDERS, make_client, provider_config

PROMPT = "调用 list_metrics 工具。"


def check(provider):
    cfg = provider_config(provider)
    print(f"== {provider}: {cfg['style']} API at {cfg['base_url']}  model={cfg['model']}")
    if not cfg["api_key"]:
        print(f"   {cfg['key_env']} not set, skipped")
        return False
    client = make_client(provider)
    try:
        ids = sorted(m.id for m in client.models.list())
        print(f"   {len(ids)} models listed; configured model present: {cfg['model'] in ids}")
        if cfg["model"] not in ids:
            print("   candidates:", [i for i in ids if provider[:4] in i.lower()][:20])
    except Exception as e:
        print(f"   model listing unavailable ({type(e).__name__}); trying the call anyway")
    try:
        if cfg["style"] == "anthropic":
            r = client.messages.create(model=cfg["model"], max_tokens=300, tools=PLAIN_TOOLS,
                                       messages=[{"role": "user", "content": PROMPT}])
            calls = [b.name for b in r.content if b.type == "tool_use"]
            finish = r.stop_reason
        else:
            r = client.chat.completions.create(model=cfg["model"], max_tokens=300, tools=OPENAI_TOOLS,
                                               messages=[{"role": "user", "content": PROMPT}])
            calls = [c.function.name for c in r.choices[0].message.tool_calls or []]
            finish = r.choices[0].finish_reason
        print(f"   tool call: {calls}  stop={finish}")
        return bool(calls)
    except Exception as e:
        print(f"   call failed: {type(e).__name__}: {str(e)[:300]}")
        return False


if __name__ == "__main__":
    names = sys.argv[1:] or list(PROVIDERS)
    print(json.dumps({p: check(p) for p in names}))
