"""Check that each provider is reachable, the key works and the configured model exists,
then make one tiny tool call. Costs a few hundred tokens per provider.

    python -m agent.check_providers            # all configured providers
    python -m agent.check_providers qwen
"""
import json
import sys

from .orchestrator import OPENAI_COMPAT, OPENAI_TOOLS, make_openai_client, provider_config


def check(provider):
    cfg = provider_config(provider)
    print(f"== {provider}: {cfg['base_url']}  model={cfg['model']}")
    if not cfg["api_key"]:
        print(f"   {cfg['key_env']} not set, skipped")
        return False
    client = make_openai_client(provider)
    try:
        ids = sorted(m.id for m in client.models.list())
        print(f"   {len(ids)} models available; configured model listed: {cfg['model'] in ids}")
        if cfg["model"] not in ids:
            print("   candidates:", [i for i in ids if provider[:4] in i.lower()][:20])
    except Exception as e:
        print(f"   models.list failed ({type(e).__name__}: {e}); trying the call anyway")
    try:
        r = client.chat.completions.create(
            model=cfg["model"], max_tokens=200, tools=OPENAI_TOOLS,
            messages=[{"role": "user", "content": "调用 list_metrics 工具。"}])
        calls = r.choices[0].message.tool_calls or []
        print(f"   tool call: {[c.function.name for c in calls]}  finish={r.choices[0].finish_reason}")
        return bool(calls)
    except Exception as e:
        print(f"   call failed: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    names = sys.argv[1:] or list(OPENAI_COMPAT)
    print(json.dumps({p: check(p) for p in names}))
