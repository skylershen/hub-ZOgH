"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_LOG = OUTPUT_DIR / "Print.log"

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))

from weather_backend import get_weather, get_location


class DualOutput:
    def __init__(self, filename):
        # 保存原始的控制台输出
        self.terminal = sys.stdout
        # 打开日志文件（强烈建议指定 utf-8 编码）
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        # 同时写入控制台和文件
        self.terminal.write(message)
        self.log.write(message)
        # 【关键】每次写入后立刻刷新文件缓冲区，确保内容落盘
        self.flush()

    def flush(self):
        # 【关键】刷新两个流的缓冲区
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # 【关键】程序结束时关闭文件
        if not self.log.closed:
            self.log.close()


# LLM 配置
PROVIDERS = {
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# 手写工具的 JSON Schema
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"——写得越具体越准。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_location",
            "description": "查询指定城市的经纬度和详细地址信息。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "通过经纬度坐标查询指定坐标位置的当前天气及未来3天预报",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "string", "description": "纬度"},
                    "longitude": {"type": "string", "description": "经度"},
                    "address": {"type": "string", "description": "详细地址"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

# 工具名 → 后端函数的 dispatch 表
# 新增工具只需：1) 在上面写 schema；2) 在这里加一行映射。这是 Function Call 的扩展方式。

TOOL_DISPATCH = {
    "get_weather": get_weather,
    "get_location": get_location,
}

SYSTEM_PROMPT = (
    "你是一名天气和城市信息查询助手。回答用户关于城市天气和城市信息的问题。"
)


def run_async(client, model: str, messages, tool_call_log) -> str:
    """
    循环异步调用tool_call并生成回答。
    """
    # 带上 tools，让模型决定是否调用工具
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        # 把 assistant 这条带 tool_calls 的消息原样回填，保持上下文
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    # 工具执行！！
                    result = fn(**args)
                    print(f"****** 成功执行 {name} 工具，获取到的结果是：{result}! ******")
                except TypeError as e:
                    result = f"参数错误：{e}"
                    print("result=", result)
                except Exception as e:
                    result = f"工具执行失败：{e}"
                    print("result=", result)
            # 以 role=tool 把每个工具的结果回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        return run_async(client, model, messages, tool_call_log)

    return msg.content or ""


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    提问 → 模型输出 tool_call → 执行 → 回填 → 最终回答。
    返回 {answer, tool_calls, elapsed} 用于对比器汇总。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    answer = run_async(client, model, messages, tool_call_log)
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "成都的天气如何？",
    "北京的坐标是啥？",
    "沈阳的天气和北京的天气如何？",
    "天津的天气和北京的坐标分别是什么？",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集", default=True)
    parser.add_argument("--provider", default="dashscope", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, verbose=not (args.quiet or args.json))
        result["question"] = q
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        # 单问题输出单对象；demo 输出数组
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    original_stdout = sys.stdout
    try:
        sys.stdout = DualOutput(OUTPUT_LOG)
        main()
    finally:
        sys.stdout = original_stdout
        sys.stdout.close()
