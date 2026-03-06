#!/usr/bin/env python3
"""
薬用量クイック計算 (Dose Calculator)
Vet Tools Daily #1

体重と薬剤名（または症状）を入力すると、
犬猫の一般的な薬用量をAIが計算して表示します。

LLM: Ollama (Qwen3 14B) をローカルで使用
"""

import json
import sys
from typing import Optional

try:
    import requests
except ImportError:
    print("エラー: requests ライブラリがインストールされていません。")
    print("以下のコマンドでインストールしてください:")
    print("  pip install requests")
    sys.exit(1)


# --- 設定 ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:14b-q8_0"

DISCLAIMER = (
    "\n⚠️ 注意: このツールの出力はAIによる参考情報です。\n"
    "臨床判断は必ず獣医師が行ってください。\n"
)

SYSTEM_PROMPT = """\
あなたは獣医療の薬用量計算を支援するAIアシスタントです。
以下のルールに従って回答してください。

【役割】
- 獣医師・動物看護師が日常診療で参照する薬用量情報を提供する
- 体重に基づいた実投与量を計算して表示する

【回答フォーマット】
以下の形式で、該当する薬剤の用量情報を表形式で出力してください:

━━━━━━━━━━━━━━━━━━━━━━━
📋 {動物種} ({体重}kg) の薬用量計算結果
━━━━━━━━━━━━━━━━━━━━━━━

■ {薬剤名1}
  用量: {用量} mg/kg
  投与経路: {PO/IV/SC/IM}
  頻度: {SID/BID/TID/etc.}
  ──────────
  💊 計算結果: {体重×用量} mg ({頻度})

（該当する薬剤が複数あれば繰り返す）

【注意事項】の見出しの下に、その薬剤使用時の一般的な注意点を簡潔に記載してください。

【ルール】
- 犬と猫で用量が異なる場合は、指定された動物種の用量のみ表示する
- 一般的に使用される標準用量範囲を表示する（最小〜最大がある場合は範囲で表示）
- 禁忌がある場合は必ず明記する
- 回答は日本語で行う
- 薬剤名が不明確な場合は、症状から推測される一般的な処方薬を複数提示する
- /no_think\
"""


def build_prompt(species: str, weight: float, query: str) -> str:
    """LLMに送信するプロンプトを構築する。

    Args:
        species: 動物種（犬 or 猫）
        weight: 体重（kg）
        query: 薬剤名または症状

    Returns:
        構築されたプロンプト文字列
    """
    return (
        f"動物種: {species}\n"
        f"体重: {weight} kg\n"
        f"薬剤名または症状: {query}\n\n"
        f"上記の情報に基づいて、薬用量を計算してください。"
    )


def validate_species(species_input: str) -> Optional[str]:
    """動物種の入力をバリデーションする。

    Args:
        species_input: ユーザーが入力した動物種

    Returns:
        正規化された動物種（犬/猫）。無効な場合はNone。
    """
    species_input = species_input.strip()
    if species_input in ("犬", "いぬ", "イヌ", "dog", "Dog", "DOG"):
        return "犬"
    elif species_input in ("猫", "ねこ", "ネコ", "cat", "Cat", "CAT"):
        return "猫"
    return None


def validate_weight(weight_input: str) -> Optional[float]:
    """体重の入力をバリデーションする。

    Args:
        weight_input: ユーザーが入力した体重

    Returns:
        体重（float）。無効な場合はNone。
    """
    weight_input = weight_input.strip()
    if not weight_input:
        return None
    try:
        weight = float(weight_input)
        if weight <= 0 or weight > 200:
            return None
        return weight
    except ValueError:
        return None


def validate_query(query_input: str) -> Optional[str]:
    """薬剤名/症状の入力をバリデーションする。

    Args:
        query_input: ユーザーが入力した薬剤名または症状

    Returns:
        バリデーション済みの入力。無効な場合はNone。
    """
    query_input = query_input.strip()
    if not query_input:
        return None
    return query_input


def call_ollama(prompt: str) -> str:
    """Ollama APIを呼び出してLLMの応答を取得する。

    Args:
        prompt: LLMに送信するプロンプト

    Returns:
        LLMの応答テキスト

    Raises:
        ConnectionError: Ollamaに接続できない場合
        RuntimeError: APIがエラーを返した場合
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Ollamaに接続できません。\n"
            "以下を確認してください:\n"
            "  1. Ollamaがインストールされているか\n"
            "  2. Ollamaが起動しているか（ターミナルで 'ollama serve' を実行）\n"
            "  3. Qwen3モデルがダウンロード済みか（'ollama pull qwen3:14b-q8_0'）"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollamaからの応答がタイムアウトしました。\n"
            "モデルの読み込みに時間がかかっている可能性があります。\n"
            "しばらく待ってから再度お試しください。"
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama APIエラー（ステータスコード: {response.status_code}）\n"
            f"レスポンス: {response.text}"
        )

    try:
        result = response.json()
        return result.get("response", "（応答が空でした）")
    except json.JSONDecodeError:
        raise RuntimeError("Ollamaからの応答を解析できませんでした。")


def format_output(response_text: str) -> str:
    """LLMの応答を整形して出力用文字列を作成する。

    Args:
        response_text: LLMからの生の応答テキスト

    Returns:
        整形された出力文字列（免責事項付き）
    """
    separator = "=" * 50
    return (
        f"\n{separator}\n"
        f"{response_text}\n"
        f"{separator}\n"
        f"{DISCLAIMER}"
    )


def interactive_session() -> None:
    """対話的セッションを実行する。"""
    print(DISCLAIMER)
    print("=" * 50)
    print("  薬用量クイック計算 (Dose Calculator)")
    print("  Vet Tools Daily #1")
    print("=" * 50)
    print("\n終了するには Ctrl+C または 'q' を入力してください。\n")

    while True:
        try:
            # --- 動物種 ---
            species_input = input("動物種を入力してください（犬 / 猫）: ")
            if species_input.strip().lower() in ("q", "quit", "exit"):
                print("\nご利用ありがとうございました。")
                break

            species = validate_species(species_input)
            if species is None:
                print("エラー: 動物種は「犬」または「猫」で入力してください。\n")
                continue

            # --- 体重 ---
            weight_input = input("体重を入力してください（kg）: ")
            if weight_input.strip().lower() in ("q", "quit", "exit"):
                print("\nご利用ありがとうございました。")
                break

            weight = validate_weight(weight_input)
            if weight is None:
                print("エラー: 体重は0より大きく200以下の数値で入力してください。\n")
                continue

            # --- 薬剤名/症状 ---
            query_input = input("薬剤名または症状を入力してください: ")
            if query_input.strip().lower() in ("q", "quit", "exit"):
                print("\nご利用ありがとうございました。")
                break

            query = validate_query(query_input)
            if query is None:
                print("エラー: 薬剤名または症状を入力してください。\n")
                continue

            # --- LLM呼び出し ---
            print("\n計算中...")
            prompt = build_prompt(species, weight, query)

            try:
                response = call_ollama(prompt)
                print(format_output(response))
            except (ConnectionError, RuntimeError) as e:
                print(f"\nエラー: {e}\n")

        except KeyboardInterrupt:
            print("\n\nご利用ありがとうございました。")
            break
        except EOFError:
            print("\n\nご利用ありがとうございました。")
            break


def main() -> None:
    """メインエントリポイント。"""
    interactive_session()


if __name__ == "__main__":
    main()
