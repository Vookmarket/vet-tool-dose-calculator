#!/usr/bin/env python3
"""
薬用量クイック計算 テストスイート
Ollama不要のモックテスト
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tool import (
    DISCLAIMER,
    SYSTEM_PROMPT,
    build_prompt,
    call_ollama,
    format_output,
    validate_query,
    validate_species,
    validate_weight,
)


# ===== プロンプト生成関数のテスト =====


class TestBuildPrompt:
    """build_prompt 関数のテスト（LLM呼び出しなし）"""

    def test_basic_prompt(self) -> None:
        """基本的なプロンプト生成"""
        result = build_prompt("犬", 10.0, "アモキシシリン")
        assert "犬" in result
        assert "10.0" in result
        assert "アモキシシリン" in result

    def test_prompt_contains_all_fields(self) -> None:
        """プロンプトに全フィールドが含まれる"""
        result = build_prompt("猫", 4.5, "嘔吐")
        assert "動物種: 猫" in result
        assert "体重: 4.5 kg" in result
        assert "薬剤名または症状: 嘔吐" in result
        assert "薬用量を計算してください" in result

    def test_prompt_with_float_weight(self) -> None:
        """小数の体重でプロンプト生成"""
        result = build_prompt("犬", 25.3, "メロキシカム")
        assert "25.3" in result

    def test_system_prompt_contains_key_elements(self) -> None:
        """システムプロンプトに必要な要素が含まれる"""
        assert "獣医療" in SYSTEM_PROMPT
        assert "薬用量" in SYSTEM_PROMPT
        assert "mg/kg" in SYSTEM_PROMPT
        assert "禁忌" in SYSTEM_PROMPT


# ===== 入力バリデーションのテスト =====


class TestValidateSpecies:
    """validate_species 関数のテスト"""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("犬", "犬"),
            ("いぬ", "犬"),
            ("イヌ", "犬"),
            ("dog", "犬"),
            ("Dog", "犬"),
            ("DOG", "犬"),
            ("猫", "猫"),
            ("ねこ", "猫"),
            ("ネコ", "猫"),
            ("cat", "猫"),
            ("Cat", "猫"),
            ("CAT", "猫"),
        ],
    )
    def test_valid_species(self, input_val: str, expected: str) -> None:
        """有効な動物種入力"""
        assert validate_species(input_val) == expected

    @pytest.mark.parametrize(
        "input_val",
        ["", " ", "うさぎ", "ハムスター", "bird", "123"],
    )
    def test_invalid_species(self, input_val: str) -> None:
        """無効な動物種入力"""
        assert validate_species(input_val) is None

    def test_whitespace_handling(self) -> None:
        """前後空白の処理"""
        assert validate_species("  犬  ") == "犬"
        assert validate_species(" 猫 ") == "猫"


class TestValidateWeight:
    """validate_weight 関数のテスト"""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("5", 5.0),
            ("10.5", 10.5),
            ("0.5", 0.5),
            ("200", 200.0),
            (" 3.2 ", 3.2),
        ],
    )
    def test_valid_weight(self, input_val: str, expected: float) -> None:
        """有効な体重入力"""
        assert validate_weight(input_val) == expected

    @pytest.mark.parametrize(
        "input_val",
        ["", " ", "0", "-5", "201", "abc", "十キロ"],
    )
    def test_invalid_weight(self, input_val: str) -> None:
        """無効な体重入力"""
        assert validate_weight(input_val) is None


class TestValidateQuery:
    """validate_query 関数のテスト"""

    def test_valid_query(self) -> None:
        """有効な薬剤名入力"""
        assert validate_query("アモキシシリン") == "アモキシシリン"
        assert validate_query("嘔吐") == "嘔吐"

    def test_whitespace_trimming(self) -> None:
        """前後空白のトリミング"""
        assert validate_query("  メロキシカム  ") == "メロキシカム"

    def test_empty_query(self) -> None:
        """空入力"""
        assert validate_query("") is None
        assert validate_query("   ") is None


# ===== 出力フォーマットのテスト（モック使用） =====


class TestFormatOutput:
    """format_output 関数のテスト"""

    def test_contains_response_text(self) -> None:
        """応答テキストが出力に含まれる"""
        response = "テスト応答テキスト"
        result = format_output(response)
        assert "テスト応答テキスト" in result

    def test_contains_disclaimer(self) -> None:
        """免責事項が出力に含まれる"""
        result = format_output("何かの応答")
        assert "注意" in result
        assert "臨床判断" in result

    def test_contains_separators(self) -> None:
        """区切り線が含まれる"""
        result = format_output("応答")
        assert "=" * 50 in result


class TestCallOllama:
    """call_ollama 関数のテスト（モック使用、実際のOllama呼び出しなし）"""

    @patch("tool.requests.post")
    def test_successful_call(self, mock_post: MagicMock) -> None:
        """正常なAPI呼び出し"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "薬用量の計算結果です"}
        mock_post.return_value = mock_response

        result = call_ollama("テストプロンプト")
        assert result == "薬用量の計算結果です"

        # APIが正しいパラメータで呼ばれたか確認
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "qwen3:14b-q8_0"
        assert payload["prompt"] == "テストプロンプト"
        assert payload["stream"] is False

    @patch("tool.requests.post")
    def test_connection_error(self, mock_post: MagicMock) -> None:
        """接続エラー時の処理"""
        import requests as req

        mock_post.side_effect = req.exceptions.ConnectionError()

        with pytest.raises(ConnectionError) as exc_info:
            call_ollama("テストプロンプト")
        assert "Ollamaに接続できません" in str(exc_info.value)

    @patch("tool.requests.post")
    def test_timeout_error(self, mock_post: MagicMock) -> None:
        """タイムアウト時の処理"""
        import requests as req

        mock_post.side_effect = req.exceptions.Timeout()

        with pytest.raises(RuntimeError) as exc_info:
            call_ollama("テストプロンプト")
        assert "タイムアウト" in str(exc_info.value)

    @patch("tool.requests.post")
    def test_api_error_status(self, mock_post: MagicMock) -> None:
        """APIエラーステータス時の処理"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError) as exc_info:
            call_ollama("テストプロンプト")
        assert "500" in str(exc_info.value)

    @patch("tool.requests.post")
    def test_empty_response(self, mock_post: MagicMock) -> None:
        """空の応答時の処理"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        result = call_ollama("テストプロンプト")
        assert result == "（応答が空でした）"


# ===== 統合テスト（モック使用） =====


class TestIntegration:
    """統合テスト（LLMはモック）"""

    @patch("tool.requests.post")
    def test_full_flow_mock(self, mock_post: MagicMock) -> None:
        """プロンプト生成 → API呼び出し → 出力整形の一連の流れ"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "■ アモキシシリン\n  用量: 20 mg/kg\n  計算結果: 200 mg"
        }
        mock_post.return_value = mock_response

        # プロンプト生成
        prompt = build_prompt("犬", 10.0, "アモキシシリン")
        assert "犬" in prompt

        # API呼び出し（モック）
        response = call_ollama(prompt)
        assert "アモキシシリン" in response

        # 出力整形
        output = format_output(response)
        assert "アモキシシリン" in output
        assert "注意" in output
