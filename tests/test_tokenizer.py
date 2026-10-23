from vuldee.data.tokenizer import create_tokens, normalize_tokens


def test_create_tokens_single_operators():
    assert create_tokens("x = a + b;") == ["x", "=", "a", "+", "b", ";"]


def test_create_tokens_multi_ops_and_strings():
    tokens = create_tokens('if (x != "hello") return;')
    assert "!=" in tokens
    assert '"hello"' in tokens


def test_normalize_tokens_renames_names_but_preserves_signal():
    tokens = normalize_tokens(create_tokens("int main() { return strlen(s); }"))
    assert "int" in tokens
    assert "main" in tokens          # entry point preserved
    assert "strlen" in tokens        # dangerous API preserved
    assert "variable_0" in tokens    # local variable abstracted


def test_normalize_tokens_function_abstraction():
    tokens = normalize_tokens(create_tokens("helper(a);"))
    assert "func_0" in tokens
