import sys
sys.path.insert(0, ".")
from scripts.run_eval_phase2 import normalize, exact_match, token_f1
import pytest

# normalize
def test_normalize_lowercase():
    assert normalize("Hello World") == "hello world"

def test_normalize_strips_punctuation():
    assert normalize("hello, world!") == "hello world"

def test_normalize_removes_article_a():
    assert normalize("a cat sat") == "cat sat"

def test_normalize_removes_article_an():
    assert normalize("an apple") == "apple"

def test_normalize_removes_article_the():
    assert normalize("the quick brown fox") == "quick brown fox"

def test_normalize_collapses_whitespace():
    assert normalize("hello   world") == "hello world"

def test_normalize_mixed():
    assert normalize("The Quick, Brown Fox!") == "quick brown fox"

def test_normalize_empty():
    assert normalize("") == ""

def test_normalize_only_articles():
    assert normalize("the a an") == ""

def test_normalize_numbers():
    assert normalize("Answer is 42.") == "answer is 42"

def test_normalize_hyphenated():
    assert normalize("state-of-the-art") == "stateoftheart"

def test_normalize_unicode_safe():
    assert normalize("naïve") == "naïve"

# exact_match
def test_em_exact():
    assert exact_match("The cat sat", "the cat sat") == 1.0

def test_em_different():
    assert exact_match("dog", "cat") == 0.0

def test_em_article_insensitive():
    assert exact_match("a cat", "cat") == 1.0

def test_em_punctuation_insensitive():
    assert exact_match("hello, world!", "hello world") == 1.0

# token_f1() 
def test_f1_perfect():
    assert token_f1("cat sat mat", "cat sat mat") == 1.0

def test_f1_partial():
    f1 = token_f1("cat sat", "cat sat mat")
    assert 0.0 < f1 < 1.0

def test_f1_no_overlap():
    assert token_f1("dog", "cat") == 0.0

def test_f1_empty_pred():
    assert token_f1("", "cat sat") == 0.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])