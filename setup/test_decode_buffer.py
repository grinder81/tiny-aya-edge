"""Boundary regression for the locally patched upstream decoder; no model needed."""
import ctypes
from types import SimpleNamespace
from unittest.mock import patch

from llama_cpp._internals import LlamaModel


def check(length):
    payload = b'x' * length
    calls = []
    def piece(vocab, token, buffer, size, lstrip, special):
        calls.append(size)
        if size < len(payload):
            return -len(payload)
        ctypes.memmove(buffer, payload, len(payload))
        return len(payload)
    model = SimpleNamespace(vocab=None, token_bos=lambda: -1)
    with patch('llama_cpp._internals.llama_cpp.llama_token_to_piece', piece):
        assert LlamaModel.detokenize(model, [42]) == payload
    assert calls == ([32] if length <= 32 else [32, length])


for size in [0, 1, 31, 32, 33, 34, 128]:
    check(size)
print('Seven decoding buffer boundary cases passed')
