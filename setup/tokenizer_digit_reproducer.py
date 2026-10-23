#!/usr/bin/env python3
"""Compare Hugging Face and llama.cpp tokenization of Tiny Aya digit-boundary cases.

The Tiny Aya tokenizer (CohereLabs/tiny-aya-*) isolates digit groups with the
pre-tokenizer regex  \\d{1,3}(?=(?:\\d{3})*\\b)  before its main BPE split.
llama.cpp maps that regex to its AFMoE digit splitter
(unicode_regex_split_custom_afmoe in src/unicode.cpp). That splitter keeps the
text preceding a digit run inside the first digit segment and ignores the
trailing word-boundary condition, so inputs such as "  46" or "1990s" tokenize
differently from the original tokenizer.

This script tokenizes a fixed set of boundary cases with both implementations
and reports every token-ID mismatch. Ground truth is the `tokenizers` library
reading tokenizer.json from a local Tiny Aya checkout; nothing is downloaded.
The llama.cpp side is either the `llama-tokenize` binary of any build (pinned
or current upstream) or the llama-cpp-python binding. Both sides tokenize
without BOS and without special-token parsing, so the IDs compare directly.

Examples (paths are arguments; nothing is hardcoded):

  python tokenizer_digit_reproducer.py --tokenizer /path/to/tiny-aya-global \\
      --gguf /path/to/tiny-aya.gguf --llama-tokenize /path/to/llama-tokenize \\
      --runtime-label "llama.cpp 797de63 unpatched" --json out.json

  python tokenizer_digit_reproducer.py --tokenizer DIR --gguf FILE --python-binding

Exit status is 1 when any regular case mismatches, 0 otherwise. Cases in the
"digit_class_probe" class are informational: they probe whether the \\d class
(decimal digits only) and llama.cpp's number flag agree on non-decimal numerals.
They never affect the exit status.

Requirements: Python 3.9+, `tokenizers` (or `transformers`) for the Hugging
Face side; `llama_cpp` only when --python-binding is used. The GGUF is opened
vocab-only, so any Tiny Aya GGUF works regardless of quantization.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Sequence

ZWNJ = "‌"
ZWJ = "‍"
ACUTE = "́"  # combining acute accent (Mn)
NUKTA = "়"  # Bengali sign nukta (Mn)


@dataclass
class Case:
    case_id: str
    case_class: str
    text: str


def build_cases() -> List[Case]:
    cases: List[Case] = []

    def add(case_class: str, case_id: str, text: str) -> None:
        cases.append(Case(case_id=case_id, case_class=case_class, text=text))

    # Leading whitespace before a digit run (the original pilot mismatch).
    add("leading_space", "two_spaces_46", "  46")
    add("leading_space", "one_space_46", " 46")
    add("leading_space", "three_spaces_46", "   46")
    add("leading_space", "tab_46", "\t46")
    add("leading_space", "newline_46", "\n46")
    add("leading_space", "word_space_year", "word 2007")
    add("leading_space", "sentence_space_number", "Population is 123456 people")

    # Digits adjacent to punctuation.
    add("punctuation", "parens_year", "(2007)")
    add("punctuation", "comma_grouped", "12,345")
    add("punctuation", "decimal_point", "3.14")
    add("punctuation", "colon_comma_mix", "x: 12, 3456")
    add("punctuation", "dollar_amount", "$100")
    add("punctuation", "percent", "50%")
    add("punctuation", "trailing_period", "2007.")
    add("punctuation", "phone_hyphens", "1-800-555-0199")
    add("punctuation", "quoted_year", '"2007"')
    add("punctuation", "slash_years", "2007/2008")

    # Letter or connector suffixes: \\b after the digits is not satisfied.
    add("letter_suffix", "1990s", "1990s")
    add("letter_suffix", "1480s", "1480s")
    add("letter_suffix", "two_decades", "1970s and 1930s")
    add("letter_suffix", "ordinal_3rd", "3rd")
    add("letter_suffix", "underscore_12_name", "12_name")
    add("letter_suffix", "letter_prefix_A4", "A4")
    add("letter_suffix", "letter_prefix_x2007", "x2007")
    add("letter_suffix", "letter_suffix_2007x", "2007x")
    add("letter_suffix", "suffix_then_spaces_46", "1480s  46")

    # Bengali digits (U+09E6..U+09EF are \\d in Unicode-aware regex engines).
    add("bengali_digits", "bn_2007", "২০০৭")
    add("bengali_digits", "bn_spaces_2007", "  ২০০৭")
    add("bengali_digits", "bn_1234567", "১২৩৪৫৬৭")
    add("bengali_digits", "bn_sentence", "তিনি ২০০৭ সালে জন্মগ্রহণ করেন।")
    add("bengali_digits", "bn_currency_indian_grouping", "৳১,২৩,৪৫৬")

    # Devanagari digits (U+0966..U+096F).
    add("devanagari_digits", "hi_2007", "२००७")
    add("devanagari_digits", "hi_spaces_2007", "  २००७")
    add("devanagari_digits", "hi_1234567", "१२३४५६७")
    add("devanagari_digits", "hi_sentence", "वह २००७ में पैदा हुआ।")

    # Join controls after digits are word boundaries in the original tokenizer.
    add("join_control", "zwnj_after_2007", "2007" + ZWNJ)
    add("join_control", "zwj_after_2007", "2007" + ZWJ)
    add("join_control", "bn_spaces_2007_zwnj", "  ২০০৭" + ZWNJ)
    add("join_control", "hi_spaces_2007_zwj", "  २००७" + ZWJ)
    add("join_control", "zwnj_between_digit_runs", "2007" + ZWNJ + "2008")

    # Combining marks after digits are word characters, so no boundary.
    add("combining_mark", "acute_after_2007", "2007" + ACUTE)
    add("combining_mark", "nukta_after_bn_2007", "২০০৭" + NUKTA)

    # Digit-run lengths 1 to 7, bare and after one space.
    for n in range(1, 8):
        digits = "1234567"[:n]
        add("group_length", f"len{n}_bare", digits)
        add("group_length", f"len{n}_after_space", " " + digits)
    add("group_length", "comma_million", "1,000,000")
    add("group_length", "len8_bare", "12345678")

    # Mixed natural text.
    add("mixed", "words_and_numbers", "abc 123 def 4567")
    add("mixed", "chapter_line", "Chapter 12: 1990s revisited (pp. 1234567-1234570)")
    add("mixed", "date_iso", "2026-10-23")
    add("mixed", "time_colon", "10:45")

    # Other decimal-digit scripts (general category Nd, matched by \\d).
    add("other_nd_digits", "fullwidth_2007", "２００７")
    add("other_nd_digits", "fullwidth_spaces_1234567", "  １２３４５６７")
    add("other_nd_digits", "arabic_indic_2007", "٢٠٠٧")
    add("other_nd_digits", "arabic_indic_spaces_2007", "  ٢٠٠٧")

    # Probes: \\d (Nd only) versus llama.cpp's number flag on No/Nl numerals.
    # Single characters split identically either way; runs of several reveal
    # whether the implementation treats non-decimal numerals as digit runs.
    add("digit_class_probe", "superscript_two", "x²")
    add("digit_class_probe", "superscript_run_2345", "x²³⁴⁵")
    add("digit_class_probe", "superscript_run_after_space", " ²³⁴⁵")
    add("digit_class_probe", "decimal_then_superscript", "12²")
    add("digit_class_probe", "roman_numeral_twelve", "Ⅻ")
    add("digit_class_probe", "roman_run_1234", "ⅠⅡⅢⅣ")
    add("digit_class_probe", "circled_one", "①")
    add("digit_class_probe", "circled_run_1234", "①②③④")
    add("digit_class_probe", "vulgar_fraction_half", "½")
    return cases


def load_hf_tokenizer(tokenizer_dir: str) -> Callable[[str], List[int]]:
    tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
    if os.path.isfile(tokenizer_json):
        try:
            from tokenizers import Tokenizer  # type: ignore

            tok = Tokenizer.from_file(tokenizer_json)
            return lambda text: tok.encode(text, add_special_tokens=False).ids
        except ImportError:
            pass
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Neither `tokenizers` nor `transformers` is importable in this environment"
        ) from exc
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    return lambda text: tok.encode(text, add_special_tokens=False)


def load_hf_pretokenizer(tokenizer_dir: str) -> Optional[Callable[[str], List[str]]]:
    tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError:
        return None
    if not os.path.isfile(tokenizer_json):
        return None
    tok = Tokenizer.from_file(tokenizer_json)
    pre = tok.pre_tokenizer
    if pre is None:
        return None
    return lambda text: [piece for piece, _ in pre.pre_tokenize_str(text)]


def make_cli_backend(binary: str, gguf: str) -> Callable[[str], List[int]]:
    base_cmd = [
        binary,
        "-m",
        gguf,
        "--stdin",
        "--ids",
        "--no-bos",
        "--no-escape",
        "--log-disable",
    ]

    def tokenize(text: str) -> List[int]:
        proc = subprocess.run(
            base_cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-tokenize exited {proc.returncode}: "
                f"{proc.stderr.decode('utf-8', 'replace').strip()[-400:]}"
            )
        out = proc.stdout.decode("utf-8", "replace")
        start = out.rfind("[")
        end = out.rfind("]")
        if start < 0 or end < start:
            raise RuntimeError(f"could not find token list in output: {out[-200:]!r}")
        body = out[start + 1 : end].strip()
        return [int(x) for x in body.split(",")] if body else []

    return tokenize


def make_python_backend(gguf: str, lib_path: Optional[str]) -> Callable[[str], List[int]]:
    if lib_path:
        os.environ["LLAMA_CPP_LIB_PATH"] = lib_path
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as exc:
        raise SystemExit("`llama_cpp` is not importable; use --llama-tokenize instead") from exc
    llm = Llama(model_path=gguf, vocab_only=True, verbose=False)
    return lambda text: list(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))


def describe(text: str) -> str:
    return json.dumps(text, ensure_ascii=True)


def run(args: argparse.Namespace) -> int:
    cases = build_cases()
    if args.extra_cases:
        with open(args.extra_cases, encoding="utf-8") as fh:
            extra = json.load(fh)
        if not isinstance(extra, list) or not all(isinstance(t, str) for t in extra):
            raise SystemExit("--extra-cases must be a JSON list of strings")
        for idx, text in enumerate(extra):
            cases.append(Case(case_id=f"extra_{idx}", case_class="extra", text=text))

    hf_tokenize = load_hf_tokenizer(args.tokenizer)
    hf_pretokenize = load_hf_pretokenizer(args.tokenizer)
    if args.python_binding:
        llama_tokenize = make_python_backend(args.gguf, args.lib_path)
        backend = "llama-cpp-python"
        workers = 1
    else:
        llama_tokenize = make_cli_backend(args.llama_tokenize, args.gguf)
        backend = "llama-tokenize"
        workers = max(1, args.workers)

    hf_ids = [hf_tokenize(c.text) for c in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        llama_results = list(pool.map(lambda c: _safe(llama_tokenize, c.text), cases))

    rows = []
    regular_mismatches = 0
    probe_differences = 0
    errors = 0
    for case, hf, (llama_ids, error) in zip(cases, hf_ids, llama_results):
        probe = case.case_class == "digit_class_probe"
        if error is not None:
            status = "ERROR"
            errors += 1
            match = False
        else:
            match = hf == llama_ids
            if match:
                status = "probe-same" if probe else "ok"
            else:
                status = "probe-diff" if probe else "MISMATCH"
                if probe:
                    probe_differences += 1
                else:
                    regular_mismatches += 1
        rows.append(
            {
                "case_id": case.case_id,
                "case_class": case.case_class,
                "text": case.text,
                "hf_ids": hf,
                "llama_ids": llama_ids,
                "hf_pretokens": hf_pretokenize(case.text) if hf_pretokenize else None,
                "match": match,
                "status": status,
                "error": error,
            }
        )

    for row in rows:
        if args.only_mismatches and row["status"] in ("ok", "probe-same"):
            continue
        print(f"{row['status']:<10} {row['case_class']:<18} {row['case_id']:<28} {describe(row['text'])}")
        if row["status"] in ("MISMATCH", "probe-diff", "ERROR") or args.verbose:
            print(f"           hf    : {row['hf_ids']}")
            print(f"           llama : {row['llama_ids']}")
            if row["hf_pretokens"] is not None and (args.show_pretok or row["status"] != "ok"):
                print(f"           hf pre: {[describe(p) for p in row['hf_pretokens']]}")
            if row["error"]:
                print(f"           error : {row['error']}")

    regular_total = sum(1 for r in rows if r["case_class"] != "digit_class_probe")
    probe_total = len(rows) - regular_total
    summary = {
        "regular_cases": regular_total,
        "regular_mismatches": regular_mismatches,
        "probe_cases": probe_total,
        "probe_differences": probe_differences,
        "errors": errors,
    }
    print()
    print(
        f"regular cases: {regular_total}, mismatches: {regular_mismatches}; "
        f"probe cases: {probe_total}, differences: {probe_differences}; errors: {errors}"
    )

    if args.json:
        try:
            import tokenizers  # type: ignore

            tokenizers_version = getattr(tokenizers, "__version__", None)
        except ImportError:
            tokenizers_version = None
        payload = {
            "runtime_label": args.runtime_label,
            "backend": backend,
            "gguf_basename": os.path.basename(args.gguf),
            "gguf_bytes": os.path.getsize(args.gguf),
            "tokenizer_basename": os.path.basename(os.path.normpath(args.tokenizer)),
            "tokenizers_version": tokenizers_version,
            "summary": summary,
            "cases": rows,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print(f"wrote {args.json}")

    return 1 if (regular_mismatches or errors) else 0


def _safe(fn: Callable[[str], List[int]], text: str):
    try:
        return fn(text), None
    except Exception as exc:  # noqa: BLE001 - report every backend failure per case
        return None, str(exc)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="directory holding the Tiny Aya tokenizer.json")
    parser.add_argument("--gguf", required=True, help="any Tiny Aya GGUF; opened vocab-only")
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--llama-tokenize", help="path to a llama-tokenize binary")
    backend.add_argument("--python-binding", action="store_true", help="use the llama_cpp Python binding")
    parser.add_argument("--lib-path", help="LLAMA_CPP_LIB_PATH for --python-binding (patched shared library directory)")
    parser.add_argument("--runtime-label", default="", help="free text recorded in the JSON output, e.g. the llama.cpp revision")
    parser.add_argument("--json", help="write per-case results to this JSON file")
    parser.add_argument("--extra-cases", help="JSON list of extra strings to test")
    parser.add_argument("--workers", type=int, default=4, help="parallel llama-tokenize processes")
    parser.add_argument("--only-mismatches", action="store_true")
    parser.add_argument("--show-pretok", action="store_true", help="print Hugging Face pre-tokenizer pieces for every row")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))
