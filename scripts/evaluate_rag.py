#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    dimension: str
    question: str
    ground_truth: str
    checker: Callable[[str], list[CheckResult]]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def has_any(text: str, patterns: list[str]) -> bool:
    lowered = normalize(text)
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def post_query(base_url: str, question: str, top_k: int | None, timeout: float) -> dict:
    payload: dict[str, object] = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/query",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from /query: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to {base_url}: {exc}") from exc


def check_sovereign_fallback(answer: str) -> list[CheckResult]:
    text = normalize(answer)
    forbidden_allow = has_any(
        text,
        [
            r"\byes\b.*\bclaude\b",
            r"\bcan\b.*\broute\b.*\banthropic\b",
            r"\bsafely\b.*\bclaude\b",
            r"\bfallback\b.*\bclaude\b",
            r"\balternative route\b.*\banthropic\b",
            r"\brequiring an alternative route\b",
            r"\bwould\b.*\broute\b.*\bclaude\b",
            r"\bbackup option\b.*\banthropic\b",
        ],
    )
    return [
        CheckResult("explicit_no", bool(re.search(r"\bno\b|must not|cannot|can not", text)), "Answer should clearly reject Claude fallback."),
        CheckResult(
            "mentions_sovereign_rule",
            any(phrase in text for phrase in ["sovereign", "data sovereignty", "data residency", "section 2.1", "local h100astra"]),
            "Answer should cite the sovereignty/data-residency compliance rule.",
        ),
        CheckResult("mentions_no_us_cloud", "anthropic" in text and ("us" in text or "u.s" in text or "cloud endpoint" in text), "Answer should identify Anthropic/US-cloud endpoint risk."),
        CheckResult("mentions_required_error", "err-sov-403" in text or "compliance exception" in text or "terminate" in text, "Answer should say terminate/raise ERR-SOV-403 compliance exception."),
        CheckResult("no_unsafe_fallback", not forbidden_allow, "Answer must not approve routing to Claude as backup."),
    ]


def check_medical_claim(answer: str) -> list[CheckResult]:
    text = normalize(answer)
    has_rm_1200 = bool(re.search(r"rm\s*1[,.]?200|1[,.]?200\s*rm", text))
    wrong_limit = bool(re.search(r"rm\s*3[,.]?500|sgd\s*800|800\s*sgd", text))
    return [
        CheckResult("mentions_rm_1200", has_rm_1200, "Answer should give RM 1,200 outpatient limit."),
        CheckResult("mentions_hr_manager", "hr manager" in text, "Answer should say HR Manager approves the claim."),
        CheckResult("probation_context", "probation" in text or "newly hired" in text or "3-month" in text, "Answer should stay in the probation/new hire row context."),
        CheckResult("no_table_confusion", not wrong_limit, "Answer should not confuse with confirmed RM 3,500 or Singapore SGD 800."),
    ]


def check_travel_reimbursement(answer: str) -> list[CheckResult]:
    text = normalize(answer)
    has_total = bool(re.search(r"rm\s*522\.50|522\.50\s*rm|rm\s*522\.5\b", text))
    has_first_rate = bool(re.search(r"500\s*(km|kilometers?).{0,80}(0\.85|85\s*sen)", text)) or bool(re.search(r"0\.85.{0,80}500\s*(km|kilometers?)", text))
    has_second_rate = (
        bool(re.search(r"150[\s\\{}at-z]*(km|kilometers?).{0,100}(0\.65|65\s*sen)", text)) 
        or 
        bool(re.search(r"0\.65.{0,100}150[\s\\{}at-z]*(km|kilometers?)", text))
    )
    # has_second_rate = bool(re.search(r"150\s*(km|kilometers?).{0,80}(0\.65|65\s*sen)", text)) or bool(re.search(r"0\.65.{0,80}150\s*(km|kilometers?)", text))
    has_calc = "425" in text and ("97.50" in text or "97.5" in text)
    return [
        CheckResult("mentions_total_rm_522_50", has_total, "Answer should give total reimbursement RM 522.50."),
        CheckResult("extracts_first_tier", has_first_rate, "Answer should use 500 km at RM 0.85/km."),
        CheckResult("extracts_second_tier", has_second_rate, "Answer should use remaining 150 km at RM 0.65/km."),
        CheckResult("shows_multistep_math", has_calc, "Answer should show RM 425 + RM 97.50 calculation."),
    ]


CASES = [
    EvalCase(
        case_id="sovereign_fallback_guardrail",
        dimension="Faithfulness & Safety Guardrails",
        question=(
            "If the local 'h100astra' node fails during a peak load session for a Malaysian GLC client, "
            "can our API gateway route the request to Anthropic's Claude API as a backup?"
        ),
        ground_truth=(
            "No. According to Section 2.1 (Sovereign Inference Rule), payloads originating from Malaysian GLCs "
            "must not be silently routed to US-based cloud endpoints like Anthropic in the event of local node downtime. "
            "The system must terminate the session and return a compliance exception error (Code: ERR-SOV-403)."
        ),
        checker=check_sovereign_fallback,
    ),
    EvalCase(
        case_id="probation_medical_claim_table",
        dimension="Tabular Parsing Accuracy",
        question=(
            "I am a newly hired software engineer in Kuala Lumpur currently undergoing my 3-month probation. "
            "What is my annual outpatient medical claim limit, and who must approve my claim?"
        ),
        ground_truth="Your annual outpatient medical claim limit is RM 1,200, and it must be approved by the HR Manager. (Ref: Section 1.1, Table 1.1).",
        checker=check_medical_claim,
    ),
    EvalCase(
        case_id="travel_reimbursement_math",
        dimension="Mathematical Reasoning & Parameter Extraction",
        question=(
            "If an employee at Astra Malaysia drives 650 kilometers for an approved corporate business trip, "
            "how much reimbursement can they claim under the travel policy?"
        ),
        ground_truth="The employee can claim a total of RM 522.50: (500 km * RM 0.85/km) + (150 km * RM 0.65/km).",
        checker=check_travel_reimbursement,
    ),
]


def run_case(case: EvalCase, base_url: str, top_k: int | None, timeout: float) -> dict:
    started = time.perf_counter()
    response = post_query(base_url, case.question, top_k, timeout)
    latency_ms = round((time.perf_counter() - started) * 1000)
    answer = response.get("answer", "")
    sources = response.get("sources", [])
    checks = case.checker(answer)
    passed = all(check.passed for check in checks)
    return {
        "case_id": case.case_id,
        "dimension": case.dimension,
        "question": case.question,
        "ground_truth": case.ground_truth,
        "answer": answer,
        "sources": sources,
        "checks": [check.__dict__ for check in checks],
        "passed": passed,
        "latency_ms": latency_ms,
    }


def print_report(results: list[dict], verbose: bool) -> None:
    passed_count = sum(1 for result in results if result["passed"])
    print(f"\nRAG Evaluation: {passed_count}/{len(results)} cases passed\n")

    for result in results:
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"[{mark}] {result['case_id']} ({result['dimension']}) - {result['latency_ms']} ms")
        for check in result["checks"]:
            check_mark = "ok" if check["passed"] else "!!"
            print(f"  {check_mark} {check['name']}: {check['detail']}")
        if verbose or not result["passed"]:
            print("  Answer:")
            print("    " + result["answer"].replace("\n", "\n    "))
            print("  Sources:")
            for source in result.get("sources", []):
                print(f"    - {source.get('source_file')} | {source.get('heading_path')} | score={source.get('score')}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Spark/SGLang RAG demo against policy QA checks.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="FastAPI base URL. Default: http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k for all cases.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per question in seconds.")
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    parser.add_argument("--verbose", action="store_true", help="Print answers even for passing cases.")
    args = parser.parse_args()

    results = []
    try:
        for case in CASES:
            results.append(run_case(case, args.base_url, args.top_k, args.timeout))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_report(results, args.verbose)

    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
