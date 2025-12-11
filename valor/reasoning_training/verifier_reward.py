import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from valor.reasoning_training.dummy_executor import mp_entry
import multiprocessing as mp

_THINK_RE = re.compile(r"<plan>(.*?)</plan>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text()


_API_PROMPT = _load_prompt("api_prompt.txt")
_LOGIC_REWARD_PROMPT = _load_prompt("logic_reward.txt")
_SPATIAL_REWARD_PROMPT = _load_prompt("spatial_reward.txt")
_ATTRIBUTE_REWARD_PROMPT = _load_prompt("attribute_reward.txt")
_CODE_PROMPT = _load_prompt("code_prompt.txt")


def get_genai_client():
    from google import genai
    from google.genai import types
    from google.oauth2 import service_account

    creds_path = os.environ.get("GENAI_CREDS_PATH")
    project = os.environ.get("GENAI_PROJECT_ID")
    if not creds_path or not project:
        raise EnvironmentError("GENAI_CREDS_PATH and GENAI_PROJECT_ID must be set")

    creds = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(
        vertexai=True,
        project=project,
        location="us-central1",
        credentials=creds,
        http_options=types.HttpOptions(api_version="v1"),
    )


def parse_answer(ans):
    m = re.search(r"(?si)<answer\b[^>]*>(.*?)</answer>", ans)
    return m.group(1).strip() if m else ans


def _strip_code_fences(s: str) -> str:
    """
    If the code is wrapped in Markdown fences (``` or ```python), remove them.
    """
    s = s.strip()
    # Matches ```python\n...\n``` or ```\n...\n```
    fence = re.compile(r"^\s*```(?:[a-zA-Z0-9_+-]*)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
    m = fence.match(s)
    return m.group(1) if m else s


def parse_llm_response(text: str) -> Dict[str, object]:
    """
    Scoring:
      - 0.0 if either <think>…</think> or <answer>…</answer> is missing, OR if there is
            any non-whitespace text outside those two tags (this rule takes precedence).
      - 0.5 if both tags are present but either appears more than once.
      - 1.0 if both tags appear exactly once and there's no outside text.
    """
    think_matches = list(_THINK_RE.finditer(text))
    answer_matches = list(_ANSWER_RE.finditer(text))

    # Extract only when exactly one match for each tag
    plan = think_matches[0].group(1).strip() if len(think_matches) == 1 else ""
    raw_code = answer_matches[0].group(1) if len(answer_matches) == 1 else ""
    code = _strip_code_fences(raw_code).strip() if raw_code else ""

    # Detect any non-whitespace text outside both tag blocks
    remainder = _THINK_RE.sub("", text)
    remainder = _ANSWER_RE.sub("", remainder)
    extraneous_text = bool(remainder.strip())

    # Scoring (outside text overrides everything)
    if len(think_matches) == 0 or len(answer_matches) == 0:
        score = 0.0
    elif len(think_matches) > 1 or len(answer_matches) > 1:
        score = 0.5
    else:
        score = 1.0

    return {"plan": plan, "code": code, "score": score}


def get_syntax_reward(code):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=mp_entry, args=(code, child))  # <- use module target
    proc.start()
    child.close()

    try:
        score, logs = parent.recv()
        if float(score) == 1.0:
            return 1.0
        else:
            return 0.0
    finally:
        if proc.is_alive():
            proc.kill()


def compute_all_rewards(query, llm_out):
    if not hasattr(compute_all_rewards, "client"):
        compute_all_rewards.client = get_genai_client()
    client = compute_all_rewards.client
    llm_response_parsed = parse_llm_response(llm_out)
    if llm_response_parsed["score"] == 0.0:
        return 0.0
    plan_text = llm_response_parsed["plan"]
    code = llm_response_parsed["code"]

    syntax_reward = get_syntax_reward(code)

    logic_reward = parse_answer(
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents=_LOGIC_REWARD_PROMPT.format(query, plan_text, _API_PROMPT),
        ).text
    )

    attribute_reward = parse_answer(
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents=_ATTRIBUTE_REWARD_PROMPT.format(query, plan_text, _API_PROMPT),
        ).text
    )

    code_reward = parse_answer(
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents=_CODE_PROMPT.format(plan_text, code, _API_PROMPT),
        ).text
    )

    spatial_reward = parse_answer(
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents=_SPATIAL_REWARD_PROMPT.format(query, plan_text, _API_PROMPT),
        ).text
    )

    try:
        logic_reward = int(logic_reward)
    except:
        logic_reward = 0
    try:
        attribute_reward = int(attribute_reward)
    except:
        attribute_reward = 0
    try:
        code_reward = int(code_reward)
    except:
        code_reward = 0
    try:
        spatial_reward = int(spatial_reward)
    except:
        spatial_reward = 0

    total = (
        0.3 * logic_reward
        + 0.1 * syntax_reward
        + 0.2 * attribute_reward
        + 0.2 * spatial_reward
        + 0.2 * code_reward
    )

    return total


# def compute_score_batched(
#     data_sources,
#     solution_strs,
#     ground_truths,
#     extra_infos,
#     *,
#     max_workers: int = 24,
#     timeout_sec: float = 30.0,
# ) -> list[float]:

#     if len(solution_strs) != len(extra_infos):
#         raise ValueError("Mismatched lengths")

#     def _get_question(info):
#         if isinstance(info, dict):
#             return info.get("problem", "")
#         return getattr(info, "problem", "")

#     results = [0.0] * len(solution_strs)

#     def _one(idx: int, code: str, info: Any):
#         q = _get_question(info)
#         results[idx] = compute_all_rewards(query=q, llm_out=code)

#     with ThreadPoolExecutor(max_workers=max_workers) as pool:
#         futs = {
#             pool.submit(_one, i, s, info): i
#             for i, (s, info) in enumerate(zip(solution_strs, extra_infos))
#         }
#         for fut in as_completed(futs):
#             _ = futs[fut]  # completion only; results set by index

#     return results


def compute_score_batched_dummy(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    max_workers: int = 24,
    timeout_sec: float = 30.0,
) -> list[float]:

    if len(solution_strs) != len(extra_infos):
        raise ValueError("Mismatched lengths")

    results = [0.0] * len(solution_strs)

    return results
