"""
Inference Script — RadiotherapyPlanningEnv
==========================================
Connects an LLM (via OpenAI-compatible API) to the RadiotherapyPlanningEnv
and runs it through all 3 tasks (prostate, head_neck, pediatric_brain).

Required env vars:
    API_BASE_URL  — LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME    — Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — HuggingFace / API key

Usage:
    python inference.py
"""

import os
import textwrap
from typing import List, Optional

import numpy as np
import gymnasium as gym
from openai import OpenAI

import radiotherapy_env  # noqa: F401 — registers gym envs

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

TEMPERATURE = 0.3
MAX_TOKENS = 100

# Task configurations
TASKS = [
    {"key": "prostate",        "env_id": "RadiotherapyEnv-prostate-v1",       "max_steps": 50, "difficulty": "easy"},
    {"key": "head_neck",       "env_id": "RadiotherapyEnv-headneck-v1",       "max_steps": 60, "difficulty": "medium"},
    {"key": "pediatric_brain", "env_id": "RadiotherapyEnv-pediatricbrain-v1", "max_steps": 70, "difficulty": "hard"},
]

N_EPISODES = 3  # episodes per task (kept low for < 20 min runtime)
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert AI radiation oncologist. Your goal: maximize tumor dose
    while keeping organs-at-risk (OARs) within safe limits.
    Score ≥ 0.60 = clinically acceptable. Target score ≥ 0.75.

    ACTIONS (reply with ONLY the digit, nothing else):
      0 — Add beam at next default angle
      1 — Rotate last beam +10°
      2 — Rotate last beam -10°
      3 — Increase last beam dose weight +10%
      4 — Decrease last beam dose weight -10%
      5 — Remove last beam
      6 — Fine-tune ALL beams (small angle + dose adjustments)
      7 — Lock plan and finish

    PROVEN STRATEGY (follow this phase plan):
      Phase 1 — BUILD (first ~35 steps):
        Repeat this per beam: action 0, then action 3 four times (boosts dose 0.6→1.0).
        Do this for all 7 beams. 7 evenly-spaced beams at max dose = ~95% tumor coverage.

      Phase 2 — OPTIMIZE (remaining steps):
        • Tumor uncoverage > 0.15  → action 6 (fine-tune angles)
        • OAR violation > 0.5      → action 4 (reduce dose)
        • OAR violation 0.2–0.5    → action 1 or 2 (rotate away)
        • Everything looks good     → action 6 (fine-tune)
        • Near end + score > 0.70  → action 7 (lock plan)

    KEY RULES:
      - NEVER use action 7 before step 40 unless score > 0.75
      - Tumor uncoverage 0.0 = perfect, 1.0 = no coverage
      - OAR violation 0.0 = safe, > 0.3 = dangerous

    REPLY WITH ONE DIGIT ONLY (0-7).
""").strip()


# ── Logging (exact format required by hackathon) ─────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation formatting ───────────────────────────────────────────────────

def format_observation(obs: dict, info: dict, step: int, max_steps: int) -> str:
    """Convert numeric observation into rich text for the LLM."""
    # Beams
    beams_desc = []
    for i in range(7):
        angle_norm, dose_weight, active = obs["beams"][i]
        if active > 0.5:
            angle_deg = angle_norm * 180.0
            bar = "█" * int(dose_weight * 10)
            beams_desc.append(
                f"  Beam {i+1}: {angle_deg:5.0f}° | dose={dose_weight:.2f} {bar}"
            )
    beams_text = "\n".join(beams_desc) if beams_desc else "  No beams placed yet"
    n_beams = len(beams_desc)

    # Constraints
    c = obs["constraints"]
    tumor_uncov  = float(c[0])
    oar1_viol    = float(c[1])
    oar2_viol    = float(c[2])
    oar3_viol    = float(c[3])
    max_oar_viol = max(oar1_viol, oar2_viol, oar3_viol)

    # DVH
    dvh_text = ""
    if "dvh_summary" in info:
        dvh = info["dvh_summary"]
        cov  = dvh.get("tumor_coverage", 0.0)
        d95  = dvh.get("tumor_d95", 0.0)
        dvh_text = (
            f"  Tumor D95:      {d95:.3f}  (need ≥ 0.95)\n"
            f"  Tumor coverage: {cov:.1%} (need ≥ 95%)"
        )

    score = info.get("score", 0.0)
    phase = "BUILD" if step <= 35 else "OPTIMIZE"

    # Decide the recommended action based on phase logic
    if phase == "BUILD":
        if n_beams < 7 and (step - 1) % 5 == 0:
            hint = "→ RECOMMENDED: 0 (add beam — build phase)"
        else:
            hint = "→ RECOMMENDED: 3 (boost last beam dose — build phase)"
    else:
        if tumor_uncov > 0.15:
            hint = "→ RECOMMENDED: 6 (fine-tune — tumor undercovered)"
        elif max_oar_viol > 0.5:
            hint = "→ RECOMMENDED: 4 (reduce dose — OAR badly violated)"
        elif max_oar_viol > 0.2:
            hint = "→ RECOMMENDED: 1 or 2 (rotate — OAR moderately violated)"
        elif step > max_steps - 5:
            hint = "→ RECOMMENDED: 7 (lock plan — near end)"
        else:
            hint = "→ RECOMMENDED: 6 (fine-tune — plan looking good)"

    return textwrap.dedent(f"""
        ═══ Step {step}/{max_steps} | Phase: {phase} | Beams: {n_beams}/7 | Score: {score:.3f} ═══

        BEAMS:
        {beams_text}

        CONSTRAINTS  (0.0=perfect, higher=worse):
          Tumor uncoverage : {tumor_uncov:.3f}  {'⚠ CRITICAL' if tumor_uncov > 0.3 else '✓ ok' if tumor_uncov < 0.1 else '~ moderate'}
          OAR 1 violation  : {oar1_viol:.3f}  {'⚠ DANGER' if oar1_viol > 0.5 else ''}
          OAR 2 violation  : {oar2_viol:.3f}  {'⚠ DANGER' if oar2_viol > 0.5 else ''}
          OAR 3 violation  : {oar3_viol:.3f}  {'⚠ DANGER' if oar3_viol > 0.5 else ''}

        DVH METRICS:
        {dvh_text}

        {hint}
        Choose action (0-7):
    """).strip()


# ── LLM interaction ──────────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, obs: dict, info: dict, step: int,
                   max_steps: int, history: List[str]) -> int:
    """Ask the LLM to choose an action given the current state."""
    user_prompt = format_observation(obs, info, step, max_steps)

    # Add recent history for context
    if history:
        recent = "\n".join(history[-5:])
        user_prompt = f"Recent actions:\n{recent}\n\n{user_prompt}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Parse action — find first digit 0-7 in response
        for char in text:
            if char.isdigit() and int(char) <= 7:
                return int(char)

        # Fallback: mirror the proven phase strategy
        n_beams = int(sum(1 for row in obs["beams"] if row[2] > 0.5))
        if step <= 35:
            return 0 if (n_beams < 7 and (step - 1) % 5 == 0) else 3
        c = obs["constraints"]
        if float(c[0]) > 0.15:
            return 6
        if max(float(c[1]), float(c[2]), float(c[3])) > 0.5:
            return 4
        return 6

    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        n_beams = int(sum(1 for row in obs["beams"] if row[2] > 0.5))
        if step <= 35:
            return 0 if (n_beams < 7 and (step - 1) % 5 == 0) else 3
        return 6 if step < max_steps - 2 else 7


# ── Episode runner ───────────────────────────────────────────────────────────

def run_episode(client: OpenAI, env_id: str, task_key: str, max_steps: int,
                seed: int) -> dict:
    """Run one episode: LLM plays the radiotherapy env."""
    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=task_key, env=env_id, model=MODEL_NAME)

    try:
        for step in range(1, max_steps + 1):
            action = get_llm_action(client, obs, info, step, max_steps, history)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards.append(float(reward))
            steps_taken = step

            action_names = [
                "add_beam", "rotate_+10", "rotate_-10",
                "increase_dose", "decrease_dose", "remove_beam",
                "fine_tune", "lock_plan"
            ]
            action_str = action_names[action]

            last_error = info.get("last_action_error", None)
            log_step(step=step, action=action_str, reward=float(reward),
                     done=done, error=last_error)

            history.append(f"Step {step}: {action_str} -> reward {reward:.2f}")

            if done:
                break

        final_score = info.get("score", 0.0)
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        final_score = 0.0
        success = False

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return {
        "task": task_key,
        "score": final_score,
        "steps": steps_taken,
        "rewards": rewards,
        "success": success,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_results = {}

    for task_cfg in TASKS:
        task_key = task_cfg["key"]
        env_id = task_cfg["env_id"]
        max_steps = task_cfg["max_steps"]
        difficulty = task_cfg["difficulty"]

        print(f"\n{'='*60}", flush=True)
        print(f"  Task: {task_key} ({difficulty})", flush=True)
        print(f"  Running {N_EPISODES} episodes, max {max_steps} steps each", flush=True)
        print(f"{'='*60}\n", flush=True)

        task_scores = []
        for ep in range(N_EPISODES):
            print(f"\n--- Episode {ep+1}/{N_EPISODES} ---\n", flush=True)
            result = run_episode(client, env_id, task_key, max_steps, seed=42 + ep)
            task_scores.append(result["score"])

        mean_score = float(np.mean(task_scores))
        all_results[task_key] = {
            "difficulty": difficulty,
            "mean_score": mean_score,
            "scores": task_scores,
            "pass_rate": float(np.mean(np.array(task_scores) >= 0.6)),
        }

        print(f"\n  {task_key} mean score: {mean_score:.3f}", flush=True)

    # Final summary
    aggregate = float(np.mean([r["mean_score"] for r in all_results.values()]))
    print(f"\n{'='*60}", flush=True)
    print(f"  AGGREGATE SCORE: {aggregate:.3f}", flush=True)
    for key, res in all_results.items():
        print(f"    {key}: {res['mean_score']:.3f} (pass rate: {res['pass_rate']*100:.0f}%)", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
