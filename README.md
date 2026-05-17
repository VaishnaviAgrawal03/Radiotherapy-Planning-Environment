---
title: Radiotherapy Planning Environment
emoji: "☢️"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# RadiotherapyPlanningEnv — RL Environment for Cancer Treatment Planning

> A Gymnasium-compatible reinforcement learning environment where an AI agent learns to plan cancer radiotherapy treatment — placing radiation beams to maximize tumor dose while protecting critical organs.

## Clinical Motivation

~14 million cancer patients per year require radiotherapy. A radiation oncologist must decide:
- How many beams to use
- At what angles
- With what dose intensity

...while ensuring the tumor receives enough radiation and nearby healthy organs stay below safe limits. This process takes human experts **2–4 hours per patient**. This environment simulates that decision-making process for RL agents.

**This is NOT a clinical tool.** It is a physically-grounded benchmark environment for testing RL algorithms on a meaningful real-world problem.

---

## Live Demo

Try the interactive Gradio demo on HuggingFace Spaces — watch a trained PPO agent plan treatment in real time, or play manually to compete against it:

**[https://huggingface.co/spaces/VaishnaviAgrawal/Radiotherapy_Planning_Environment](https://huggingface.co/spaces/VaishnaviAgrawal/Radiotherapy_Planning_Environment)**

---

## Quick Start

```bash
git clone https://github.com/VaishnaviAgrawal03/Radiotherapy-Planning-Environment.git
cd Radiotherapy-Planning-Environment
pip install -e .
```

```python
import gymnasium as gym
import radiotherapy_env

env = gym.make("RadiotherapyEnv-prostate-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Final score: {info['score']:.3f}")
env.close()
```

---

## Tasks — 3 Difficulty Levels

| Task | Gym ID | Difficulty | OARs | Max Steps | Clinical Context |
|------|--------|-----------|------|-----------|-----------------|
| **Prostate** | `RadiotherapyEnv-prostate-v1` | Easy | 2 (rectum, bladder) | 50 | Clear geometry, well-separated organs |
| **Head & Neck** | `RadiotherapyEnv-headneck-v1` | Medium | 7 (spinal cord, brainstem, parotids, etc.) | 60 | Complex anatomy, many competing constraints |
| **Pediatric Brain** | `RadiotherapyEnv-pediatricbrain-v1` | Hard | 5 (brainstem 2-3mm away) | 70 | Near-zero margin for error, catastrophic brainstem penalty |

---

## Baseline Results

### PPO Agent (stable-baselines3, MultiInputPolicy)

| Task | Mean Score | Std | Pass Rate | Training Steps |
|------|-----------|-----|-----------|---------------|
| Prostate | 0.697 | ±0.054 | **100%** | 200K |
| Head & Neck | 0.750 | ±0.059 | **96.7%** | 350K |
| Pediatric Brain | 0.717 | ±0.090 | **95.0%** | 1M |
| **Aggregate** | **0.721** | — | **97.2%** | — |

> Score ≥ 0.6 = clinically acceptable treatment plan.

### LLM Agent (Llama 3.3 70B via inference.py)

| Task | Mean Score | Episodes |
|------|-----------|----------|
| Prostate | 0.540 | 3 |
| Head & Neck | 0.560 | 3 |
| Pediatric Brain | 0.523 | 3 |
| **Aggregate** | **0.541** | 9 |

---

## Action Space — `Discrete(8)`

| Action | Description |
|--------|-------------|
| 0 | Add beam at next default angle |
| 1 | Rotate last beam +10° |
| 2 | Rotate last beam −10° |
| 3 | Increase last beam dose weight by 10% |
| 4 | Decrease last beam dose weight by 10% |
| 5 | Remove last beam |
| 6 | Fine-tune all beams (small random perturbation) |
| 7 | Lock plan (terminate episode) |

---

## Observation Space — `Dict`

| Key | Shape | Description |
|-----|-------|-------------|
| `dvh_tumor` | `Box(50,)` | Cumulative Dose-Volume Histogram for tumor |
| `dvh_oar` | `Box(3, 50)` | DVH for top 3 organs-at-risk |
| `beams` | `Box(7, 3)` | Per-beam: `[angle/180, dose_weight, is_active]` |
| `constraints` | `Box(4,)` | Normalized constraint violations `[tumor, oar1, oar2, oar3]` |
| `step_frac` | `Box(1,)` | Episode progress fraction `[0, 1]` |

All observations normalized to `[0, 1]` for neural network training stability.

---

## Reward Function

Dense per-step reward in `[0.0, 1.0]`:

```
reward = tumor_coverage × 0.55
       − oar_penalty    × 0.40   (priority-weighted: critical=1.5×, moderate=0.5×)
       + plan_efficiency × 0.05
```

- **Tumor coverage (55%):** D95 metric + fraction of tumor receiving ≥ 95% prescription dose
- **OAR penalty (40%):** Priority-weighted organ violations (critical organs penalized 1.5×)
- **Plan efficiency (5%):** Optimal beam count around 5–7

The reward provides **meaningful partial progress signals** — distinct from the stricter `compute_score()` used for final grading (which uses binary pass/fail for critical OARs).

---

## Physics Model

Gaussian pencil-beam dose calculation:

```
beam_dose = lateral_profile × depth_attenuation × dose_weight × BEAM_SCALE
```

- **Lateral profile:** Gaussian falloff from beam central axis (σ = 4.0)
- **Depth attenuation:** Exponential decay through tissue (Beer-Lambert Law, μ = 0.012)
- **Beam superposition:** Total dose = sum of all beam contributions
- **Isocenter convergence:** All beams aimed at tumor center

Simplified from clinical Monte Carlo (milliseconds vs. hours) but preserves the core trade-off: multiple beams overlap at the tumor for high dose while surrounding organs receive minimal radiation.

---

## Installation

```bash
# Core environment
pip install -e .

# With PPO training support
pip install -e ".[training]"

# With LLM inference support
pip install -e ".[inference]"

# With Gradio demo
pip install -e ".[demo]"
```

---

## Running the LLM Inference Script

Connects an LLM to the environment via an OpenAI-compatible API:

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export API_KEY="your_api_key"
python inference.py
```

Required log output format:

```
[START] task=prostate env=RadiotherapyEnv-prostate-v1 model=llama-3.3-70b-versatile
[STEP] step=1 action=add_beam reward=0.03 done=false error=null
[STEP] step=2 action=add_beam reward=0.07 done=false error=null
...
[END] success=true steps=50 score=0.635 rewards=0.03,0.07,...
```

---

## Training a PPO Agent

```bash
pip install -e ".[training]"
python baseline/train_ppo.py
```

Trains on all three tasks using 4 parallel vectorized environments. Saved checkpoints land in `baseline/models/`.

---

## Auto-Grader

```python
from radiotherapy_env.reward.grader import grade_all

def my_agent(obs, env):
    return env.action_space.sample()

results = grade_all(my_agent, n_episodes=20, seed=42)
print(f"Aggregate: {results['aggregate_score']:.3f}")
```

Each grader is deterministic with seed; pass threshold is 0.60.

---

## Running Tests

```bash
pytest tests/ -v
# 25 tests: Gymnasium compliance, physics, reward, task difficulty
```

---

## Docker

```bash
docker build -t radiotherapy-env:latest .
docker run -p 7860:7860 radiotherapy-env:latest
```

---

## OpenEnv HTTP Server

The FastAPI server exposes the environment over HTTP (required by `openenv validate`):

```bash
uv run server
# or
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Returns `{"status": "healthy"}` |
| `/metadata` | GET | Environment name, description, tasks |
| `/schema` | GET | Pydantic JSON schemas for action/observation |
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Apply action, returns next observation |
| `/state` | GET | Full internal environment state |

---

## Repository Structure

```
Radiotherapy-Planning-Environment/
├── inference.py               # LLM inference script
├── openenv.yaml               # OpenEnv spec metadata
├── Dockerfile                 # Container build (port 7860)
├── radiotherapy_env/          # Main Python package
│   ├── env.py                 # Core RadiotherapyEnv class
│   ├── physics/               # Dose calculator, DVH, patient models
│   ├── tasks/                 # 3 task definitions (prostate, head_neck, pediatric_brain)
│   ├── reward/                # Reward function, scoring, auto-grader
│   └── rendering/             # Dose heatmap + DVH visualization
├── server/                    # OpenEnv HTTP server (FastAPI)
├── baseline/                  # PPO training, evaluation, saved models, results
├── app/                       # Gradio interactive demo
└── tests/                     # 25-test pytest suite
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 64×64 grid | Balances fidelity with RL training speed |
| Max 7 beams | Clinically realistic IMRT constraint |
| Gaussian pencil-beam model | Fast (~ms) vs. Monte Carlo (minutes). Preserves core trade-offs |
| Discrete(8) action space | Clinically meaningful, small enough for exploration |
| Dense per-step rewards | Enables gradient-based RL learning (not sparse end-of-episode) |
| Normalized observations [0,1] | Improves neural network training stability |
| Three task tiers | Graduated difficulty for curriculum learning research |
| Separate reward vs. score | Training needs smooth gradients; evaluation needs strict clinical criteria |
| DVH as observation | Compact, rotation-invariant, clinically standard representation |
| Beam angles [0, 180) | Avoids redundancy (opposite angles are equivalent) |

---

## Author

**Vaishnavi Agrawal** — vagrawal_be22@thapar.edu
