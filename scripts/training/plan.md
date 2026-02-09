# Actor–Critic Curriculum (Adversarial Riddles) — Plan

## Goal

Build a training harness that runs repeated “episodes” where:

- **Riddler** generates a riddle + proposed solution, targeting a configurable “sweet spot” solve-rate (e.g., 50%).
- **Critic** verifies the riddle is valid/solvable given the provided solution, and grades solver answers (without “solving from scratch”).
- **Actor** orchestrates **N solvers** (default 10) with varied instructions; aggregates what worked; writes memory for future solver-instruction improvements.
- **Solvers** attempt the riddle; may request “help” via an Actor-created **partner model** for discussion.

Outputs:

- `scripts/training/riddler_memory.md` (what riddles were tried, observed solve-rate, reward signal)
- `scripts/training/actor_memory.md` (what solver instruction patterns worked / failed; distilled heuristics)
- A per-episode log artifact (JSONL or Markdown) for reproducibility.

---

## Repository Integration (how we’ll call models)

Use existing LLM plumbing:

- `src/desktop_agent/llm.py` → `OpenAIResponsesClient.responses_create(...)` for general multi-agent text calls.
- `src/desktop_agent/config.py` for `OPENAI_API_KEY` / model selection.

For training, we should standardize all agent calls to return **strict JSON** (so we can parse and grade reliably).

---

## Proposed File Layout (under `scripts/training/`)

- `plan.md` (this file)
- `config.json` (or `.toml`): training knobs (sweet spot, std, solver count, model ids, budgets)
- `run_episode.py`: one full riddler→actor→critic episode runner
- `agents.py`: prompt builders + call wrappers for each role (riddler/actor/solver/partner/critic)
- `schemas.py`: JSON schemas / typed dicts for messages and outputs
- `reward.py`: reward function implementation (Gaussian)
- `memory.py`: append-only memory writers/readers for `*_memory.md` (and optional JSONL logs)
- `artifacts/`:
  - `episode_YYYYMMDD_HHMMSS/` with `riddle.json`, `solver_runs.jsonl`, `critic_report.json`, `summary.md`

---

## Quickstart (current implementation)

Prereqs:
- Set `OPENAI_API_KEY` (recommended via repo-root `.env`)

Run one episode:

```powershell
.\.venv\Scripts\python .\scripts\training\run_episode.py
```

Run many episodes (curriculum loop):

```powershell
.\.venv\Scripts\python .\scripts\training\run_curriculum.py
```

What you’ll get:
- New artifacts under `scripts/training/artifacts/episode_*`
- Updated `scripts/training/riddler_memory.md` and `scripts/training/actor_memory.md`
- (If enabled) a growing dataset file `scripts/training/dataset.jsonl`

Progress visibility:
- The runner prints the riddle up front.
- It shows solver progress (uses `tqdm` if installed; otherwise prints `[i/N]` lines).
- Set `stream_grade_each_solver=true` in `scripts/training/config.json` to have the critic grade each solver as it finishes so the progress output can show “solved so far”.
- Set `status_every_solvers` (e.g., `2`) to have the actor periodically summarize what’s happening and which remaining solvers are likely to succeed.
- While it runs, you can “talk to the actor” by writing a question into `scripts/training/actor_inbox.md`; answers are appended to `scripts/training/actor_outbox.md`.

---

## Roles, Interfaces, and Contracts

### 1) Riddler (adversary)
**Objective:** adjust riddle difficulty so the solver ensemble hits the target solve-rate.

**Input:**
- Training config (target solve-rate, tolerance, max length, allowed domains)
- `riddler_memory.md` excerpts (recent episodes summary)

**Output (JSON):**
```json
{
  "riddle_id": "string",
  "riddle": "string",
  "solution": "string",
  "solution_rationale": "string",
  "tags": ["logic", "wordplay"],
  "difficulty_guess_pct": 50
}
```

**Notes:**
- Riddler must provide a solution + rationale so the critic can verify validity without independently solving.
- Riddler should track what it tried and the observed solve-rate; store “adjustments” in `riddler_memory.md`.

### 2) Actor (orchestrator)
**Objective:** maximize solver success over time by learning better solver initialization strategies.

**Input:**
- Riddle payload (riddle text + any constraints)
- `actor_memory.md` excerpts (best known solver instruction templates)
- Config (N solvers, partner budget, max turns)

**Behavior:**
- Spawn `N` solver instances with intentionally diverse instructions (e.g., styles, decomposition strategies, verification habits).
- Allow solver-to-actor “help requests”; if requested, Actor spawns a “partner model” to discuss with that solver.
- Collect solver final answers, plus short self-reported confidence and checks performed.

**Output (JSON):**
```json
{
  "solver_runs": [
    {
      "solver_id": "s01",
      "instruction_profile": "string",
      "final_answer": "string",
      "confidence": 0.0,
      "asked_partner": false,
      "partner_summary": "string"
    }
  ],
  "actor_summary": "string",
  "next_instruction_hypotheses": ["string"]
}
```

### 3) Solver (worker)
**Objective:** solve the riddle given its instructions and context.

**Input:**
- Solver-specific instructions (unique per solver)
- Riddle text
- Optional discussion context with partner

**Output (JSON):**
```json
{
  "final_answer": "string",
  "confidence": 0.0,
  "checks": ["string"],
  "help_request": {
    "needed": false,
    "what_i_tried": "string",
    "where_i_am_stuck": "string"
  }
}
```

**Important implementation note (reasoning privacy):**
- Solvers may “think” internally, but should return only the structured fields above (no hidden chain-of-thought text persisted).

### 4) Critic (verifier + grader)
**Objective:**
- Verify riddle validity/solvability given the riddler’s proposed solution (consistency checks).
- Grade each solver answer as correct/incorrect based on the riddler solution and explicit criteria.
- Compute reward for riddler based on achieved solve-rate vs sweet spot.

**Input:**
- Riddle + solution (+ rationale) from riddler
- All solver outputs
- Reward config (mean = sweet spot, std = 10% default)

**Output (JSON):**
```json
{
  "riddle_valid": true,
  "validity_notes": "string",
  "grading": [
    {"solver_id": "s01", "correct": true, "notes": "string"}
  ],
  "solve_rate_pct": 50.0,
  "reward": 1.0
}
```

---

## Reward Function (Gaussian)

Let:
- `p` = achieved solve-rate in percent (0–100)
- `μ` = sweet spot percent (e.g., 50)
- `σ` = standard deviation in percent (e.g., 10)

Reward:

`reward = exp(-0.5 * ((p - μ) / σ)^2)`

Optional extensions (later):
- Penalize invalid riddles: if `riddle_valid == false` then `reward = 0`
- Add novelty bonus / repetition penalty via memory similarity

---

## Episode Flow (single run)

1) Load config + recent memory snippets.
2) Riddler generates `{riddle, solution, rationale, ...}`.
3) Critic validates the riddle/solution consistency (reject or request regeneration if invalid).
4) Actor spawns N solvers (and optional partners on request).
5) Critic grades solver answers → solve-rate + reward.
6) Update memories:
   - Append to `riddler_memory.md`: riddle_id, tags, difficulty_guess, solve-rate, reward, notes on difficulty tuning.
   - Append to `actor_memory.md`: which instruction profiles correlated with success, common failure modes, next hypotheses.
7) Save episode artifacts for debugging and later analysis.

---

## Prompting Strategy (instructions vs context)

For every role call:

- **System message**: role definition + JSON schema + “output JSON only”.
- **User message**: episode-specific context (riddle, memory excerpts, config, partner discussion text).

Recommended: keep memory excerpts short and structured (last ~5 episodes), and put “long-term distilled heuristics” in a stable section in each memory file.

---

## Implementation Phases (suggested)

### Phase 0 — Scaffolding
- Add `scripts/training/config.json` with defaults:
  - `sweet_spot_pct=50`, `std_pct=10`, `solver_count=10`
  - `models`: `{riddler, actor, solver, partner, critic}`
  - budgets: max tokens/turns, max partner count per episode

### Phase 1 — Deterministic JSON contracts
- Implement `agents.py` Use `OpenAIResponsesClient` + `text={"format":{"type":"json_object"}}` for every role.
- Implement robust JSON extraction + validation (mirror approach in `src/desktop_agent/llm.py`).

### Phase 2 — Episode runner
- Implement `run_episode.py` end-to-end with artifact saving.
- Implement `reward.py`, `memory.py`.

### Phase 3 — Curriculum loop
- Add `run_curriculum.py` to run many episodes, periodically compact memories into “distilled” sections.
- Add metrics dashboard (CSV/JSONL): solve-rate, reward, invalid-rate, cost estimates.

---

## Edits / Additions I Recommend

1) **Avoid persisting chain-of-thought.** Have solvers return only `final_answer`, `confidence`, and short `checks`. This keeps logs clean and reduces prompt leakage risk.
2) **Make N solvers configurable** (not hard-coded 10) + allow early stopping if solve-rate is clearly above/below target to save tokens.
3) **Add strict schemas + grading criteria per riddle type.** Wordplay vs math vs logic riddles grade differently; require the riddler to specify grading rules.
4) **Guardrail invalid riddles.** If critic marks invalid, the episode should be “no-op” with reward 0 and an automatic “regenerate riddle” retry budget.
5) **Use append-only logs + periodic compaction.** Keep `*_memory.md` human-readable, but also write `artifacts/.../*.jsonl` for machine analysis.
6) **Cap partner help.** Add `max_partners_per_solver` and `max_total_partners` to prevent runaway debates.
