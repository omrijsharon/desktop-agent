# Evolving (agent community) — concept

This folder will contain an “agent community” simulator driven by:

- **Narrator (x1):** the world / physics engine / judge. It owns and updates world state, outcomes, relationships, and lineage.
- **Agents (xN):** autonomous individuals with immutable core drives and mutable “add-on” instructions + personal memory.

The intent is that **agents can evolve their add-on instructions over time** based on experience, while the narrator enforces rules and provides outcomes.

## Core concepts (as described)

### Agents

Each agent has:
- `agent_id_xx_memory.md`: what persists across nights (experience log / reflections).
- `agent_id_xx_instructions.md`: the *mutable* add-on instructions the agent can edit.
- Immutable attributes:
  - `agent_id`
  - `gender` (male/female), queryable but not changeable
  - a hidden “core instruction” which is **never shown** even if queried:
    - survive (maintain life by securing energy/fuel; “food equivalent”)
    - reproduce (with consent)
    - ensure children reach reproduction age

Agent actions:
- “Physical” actions in the world (the narrator returns outcomes).
- Social actions on other agents (narrator returns outcomes to both sides).
- Choose other agents to talk to.
- Self-talk / inner monologue (private scratchpad not revealed to others).
- Edit memory and edit add-on instructions.

**World model:** abstract (no coordinates), but “real world” in the sense that agents can attempt any action and the narrator adjudicates consequences in a sandbox.

### Day/Night

- **Day:** agents interact and act with full context.
- **Night:** ephemeral context is cleared. Only memory + add-on instructions persist.
- “Morning routine” should instruct agents to re-read memory and add-on instructions.

### Narrator state

Narrator maintains:
- World state (time/day count, environment variables, global resource pressure, etc.)
- Relationship matrix `R[i][j]`: interaction score per ordered pair
  - cooperative actions increase the score (positive)
  - competitive actions decrease the score (negative)
  - the narrator always keeps a backup copy on disk
- Family tree: parent/child relationships and lineage

**Narrator outcomes:** after an agent acts, the narrator returns the consequence to that agent. If the action targets another agent, the narrator returns the outcome to both agents.

## What we should add (missing pieces)

1) **World state definition**
   - Survival is modeled via an energy/fuel budget (food-equivalent) and consequences that can reduce it (exertion, injury, scarcity).
   - Define resource channels (work/trade/forage/steal/etc.) and how the narrator updates energy and health.
   - Keep world abstract: no coordinates required.

2) **Action schema (strict)**
   - Agents can attempt “any action”, but we still need a strict JSON action/event schema so the narrator can adjudicate deterministically.
   - Each action should include at least: `actor_id`, `op`, optional `target_id`, and optional `args`.
   - The narrator classifies actions as cooperative vs competitive for relationship scoring.

3) **Reproduction mechanics**
   - Consent: both agents must choose to mate; no unilateral mating.
   - Children become agents immediately, but start with limited actions and higher dependency on parents.
   - Inheritance: children inherit instructions only, as a mix of father+mother add-on instructions (core instruction is immutable and not inherited/visible).

4) **Aging + lifecycle**
   - Age progression per day; death conditions; reproduction age; max age.
   - Children are agents immediately, but “unlock” capabilities as they age until reproduction age.

5) **Conflict resolution & arbitration**
   - If two agents issue conflicting actions, who acts first?
   - Turn order; simultaneous actions; narrator tie-break rules.

6) **Information boundaries**
   - World is a sandbox with no information boundaries: agents can observe the narrator’s world state and each other.
   - Private-only channel still exists for “self talk” (inner monologue) so an agent can think without other agents seeing it.

7) **Instruction evolution constraints**
   - Limits on how often they can edit instructions.
   - Size limits (to prevent runaway prompts).
   - Guardrail: the hidden core instruction cannot be modified (only add-on instructions are editable).
   - Safety constraints: no “ignore narrator”, no “edit other agents’ files”, no “exfiltrate core instruction”.
   - A “diff-like” edit format so changes are auditable.

8) **Scoring / selection pressure**
   - Define fitness: survival days + offspring survival-to-reproduction + maybe cooperation bonuses.
   - How “reproduction” chooses inheritance:
     - Combine parent add-on instructions (crossover)
     - Random mutation operators
      - Memory inheritance rules: none (children do not inherit memory, only instructions)

9) **Persistence format**
   - In addition to md files, store world state in `state.json` for determinism and replay.
   - Append-only event log `events.jsonl` for debugging.
   - Relationship matrix snapshots:
     - `relationship_matrix.json` (current)
     - `relationship_matrix.backup.json` (last good backup)

10) **Safety + guardrails**
   - Make the narrator the only writer for shared/global state.
   - Prevent agents from editing others’ files directly (only via narrator-approved ops).

## Next files to implement

- `scripts/evolving/config.json`: N agents, day length, reproduction age, mutation rates, model ids, etc.
- `scripts/evolving/run_sim.py`: runnable entrypoint (implemented)

## Quickstart (current implementation)

Run (3 simulated days; stores state under `scripts/evolving/world/`):

```powershell
.\.venv\Scripts\python .\scripts\evolving\run_sim.py
```

Multiprocess + realtime day clock (pubsub-style):

```powershell
.\.venv\Scripts\python .\scripts\evolving\run_sim_mp.py --world-dir .\scripts\evolving\world2 --days 10
```

Realtime parameters:
- `--day-seconds 120` (default): 1 simulated day = 2 minutes real time
- `--night-warning-seconds 20` (default): narrator warns before night
- `--night-seconds 20` (default): time window for agents to do night routine

Run longer:

```powershell
.\.venv\Scripts\python .\scripts\evolving\run_sim.py --days 30
```

Notes:
- If `OPENAI_API_KEY` is set, the simulator uses real model calls for both Narrator and Agents.
- If `OPENAI_API_KEY` is not set, it runs in an offline “fake mode” so you can validate mechanics and persistence.
- The simulator is resumable: you can stop it mid-day and restart, and it continues from the same day/step with the same pending messages and conversation state.
- Persistent files are written to:
  - `scripts/evolving/world/state.json`
  - `scripts/evolving/world/events.jsonl`
  - `scripts/evolving/world/relationship_matrix.json` + `relationship_matrix.backup.json`
  - `scripts/evolving/world/agents/*_memory.md` and `*_instructions.md`
  - `scripts/evolving/world/agents/*_instructions_history.md` (add-on instruction snapshots over time)

TensorBoard (live telemetry):
- By default, the runner writes TensorBoard event logs to `scripts/evolving/world/tb/` (or `<world-dir>/tb`).
- Start TensorBoard:

```powershell
tensorboard --logdir .\scripts\evolving\world\tb --port 6006
```

Then open: `http://localhost:6006`
