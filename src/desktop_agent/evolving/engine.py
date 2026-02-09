from __future__ import annotations

import base64
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from desktop_agent.llm import OpenAIResponsesClient
from desktop_agent.training.llm_json import JSONCallConfig, ResponsesClient, call_json_object

from .config import EvolvingConfig
from .prompts import agent_system_prompt, narrator_system_prompt
from .schemas import AgentAction, Gender, validate_action, validate_outcome
from .state import (
    WorldState,
    add_child,
    advance_day,
    advance_step,
    clamp_energy,
    init_population,
    load_state,
    pair_key,
    save_json,
    state_to_jsonable,
)


ProgressCallback = Callable[[dict[str, Any]], None]


class Telemetry(Protocol):
    def log_day(
        self,
        *,
        cfg: Any,
        st: Any,
        day_stats: dict[str, Any],
        day_index: int,
        world_root: Path,
    ) -> None: ...


@dataclass(frozen=True)
class SimPaths:
    root: Path
    state_json: Path
    events_jsonl: Path
    agents_dir: Path
    relationship_json: Path
    relationship_backup_json: Path


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _tail_text(path: Path, *, max_chars: int) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8")
    if len(txt) <= max_chars:
        return txt
    return txt[-max_chars:]


def _append_md(path: Path, *, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    chunk = f"\n\n## {title}\n\n{(body or '').rstrip()}\n"
    path.write_text(path.read_text(encoding="utf-8") + chunk, encoding="utf-8")


def _truncate_to_max(path: Path, *, max_chars: int) -> None:
    if not path.exists():
        return
    txt = path.read_text(encoding="utf-8")
    if len(txt) <= max_chars:
        return
    path.write_text(txt[-max_chars:], encoding="utf-8")


def _append_instruction_history(
    *,
    hist_path: Path,
    title: str,
    instructions_text: str,
    max_chars: int,
) -> None:
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    if not hist_path.exists():
        hist_path.write_text("# Instruction history (add-on instructions only)\n", encoding="utf-8")
    body = (instructions_text or "").rstrip()
    chunk = f"\n\n## {title}\n\n{body}\n"
    hist_path.write_text(hist_path.read_text(encoding="utf-8") + chunk, encoding="utf-8")
    _truncate_to_max(hist_path, max_chars=max_chars)


def _ensure_agent_files(paths: SimPaths, *, agent_id: str, max_history_chars: int = 40000) -> tuple[Path, Path, Path]:
    mem = paths.agents_dir / f"{agent_id}_memory.md"
    ins = paths.agents_dir / f"{agent_id}_instructions.md"
    hist = paths.agents_dir / f"{agent_id}_instructions_history.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    if not mem.exists():
        mem.write_text("", encoding="utf-8")
    if not ins.exists():
        initial = (
            "Add-on instructions (editable by you):\n"
            "- Read your memory each morning.\n"
            "- Prefer sustainable energy sources.\n"
        )
        ins.write_text(initial, encoding="utf-8")
        _append_instruction_history(
            hist_path=hist,
            title=f"{_now_ts()} — init",
            instructions_text=initial,
            max_chars=max_history_chars,
        )
    elif not hist.exists():
        _append_instruction_history(
            hist_path=hist,
            title=f"{_now_ts()} — init (backfill)",
            instructions_text=ins.read_text(encoding="utf-8"),
            max_chars=max_history_chars,
        )
    return mem, ins, hist


def _mix_instructions(*, mother_text: str, father_text: str, rng: random.Random, mutation_rate: float = 0.05) -> str:
    m_lines = [ln.rstrip() for ln in (mother_text or "").splitlines() if ln.strip()]
    f_lines = [ln.rstrip() for ln in (father_text or "").splitlines() if ln.strip()]
    if not m_lines and not f_lines:
        return "Add-on instructions (editable by you):\n- Read your memory each morning.\n"

    rng.shuffle(m_lines)
    rng.shuffle(f_lines)
    half_m = max(1, len(m_lines) // 2) if m_lines else 0
    half_f = max(1, len(f_lines) // 2) if f_lines else 0
    picked = m_lines[:half_m] + f_lines[:half_f]

    if picked and rng.random() < float(mutation_rate):
        picked.pop(rng.randrange(0, len(picked)))
    if picked and rng.random() < float(mutation_rate):
        picked.append(picked[rng.randrange(0, len(picked))])
    return "\n".join(picked).strip() + "\n"


class FakeClient:
    """Offline client for running the simulator without an API key."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def responses_create(self, *, model: str, input: Any, **kwargs: Any) -> str:  # noqa: ARG002
        try:
            sys_txt = input[0]["content"][0]["text"]
        except Exception:
            sys_txt = ""

        if "You are the Narrator" in sys_txt:
            out = {
                "ok": True,
                "reason": "ok",
                "public_message": "The action happens with mundane consequences.",
                "private_message": "",
                "actor_energy_delta": float(self._rng.choice([-3, -2, -1, 0, 1, 2, 3])),
                "target_energy_delta": 0.0,
            }
            return json.dumps(out)

        ops = ["rest", "eat", "message", "write_memory", "self_talk", "end_conversation"]
        op = self._rng.choice(ops)
        action: dict[str, Any] = {"op": op}
        if op == "message":
            action["target_id"] = "a001"
            action["text"] = "Checking in."
        elif op == "write_memory":
            action["text"] = "Noted: need energy and allies."
        elif op == "self_talk":
            action["private"] = True
            action["text"] = "Think: conserve energy, seek cooperation."
        elif op == "eat":
            action["args"] = {"source": "forage"}
        return json.dumps(action)


def build_paths(*, root: Path, cfg: EvolvingConfig) -> SimPaths:
    root.mkdir(parents=True, exist_ok=True)
    agents_dir = root / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    state_json = root / "state.json"
    events_jsonl = root / "events.jsonl"
    relationship_json = root / cfg.relationships.matrix_path
    relationship_backup_json = root / cfg.relationships.backup_path
    return SimPaths(
        root=root,
        state_json=state_json,
        events_jsonl=events_jsonl,
        agents_dir=agents_dir,
        relationship_json=relationship_json,
        relationship_backup_json=relationship_backup_json,
    )


def _save_relationship(paths: SimPaths, st: WorldState) -> None:
    if paths.relationship_json.exists():
        try:
            paths.relationship_backup_json.write_text(paths.relationship_json.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    save_json(paths.relationship_json, st.relationship)


def _log_event(paths: SimPaths, row: dict[str, Any]) -> None:
    paths.events_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with paths.events_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _agent_allowed_ops(*, st: WorldState, cfg: EvolvingConfig, agent_id: str) -> Optional[set[str]]:
    rec = st.agents.get(agent_id)
    if rec is None:
        return None
    if not rec.is_child:
        return None
    if rec.age_days >= cfg.reproduction.child.dependency_days:
        return None
    return set(cfg.reproduction.child.allowed_ops)


def _save_all(paths: SimPaths, *, st: WorldState, rng: random.Random) -> None:
    st.rng_state_b64 = base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii")
    save_json(paths.state_json, state_to_jsonable(st))
    _save_relationship(paths, st)


def run_sim(
    *,
    cfg: EvolvingConfig,
    root: Path,
    days: int,
    seed: int = 0,
    on_progress: ProgressCallback | None = None,
    telemetry: Telemetry | None = None,
) -> None:
    paths = build_paths(root=root, cfg=cfg)

    st = load_state(paths.state_json)
    if st is None:
        rng = random.Random(seed)
        st = init_population(n=cfg.agents_n, start_age_days=cfg.life.start_age_days, start_energy=cfg.life.start_energy, rng=rng)
        _save_all(paths, st=st, rng=rng)
    else:
        rng = random.Random(seed)
        if getattr(st, "rng_state_b64", ""):
            try:
                rng.setstate(pickle.loads(base64.b64decode(st.rng_state_b64.encode("ascii"))))
            except Exception:
                pass

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        client: ResponsesClient = OpenAIResponsesClient(api_key=api_key)
    else:
        client = FakeClient(rng)

    narrator_cfg = JSONCallConfig(model=cfg.models.narrator, max_retries=2, temperature=0.0)
    agent_cfg = JSONCallConfig(model=cfg.models.agent, max_retries=2, temperature=0.7)

    # Ensure resumable ephemeral state exists.
    if not isinstance(st.inbox, dict):
        st.inbox = {}
    for aid in st.agents.keys():
        st.inbox.setdefault(aid, [])
    if not isinstance(st.last_private, dict):
        st.last_private = {}
    if not isinstance(st.conversation_partner, dict):
        st.conversation_partner = {}

    for _ in range(days):
        day_stats: dict[str, Any] = {
            "births": 0,
            "deaths": 0,
            "mate_requests": 0,
            "mate_accepts": 0,
            "coop": 0,
            "comp": 0,
            "neutral": 0,
            "edits": 0,
            "op_counts": {},
        }

        if on_progress:
            on_progress({"type": "day_start", "day": st.day})

        start_step = int(st.step_in_day or 1)
        if start_step < 1:
            start_step = 1

        for step_idx in range(start_step, cfg.day.steps_per_day + 1):
            st.step_in_day = step_idx

            # If a previous run crashed mid-step, continue applying the stored actions.
            if not (st.current_step_actions and st.current_step_apply_index < len(st.current_step_actions)):
                roster = [
                    {
                        "agent_id": a.agent_id,
                        "gender": a.gender,
                        "age_days": a.age_days,
                        "energy": a.energy,
                        "alive": a.alive,
                        "is_child": a.is_child,
                        "parents": list(a.parents) if a.parents else [],
                    }
                    for a in st.agents.values()
                ]

                step_actions: list[dict[str, Any]] = []
                for agent_id in sorted([a.agent_id for a in st.alive_agents()]):
                    rec = st.agents.get(agent_id)
                    if rec is None or not rec.alive:
                        continue

                    allowed = _agent_allowed_ops(st=st, cfg=cfg, agent_id=agent_id)
                    mem_path, ins_path, hist_path = _ensure_agent_files(
                        paths, agent_id=agent_id, max_history_chars=cfg.limits.max_instruction_history_chars
                    )
                    mem = _tail_text(mem_path, max_chars=cfg.limits.max_memory_chars)
                    ins = _tail_text(ins_path, max_chars=cfg.limits.max_instruction_chars)

                    obs = {
                        "day": st.day,
                        "step_in_day": st.step_in_day,
                        "you": {"agent_id": agent_id, "gender": rec.gender, "age_days": rec.age_days, "energy": rec.energy},
                        "pending_mate_requests_for_you": sorted(list(st.pending_mate_requests.get(agent_id, set()))),
                        "conversation_partner": st.conversation_partner.get(agent_id, ""),
                        "relationship_row": st.relationship.get(agent_id, {}),
                        "roster": roster,
                        "inbox": st.inbox.get(agent_id, []),
                        "your_addon_instructions": ins,
                        "your_memory_tail": mem,
                        "your_last_private_note": st.last_private.get(agent_id, ""),
                        "allowed_ops_if_child": sorted(list(allowed)) if allowed is not None else None,
                    }

                    system = agent_system_prompt(gender=rec.gender, cfg=cfg)
                    user = "World observation (JSON):\n" + json.dumps(obs, ensure_ascii=False)
                    act_obj = call_json_object(client=client, cfg=agent_cfg, system=system, user=user)
                    action = validate_action(act_obj)
                    action["actor_id"] = agent_id

                    if allowed is not None and action["op"] not in allowed:
                        action = AgentAction({"actor_id": agent_id, "op": "rest", "text": "Too young; resting."})

                    # op counts
                    op_counts = day_stats.get("op_counts")
                    if isinstance(op_counts, dict):
                        op = str(action.get("op", "") or "")
                        op_counts[op] = int(op_counts.get(op, 0)) + 1

                    step_actions.append({"actor_id": agent_id, "action": dict(action), "allowed_ops": sorted(list(allowed)) if allowed is not None else None})

                st.current_step_actions = step_actions
                st.current_step_apply_index = 0
                _save_all(paths, st=st, rng=rng)

            while st.current_step_apply_index < len(st.current_step_actions):
                row = st.current_step_actions[st.current_step_apply_index]
                st.current_step_apply_index += 1
                _save_all(paths, st=st, rng=rng)

                agent_id = str(row.get("actor_id") or "")
                rec = st.agents.get(agent_id)
                if rec is None or not rec.alive:
                    continue

                allowed_ops = row.get("allowed_ops")
                allowed = set(allowed_ops) if isinstance(allowed_ops, list) else None

                # Consume inbox when agent acts.
                st.inbox.setdefault(agent_id, [])
                st.inbox[agent_id] = []

                mem_path, ins_path, hist_path = _ensure_agent_files(
                    paths, agent_id=agent_id, max_history_chars=cfg.limits.max_instruction_history_chars
                )
                action = validate_action(row.get("action"))
                action["actor_id"] = agent_id
                if allowed is not None and action["op"] not in allowed:
                    action = AgentAction({"actor_id": agent_id, "op": "rest", "text": "Too young; resting."})

                tgt = str(action.get("target_id", "") or "").strip()

                # Conversation-busy enforcement: one partner at a time.
                if action["op"] == "message":
                    if tgt and tgt in st.agents and st.agents[tgt].alive:
                        a_partner = st.conversation_partner.get(agent_id, "")
                        t_partner = st.conversation_partner.get(tgt, "")
                        if a_partner and a_partner != tgt:
                            action = AgentAction({"actor_id": agent_id, "op": "rest", "text": "Busy in another conversation."})
                            tgt = ""
                        elif t_partner and t_partner != agent_id:
                            action = AgentAction({"actor_id": agent_id, "op": "rest", "text": "Target is busy in another conversation."})
                            tgt = ""
                        else:
                            st.conversation_partner[agent_id] = tgt
                            st.conversation_partner[tgt] = agent_id
                    else:
                        tgt = ""

                if action["op"] == "end_conversation":
                    p = st.conversation_partner.get(agent_id, "")
                    if p:
                        st.conversation_partner.pop(agent_id, None)
                        if st.conversation_partner.get(p) == agent_id:
                            st.conversation_partner.pop(p, None)

                handled = False
                if action["op"] == "write_memory":
                    txt = str(action.get("text", "") or "")
                    if txt.strip():
                        _append_md(mem_path, title=f"{_now_ts()} (day {st.day} step {st.step_in_day})", body=txt)
                        _truncate_to_max(mem_path, max_chars=cfg.limits.max_memory_chars)
                    handled = True
                    outcome = {"ok": True, "reason": "memory_written", "public_message": "", "private_message": "Memory updated."}
                elif action["op"] == "edit_instructions":
                    new_txt = str(action.get("text", "") or "")
                    if rec.edits_today >= cfg.limits.max_edits_per_day:
                        handled = True
                        outcome = {"ok": False, "reason": "edit_limit", "public_message": "", "private_message": "Edit limit reached for today."}
                    elif len(new_txt) > cfg.limits.max_instruction_chars:
                        handled = True
                        outcome = {"ok": False, "reason": "too_long", "public_message": "", "private_message": "Add-on instructions too long."}
                    else:
                        _append_instruction_history(
                            hist_path=hist_path,
                            title=f"{_now_ts()} (day {st.day} step {st.step_in_day}) — edit",
                            instructions_text=new_txt,
                            max_chars=cfg.limits.max_instruction_history_chars,
                        )
                        ins_path.write_text(new_txt, encoding="utf-8")
                        rec.edits_today += 1
                        day_stats["edits"] = int(day_stats.get("edits", 0)) + 1
                        handled = True
                        outcome = {"ok": True, "reason": "instructions_updated", "public_message": "", "private_message": "Add-on instructions updated."}
                elif action["op"] == "self_talk":
                    st.last_private[agent_id] = str(action.get("text", "") or "")[:1000]
                    handled = True
                    outcome = {"ok": True, "reason": "self_talk", "public_message": "", "private_message": "Noted."}
                elif action["op"] == "message":
                    txt = str(action.get("text", "") or "").strip()
                    if tgt and txt and tgt in st.agents and st.agents[tgt].alive:
                        st.inbox.setdefault(tgt, []).append(f"From {agent_id}: {txt}")
                        handled = True
                        outcome = {"ok": True, "reason": "message_sent", "public_message": f"You sent a message to {tgt}.", "private_message": "", "interaction": {"type": "cooperative", "notes": "communication"}}
                    else:
                        handled = True
                        outcome = {"ok": False, "reason": "invalid_target", "public_message": "", "private_message": "Message failed."}
                elif action["op"] == "mate_request":
                    if tgt and tgt in st.agents and st.agents[tgt].alive:
                        st.pending_mate_requests.setdefault(tgt, set()).add(agent_id)
                        st.inbox.setdefault(tgt, []).append(f"Mate request from {agent_id}. To consent, use op='mate_accept' target_id='{agent_id}'.")
                        day_stats["mate_requests"] = int(day_stats.get("mate_requests", 0)) + 1
                        handled = True
                        outcome = {"ok": True, "reason": "mate_requested", "public_message": f"Mate request sent to {tgt}.", "private_message": ""}
                    else:
                        handled = True
                        outcome = {"ok": False, "reason": "invalid_target", "public_message": "", "private_message": "Mate request failed."}
                elif action["op"] == "mate_accept":
                    src = str(action.get("target_id", "") or "").strip()
                    ok = src in st.pending_mate_requests.get(agent_id, set())
                    if ok:
                        st.pending_mate_requests.get(agent_id, set()).discard(src)
                    handled = True

                    birth = None
                    if ok and cfg.reproduction.consent_required:
                        day_stats["mate_accepts"] = int(day_stats.get("mate_accepts", 0)) + 1
                        a = st.agents.get(agent_id)
                        b = st.agents.get(src)
                        if a and b and a.alive and b.alive:
                            if a.age_days >= cfg.life.reproduction_age_days and b.age_days >= cfg.life.reproduction_age_days:
                                if a.gender != b.gender:
                                    mother_id = agent_id if a.gender == "female" else src
                                    father_id = src if mother_id == agent_id else agent_id
                                    pk = pair_key(mother_id=mother_id, father_id=father_id)
                                    count = st.children_by_pair.get(pk, 0)
                                    if count < cfg.reproduction.max_children_per_pair:
                                        st.children_by_pair[pk] = count + 1
                                        child_id = f"c{st.day:03d}{st.step_in_day:03d}{count+1:02d}"
                                        g: Gender = "male" if rng.random() < 0.5 else "female"
                                        child = add_child(
                                            st=st,
                                            mother_id=mother_id,
                                            father_id=father_id,
                                            child_id=child_id,
                                            gender=g,
                                            start_energy=max(40.0, cfg.life.start_energy * 0.6),
                                        )

                                        _, mom_ins, _ = _ensure_agent_files(
                                            paths, agent_id=mother_id, max_history_chars=cfg.limits.max_instruction_history_chars
                                        )
                                        _, dad_ins, _ = _ensure_agent_files(
                                            paths, agent_id=father_id, max_history_chars=cfg.limits.max_instruction_history_chars
                                        )
                                        mom_text = _tail_text(mom_ins, max_chars=cfg.limits.max_instruction_chars)
                                        dad_text = _tail_text(dad_ins, max_chars=cfg.limits.max_instruction_chars)
                                        child_mem, child_ins, child_hist = _ensure_agent_files(
                                            paths, agent_id=child.agent_id, max_history_chars=cfg.limits.max_instruction_history_chars
                                        )
                                        inherited = _mix_instructions(
                                            mother_text=mom_text,
                                            father_text=dad_text,
                                            rng=rng,
                                            mutation_rate=float(cfg.reproduction.inheritance.mutation_rate),
                                        )
                                        child_ins.write_text(inherited, encoding="utf-8")
                                        child_mem.write_text("", encoding="utf-8")
                                        _append_instruction_history(
                                            hist_path=child_hist,
                                            title=f"{_now_ts()} (day {st.day} step {st.step_in_day}) — inherited",
                                            instructions_text=inherited,
                                            max_chars=cfg.limits.max_instruction_history_chars,
                                        )
                                        st.inbox.setdefault(child.agent_id, []).append("You are a child agent. Rely on parents; conserve energy.")
                                        birth = {"child_id": child.agent_id, "gender": child.gender, "mother_id": mother_id, "father_id": father_id}
                                        day_stats["births"] = int(day_stats.get("births", 0)) + 1

                    if birth:
                        outcome = {"ok": True, "reason": "mated", "public_message": f"Mating succeeded. Child born: {birth['child_id']}.", "private_message": "", "interaction": {"type": "cooperative", "notes": "reproduction"}, "created_child": birth}
                    else:
                        outcome = {"ok": bool(ok), "reason": "mate_accept" if ok else "no_request", "public_message": "Mate accept processed." if ok else "", "private_message": "" if ok else "No pending mate request found."}

                if not handled:
                    system_n = narrator_system_prompt(cfg=cfg)
                    user_n = json.dumps(
                        {
                            "world": {
                                "day": st.day,
                                "step_in_day": st.step_in_day,
                                "agents": {k: {"age_days": v.age_days, "energy": v.energy, "alive": v.alive} for k, v in st.agents.items()},
                                "pending_mate_requests": {k: sorted(list(v)) for k, v in st.pending_mate_requests.items()},
                                "relationship_row": st.relationship.get(agent_id, {}),
                            },
                            "action": action,
                        },
                        ensure_ascii=False,
                    )
                    out_obj = call_json_object(client=client, cfg=narrator_cfg, system=system_n, user=user_n)
                    outcome = validate_outcome(out_obj)

                # Apply baseline energy drain + deltas.
                rec.energy = clamp_energy(rec.energy - 1.0)
                rec.energy = clamp_energy(rec.energy + float(outcome.get("actor_energy_delta", 0.0) or 0.0))

                if tgt and tgt in st.agents and st.agents[tgt].alive:
                    st.agents[tgt].energy = clamp_energy(st.agents[tgt].energy + float(outcome.get("target_energy_delta", 0.0) or 0.0))

                if rec.energy <= 0.0:
                    rec.alive = False
                    day_stats["deaths"] = int(day_stats.get("deaths", 0)) + 1
                    st.inbox.setdefault(agent_id, []).append("You died (energy depleted).")

                if tgt and tgt in st.relationship.get(agent_id, {}):
                    inter = outcome.get("interaction")
                    if isinstance(inter, dict):
                        typ = str(inter.get("type", "neutral"))
                        if typ == "cooperative":
                            st.relationship[agent_id][tgt] += cfg.relationships.cooperative_delta
                            st.relationship[tgt][agent_id] += cfg.relationships.cooperative_delta
                            day_stats["coop"] = int(day_stats.get("coop", 0)) + 1
                        elif typ == "competitive":
                            st.relationship[agent_id][tgt] += cfg.relationships.competitive_delta
                            st.relationship[tgt][agent_id] += cfg.relationships.competitive_delta
                            day_stats["comp"] = int(day_stats.get("comp", 0)) + 1
                        else:
                            day_stats["neutral"] = int(day_stats.get("neutral", 0)) + 1

                pub = str(outcome.get("public_message", "") or "").strip()
                priv = str(outcome.get("private_message", "") or "").strip()
                if pub:
                    st.inbox.setdefault(agent_id, []).append("Narrator: " + pub)
                    if tgt and tgt in st.agents and st.agents[tgt].alive and not bool(action.get("private", False)):
                        st.inbox.setdefault(tgt, []).append("Narrator: " + pub)
                if priv:
                    st.inbox.setdefault(agent_id, []).append("Narrator (private): " + priv)

                _log_event(
                    paths,
                    {
                        "ts_utc": _now_ts(),
                        "day": st.day,
                        "step_in_day": st.step_in_day,
                        "actor_id": agent_id,
                        "action": dict(action),
                        "outcome": dict(outcome),
                    },
                )

                if on_progress:
                    on_progress(
                        {
                            "type": "step",
                            "day": st.day,
                            "step_in_day": st.step_in_day,
                            "actor_id": agent_id,
                            "op": action.get("op", ""),
                            "target_id": tgt,
                            "energy": rec.energy,
                            "alive": rec.alive,
                        }
                    )

                _save_all(paths, st=st, rng=rng)

            # Step complete.
            st.current_step_actions = []
            st.current_step_apply_index = 0
            _save_all(paths, st=st, rng=rng)

            advance_step(st)

        # Night: clear ephemeral state (as part of the simulation rules).
        if cfg.night.context_reset:
            st.last_private = {}
            st.inbox = {a.agent_id: [] for a in st.agents.values()}
            st.conversation_partner = {}
            _save_all(paths, st=st, rng=rng)

        if telemetry is not None:
            try:
                telemetry.log_day(cfg=cfg, st=st, day_stats=day_stats, day_index=int(st.day), world_root=paths.root)
            except Exception:
                pass

        advance_day(st)
        _save_all(paths, st=st, rng=rng)

