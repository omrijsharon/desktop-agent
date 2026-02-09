from __future__ import annotations

import base64
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from multiprocessing import Event as MpEvent
from multiprocessing import Process, Queue, get_context
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from desktop_agent.llm import OpenAIResponsesClient
from desktop_agent.training.llm_json import JSONCallConfig, ResponsesClient, call_json_object

from .config import EvolvingConfig
from .engine import (
    SimPaths,
    _append_instruction_history,
    _append_md,
    _agent_allowed_ops,
    _ensure_agent_files,
    _log_event,
    _mix_instructions,
    _now_ts,
    _save_relationship,
    _tail_text,
    _truncate_to_max,
    build_paths,
)
from .prompts import agent_system_prompt, narrator_system_prompt
from .schemas import AgentAction, Gender, validate_action, validate_outcome
from .state import (
    add_child,
    advance_day,
    clamp_energy,
    init_population,
    load_state,
    pair_key,
    save_json,
    state_to_jsonable,
)


class Telemetry(Protocol):
    def log_day(self, *, cfg: Any, st: Any, day_stats: dict[str, Any], day_index: int, world_root: Path) -> None: ...

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RealtimeConfig:
    day_seconds: float = 120.0
    night_seconds: float = 20.0
    night_warning_seconds: float = 20.0
    agent_cooldown_s: float = 1.0
    status_every_s: float = 1.0


def _b64_state(rng: random.Random) -> str:
    return base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii")


def _restore_rng(seed: int, b64: str) -> random.Random:
    rng = random.Random(seed)
    if b64:
        try:
            rng.setstate(pickle.loads(base64.b64decode(b64.encode("ascii"))))
        except Exception:
            pass
    return rng


def _agent_worker(
    *,
    agent_id: str,
    gender: str,
    cfg_dict: dict[str, Any],
    api_key: Optional[str],
    in_q: "Queue[dict[str, Any]]",
    out_q: "Queue[dict[str, Any]]",
    stop_evt: MpEvent,
) -> None:
    # Reconstruct config. Dataclasses are picklable, but this keeps the payload minimal.
    cfg = EvolvingConfig(**cfg_dict)  # type: ignore[arg-type]

    if api_key:
        client: ResponsesClient = OpenAIResponsesClient(api_key=api_key)
    else:
        class _Fake:
            def responses_create(self, *, model: str, input: Any, **kwargs: Any) -> str:  # noqa: ARG002
                return json.dumps({"op": "rest", "text": "No API key; resting."})

        client = _Fake()

    call_cfg = JSONCallConfig(model=cfg.models.agent, max_retries=2, temperature=0.7)

    while not stop_evt.is_set():
        try:
            msg = in_q.get(timeout=0.2)
        except Exception:
            continue
        typ = msg.get("type")
        if typ == "shutdown":
            return

        if typ in {"request_action", "night_routine"}:
            obs = str(msg.get("observation_json", "") or "")
            system = agent_system_prompt(gender=gender, cfg=cfg)
            if typ == "night_routine":
                user = (
                    "Night has come. Finish up quickly and go to sleep.\n"
                    "Make ONE final action that helps tomorrow (prefer write_memory or edit_instructions).\n"
                    "World observation (JSON):\n"
                    + obs
                )
            else:
                user = "World observation (JSON):\n" + obs

            try:
                obj = call_json_object(client=client, cfg=call_cfg, system=system, user=user)
                action = validate_action(obj)
            except Exception as e:
                action = AgentAction({"op": "rest", "text": f"error: {e}"})

            out_q.put(
                {
                    "type": "night_action" if typ == "night_routine" else "action",
                    "agent_id": agent_id,
                    "action": dict(action),
                    "ts": time.time(),
                }
            )


def run_sim_mp(
    *,
    cfg: EvolvingConfig,
    root: Path,
    days: int,
    seed: int = 0,
    realtime: RealtimeConfig = RealtimeConfig(),
    telemetry: Telemetry | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    paths: SimPaths = build_paths(root=root, cfg=cfg)

    st = load_state(paths.state_json)
    if st is None:
        rng = random.Random(seed)
        st = init_population(n=cfg.agents_n, start_age_days=cfg.life.start_age_days, start_energy=cfg.life.start_energy, rng=rng)
        st.phase = "day"
        st.day_remaining_s = float(realtime.day_seconds)
        st.rng_state_b64 = _b64_state(rng)
        save_json(paths.state_json, state_to_jsonable(st))
        _save_relationship(paths, st)
    else:
        rng = _restore_rng(seed, getattr(st, "rng_state_b64", ""))
        if not st.day_remaining_s:
            st.day_remaining_s = float(realtime.day_seconds)

    # Ensure resumable ephemeral fields exist.
    st.inbox = st.inbox or {}
    for aid in st.agents.keys():
        st.inbox.setdefault(aid, [])
    st.last_private = st.last_private or {}
    st.conversation_partner = st.conversation_partner or {}

    ctx = get_context("spawn")
    stop_evt = ctx.Event()

    api_key = os.environ.get("OPENAI_API_KEY")

    cfg_dict = {
        "agents_n": cfg.agents_n,
        "models": cfg.models,
        "day": cfg.day,
        "night": cfg.night,
        "life": cfg.life,
        "reproduction": cfg.reproduction,
        "relationships": cfg.relationships,
        "limits": cfg.limits,
    }

    in_queues: dict[str, Queue] = {}
    out_queues: dict[str, Queue] = {}
    procs: dict[str, Process] = {}

    for aid, rec in st.agents.items():
        in_q: Queue = ctx.Queue()
        out_q: Queue = ctx.Queue()
        in_queues[aid] = in_q
        out_queues[aid] = out_q
        p = ctx.Process(
            target=_agent_worker,
            kwargs={
                "agent_id": aid,
                "gender": rec.gender,
                "cfg_dict": cfg_dict,
                "api_key": api_key,
                "in_q": in_q,
                "out_q": out_q,
                "stop_evt": stop_evt,
            },
            daemon=True,
        )
        procs[aid] = p
        p.start()

    narrator_client: ResponsesClient
    if api_key:
        narrator_client = OpenAIResponsesClient(api_key=api_key)
    else:
        class _NF:
            def responses_create(self, *, model: str, input: Any, **kwargs: Any) -> str:  # noqa: ARG002
                return json.dumps(
                    {
                        "ok": True,
                        "reason": "ok",
                        "public_message": "Mundane outcome.",
                        "private_message": "",
                        "actor_energy_delta": 0.0,
                        "target_energy_delta": 0.0,
                    }
                )

        narrator_client = _NF()

    narrator_call_cfg = JSONCallConfig(model=cfg.models.narrator, max_retries=2, temperature=0.0)

    last_request_at: dict[str, float] = {aid: 0.0 for aid in st.agents.keys()}
    inflight: set[str] = set()

    def persist() -> None:
        st.rng_state_b64 = _b64_state(rng)
        save_json(paths.state_json, state_to_jsonable(st))
        _save_relationship(paths, st)

    try:
        for _ in range(days):
            st.phase = "day"
            day_end = time.monotonic() + float(st.day_remaining_s or realtime.day_seconds)
            warned = False
            last_status_emit = 0.0

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

            while True:
                now = time.monotonic()
                remaining = max(0.0, day_end - now)
                st.day_remaining_s = float(remaining)

                if on_progress and float(realtime.status_every_s) > 0:
                    if (now - last_status_emit) >= float(realtime.status_every_s):
                        last_status_emit = now
                        alive = sum(1 for a in st.agents.values() if a.alive)
                        on_progress(
                            {
                                "type": "heartbeat",
                                "phase": st.phase,
                                "day": st.day,
                                "day_remaining_s": st.day_remaining_s,
                                "alive": alive,
                                "inflight": len(inflight),
                                "births": int(day_stats.get("births", 0)),
                                "deaths": int(day_stats.get("deaths", 0)),
                                "coop": int(day_stats.get("coop", 0)),
                                "comp": int(day_stats.get("comp", 0)),
                            }
                        )

                if not warned and remaining <= float(realtime.night_warning_seconds):
                    warned = True
                    for aid in st.agents.keys():
                        if st.agents[aid].alive:
                            st.inbox.setdefault(aid, []).append(
                                "Narrator: Night is coming soon. Finish up conversations quickly and prepare to sleep."
                            )
                    persist()
                    if on_progress:
                        on_progress({"type": "night_warning", "day": st.day, "seconds_left": remaining})

                if remaining <= 0.0:
                    break

                # Schedule new requests if agent is idle.
                for aid, rec in st.agents.items():
                    if not rec.alive:
                        continue
                    if aid in inflight:
                        continue
                    if (now - last_request_at.get(aid, 0.0)) < float(realtime.agent_cooldown_s):
                        continue

                    mem_path, ins_path, _ = _ensure_agent_files(paths, agent_id=aid, max_history_chars=cfg.limits.max_instruction_history_chars)
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
                    allowed = _agent_allowed_ops(st=st, cfg=cfg, agent_id=aid)
                    obs = {
                        "phase": st.phase,
                        "day": st.day,
                        "real_time_day_remaining_s": remaining,
                        "you": {"agent_id": aid, "gender": rec.gender, "age_days": rec.age_days, "energy": rec.energy},
                        "pending_mate_requests_for_you": sorted(list(st.pending_mate_requests.get(aid, set()))),
                        "conversation_partner": st.conversation_partner.get(aid, ""),
                        "relationship_row": st.relationship.get(aid, {}),
                        "roster": roster,
                        "inbox": st.inbox.get(aid, []),
                        "your_addon_instructions": _tail_text(ins_path, max_chars=cfg.limits.max_instruction_chars),
                        "your_memory_tail": _tail_text(mem_path, max_chars=cfg.limits.max_memory_chars),
                        "your_last_private_note": st.last_private.get(aid, ""),
                        "allowed_ops_if_child": sorted(list(allowed)) if allowed is not None else None,
                    }
                    in_queues[aid].put({"type": "request_action", "observation_json": json.dumps(obs, ensure_ascii=False)})
                    inflight.add(aid)
                    last_request_at[aid] = now

                # Drain outputs.
                for aid, out_q in out_queues.items():
                    try:
                        msg = out_q.get_nowait()
                    except Exception:
                        continue
                    if msg.get("type") != "action":
                        continue
                    inflight.discard(aid)
                    act = validate_action(msg.get("action"))
                    act["actor_id"] = aid

                    rec = st.agents.get(aid)
                    if rec is None or not rec.alive:
                        continue

                    # Baseline energy drain per action.
                    rec.energy = clamp_energy(rec.energy - 1.0)

                    # Consume inbox on action.
                    st.inbox[aid] = []

                    allowed = _agent_allowed_ops(st=st, cfg=cfg, agent_id=aid)
                    if allowed is not None and act["op"] not in allowed:
                        act = AgentAction({"actor_id": aid, "op": "rest", "text": "Too young; resting."})

                    oc = day_stats.get("op_counts")
                    if isinstance(oc, dict):
                        op = str(act.get("op", "") or "")
                        oc[op] = int(oc.get(op, 0)) + 1

                    tgt = str(act.get("target_id", "") or "").strip()

                    # Conversation busy enforcement.
                    if act["op"] == "message":
                        if tgt and tgt in st.agents and st.agents[tgt].alive:
                            ap = st.conversation_partner.get(aid, "")
                            tp = st.conversation_partner.get(tgt, "")
                            if ap and ap != tgt:
                                act = AgentAction({"actor_id": aid, "op": "rest", "text": "Busy in another conversation."})
                                tgt = ""
                            elif tp and tp != aid:
                                act = AgentAction({"actor_id": aid, "op": "rest", "text": "Target is busy in another conversation."})
                                tgt = ""
                            else:
                                st.conversation_partner[aid] = tgt
                                st.conversation_partner[tgt] = aid
                        else:
                            tgt = ""

                    if act["op"] == "end_conversation":
                        p = st.conversation_partner.get(aid, "")
                        if p:
                            st.conversation_partner.pop(aid, None)
                            if st.conversation_partner.get(p) == aid:
                                st.conversation_partner.pop(p, None)

                    mem_path, ins_path, hist_path = _ensure_agent_files(paths, agent_id=aid, max_history_chars=cfg.limits.max_instruction_history_chars)

                    handled = False
                    if act["op"] == "write_memory":
                        txt = str(act.get("text", "") or "")
                        if txt.strip():
                            _append_md(mem_path, title=f"{_now_ts()} (day {st.day})", body=txt)
                            _truncate_to_max(mem_path, max_chars=cfg.limits.max_memory_chars)
                        handled = True
                        outcome = {"ok": True, "reason": "memory_written", "public_message": "", "private_message": "Memory updated."}
                    elif act["op"] == "edit_instructions":
                        new_txt = str(act.get("text", "") or "")
                        if rec.edits_today >= cfg.limits.max_edits_per_day:
                            handled = True
                            outcome = {"ok": False, "reason": "edit_limit", "public_message": "", "private_message": "Edit limit reached for today."}
                        elif len(new_txt) > cfg.limits.max_instruction_chars:
                            handled = True
                            outcome = {"ok": False, "reason": "too_long", "public_message": "", "private_message": "Add-on instructions too long."}
                        else:
                            _append_instruction_history(
                                hist_path=hist_path,
                                title=f"{_now_ts()} (day {st.day}) — edit",
                                instructions_text=new_txt,
                                max_chars=cfg.limits.max_instruction_history_chars,
                            )
                            ins_path.write_text(new_txt, encoding="utf-8")
                            rec.edits_today += 1
                            day_stats["edits"] = int(day_stats.get("edits", 0)) + 1
                            handled = True
                            outcome = {"ok": True, "reason": "instructions_updated", "public_message": "", "private_message": "Add-on instructions updated."}
                    elif act["op"] == "self_talk":
                        st.last_private[aid] = str(act.get("text", "") or "")[:1000]
                        handled = True
                        outcome = {"ok": True, "reason": "self_talk", "public_message": "", "private_message": "Noted."}
                    elif act["op"] == "message":
                        txt = str(act.get("text", "") or "").strip()
                        if tgt and txt and tgt in st.agents and st.agents[tgt].alive:
                            st.inbox.setdefault(tgt, []).append(f"From {aid}: {txt}")
                            handled = True
                            outcome = {"ok": True, "reason": "message_sent", "public_message": f"Message delivered to {tgt}.", "private_message": "", "interaction": {"type": "cooperative", "notes": "communication"}}
                        else:
                            handled = True
                            outcome = {"ok": False, "reason": "invalid_target", "public_message": "", "private_message": "Message failed."}
                    elif act["op"] == "mate_request":
                        if tgt and tgt in st.agents and st.agents[tgt].alive:
                            st.pending_mate_requests.setdefault(tgt, set()).add(aid)
                            st.inbox.setdefault(tgt, []).append(f"Mate request from {aid}. To consent, use op='mate_accept' target_id='{aid}'.")
                            day_stats["mate_requests"] = int(day_stats.get("mate_requests", 0)) + 1
                            handled = True
                            outcome = {"ok": True, "reason": "mate_requested", "public_message": f"Mate request sent to {tgt}.", "private_message": ""}
                        else:
                            handled = True
                            outcome = {"ok": False, "reason": "invalid_target", "public_message": "", "private_message": "Mate request failed."}
                    elif act["op"] == "mate_accept":
                        src = str(act.get("target_id", "") or "").strip()
                        ok = src in st.pending_mate_requests.get(aid, set())
                        if ok:
                            st.pending_mate_requests.get(aid, set()).discard(src)
                        handled = True
                        birth = None
                        if ok and cfg.reproduction.consent_required:
                            day_stats["mate_accepts"] = int(day_stats.get("mate_accepts", 0)) + 1
                            a = st.agents.get(aid)
                            b = st.agents.get(src)
                            if a and b and a.alive and b.alive and a.age_days >= cfg.life.reproduction_age_days and b.age_days >= cfg.life.reproduction_age_days and a.gender != b.gender:
                                mother_id = aid if a.gender == "female" else src
                                father_id = src if mother_id == aid else aid
                                pk = pair_key(mother_id=mother_id, father_id=father_id)
                                count = st.children_by_pair.get(pk, 0)
                                if count < cfg.reproduction.max_children_per_pair:
                                    st.children_by_pair[pk] = count + 1
                                    child_id = f"c{st.day:03d}{count+1:02d}"
                                    g: Gender = "male" if rng.random() < 0.5 else "female"
                                    child = add_child(st=st, mother_id=mother_id, father_id=father_id, child_id=child_id, gender=g, start_energy=max(40.0, cfg.life.start_energy * 0.6))
                                    _, mom_ins, _ = _ensure_agent_files(paths, agent_id=mother_id, max_history_chars=cfg.limits.max_instruction_history_chars)
                                    _, dad_ins, _ = _ensure_agent_files(paths, agent_id=father_id, max_history_chars=cfg.limits.max_instruction_history_chars)
                                    inherited = _mix_instructions(
                                        mother_text=_tail_text(mom_ins, max_chars=cfg.limits.max_instruction_chars),
                                        father_text=_tail_text(dad_ins, max_chars=cfg.limits.max_instruction_chars),
                                        rng=rng,
                                        mutation_rate=float(cfg.reproduction.inheritance.mutation_rate),
                                    )
                                    child_mem, child_ins, child_hist = _ensure_agent_files(paths, agent_id=child.agent_id, max_history_chars=cfg.limits.max_instruction_history_chars)
                                    child_ins.write_text(inherited, encoding="utf-8")
                                    child_mem.write_text("", encoding="utf-8")
                                    _append_instruction_history(
                                        hist_path=child_hist,
                                        title=f"{_now_ts()} (day {st.day}) — inherited",
                                        instructions_text=inherited,
                                        max_chars=cfg.limits.max_instruction_history_chars,
                                    )
                                    st.inbox.setdefault(child.agent_id, []).append("You are a child agent. Rely on parents; conserve energy.")
                                    birth = {"child_id": child.agent_id, "gender": child.gender, "mother_id": mother_id, "father_id": father_id}
                                    day_stats["births"] = int(day_stats.get("births", 0)) + 1

                        if birth:
                            outcome = {"ok": True, "reason": "mated", "public_message": f"Child born: {birth['child_id']}.", "private_message": "", "interaction": {"type": "cooperative", "notes": "reproduction"}, "created_child": birth}
                        else:
                            outcome = {"ok": bool(ok), "reason": "mate_accept" if ok else "no_request", "public_message": "", "private_message": "" if ok else "No pending mate request found."}

                    if not handled:
                        sys_n = narrator_system_prompt(cfg=cfg)
                        user_n = json.dumps(
                            {
                                "world": {"phase": st.phase, "day": st.day, "real_time_day_remaining_s": remaining},
                                "action": dict(act),
                            },
                            ensure_ascii=False,
                        )
                        out_obj = call_json_object(client=narrator_client, cfg=narrator_call_cfg, system=sys_n, user=user_n)
                        outcome = validate_outcome(out_obj)

                    # Apply deltas.
                    rec.energy = clamp_energy(rec.energy + float(outcome.get("actor_energy_delta", 0.0) or 0.0))
                    if tgt and tgt in st.agents and st.agents[tgt].alive:
                        st.agents[tgt].energy = clamp_energy(st.agents[tgt].energy + float(outcome.get("target_energy_delta", 0.0) or 0.0))

                    # Relationship updates.
                    if tgt and tgt in st.relationship.get(aid, {}):
                        inter = outcome.get("interaction")
                        if isinstance(inter, dict):
                            typ = str(inter.get("type", "neutral"))
                            if typ == "cooperative":
                                st.relationship[aid][tgt] += cfg.relationships.cooperative_delta
                                st.relationship[tgt][aid] += cfg.relationships.cooperative_delta
                                day_stats["coop"] = int(day_stats.get("coop", 0)) + 1
                            elif typ == "competitive":
                                st.relationship[aid][tgt] += cfg.relationships.competitive_delta
                                st.relationship[tgt][aid] += cfg.relationships.competitive_delta
                                day_stats["comp"] = int(day_stats.get("comp", 0)) + 1
                            else:
                                day_stats["neutral"] = int(day_stats.get("neutral", 0)) + 1

                    pub = str(outcome.get("public_message", "") or "").strip()
                    priv = str(outcome.get("private_message", "") or "").strip()
                    if pub:
                        st.inbox.setdefault(aid, []).append("Narrator: " + pub)
                        if tgt and tgt in st.agents and st.agents[tgt].alive and not bool(act.get("private", False)):
                            st.inbox.setdefault(tgt, []).append("Narrator: " + pub)
                    if priv:
                        st.inbox.setdefault(aid, []).append("Narrator (private): " + priv)

                    if rec.energy <= 0.0:
                        rec.alive = False
                        day_stats["deaths"] = int(day_stats.get("deaths", 0)) + 1
                        st.inbox.setdefault(aid, []).append("You died (energy depleted).")

                    _log_event(paths, {"ts_utc": _now_ts(), "day": st.day, "actor_id": aid, "action": dict(act), "outcome": dict(outcome)})
                    persist()

                time.sleep(0.05)

            # Night broadcast.
            st.phase = "night"
            if on_progress:
                on_progress({"type": "night_start", "day": st.day})
            for aid in st.agents.keys():
                if st.agents[aid].alive:
                    st.inbox.setdefault(aid, []).append("Narrator: Night has come. Stop actions and go to sleep now.")
            st.conversation_partner = {}
            persist()

            # Night routine per agent (one action).
            night_deadline = time.monotonic() + float(realtime.night_seconds)
            inflight.clear()
            for aid, rec in st.agents.items():
                if rec.alive:
                    mem_path, ins_path, _ = _ensure_agent_files(paths, agent_id=aid, max_history_chars=cfg.limits.max_instruction_history_chars)
                    obs = {
                        "phase": "night",
                        "day": st.day,
                        "you": {"agent_id": aid, "gender": rec.gender, "age_days": rec.age_days, "energy": rec.energy},
                        "your_addon_instructions": _tail_text(ins_path, max_chars=cfg.limits.max_instruction_chars),
                        "your_memory_tail": _tail_text(mem_path, max_chars=cfg.limits.max_memory_chars),
                    }
                    in_queues[aid].put({"type": "night_routine", "observation_json": json.dumps(obs, ensure_ascii=False)})
                    inflight.add(aid)

            while inflight and time.monotonic() < night_deadline:
                for aid, out_q in out_queues.items():
                    if aid not in inflight:
                        continue
                    try:
                        msg = out_q.get_nowait()
                    except Exception:
                        continue
                    if msg.get("type") != "night_action":
                        continue
                    inflight.discard(aid)
                    act = validate_action(msg.get("action"))
                    act["actor_id"] = aid
                    if act["op"] not in {"write_memory", "edit_instructions", "self_talk", "rest"}:
                        act = AgentAction({"actor_id": aid, "op": "write_memory", "text": "Night reflection: prepare for tomorrow."})

                    rec = st.agents.get(aid)
                    if rec is None or not rec.alive:
                        continue
                    mem_path, ins_path, hist_path = _ensure_agent_files(paths, agent_id=aid, max_history_chars=cfg.limits.max_instruction_history_chars)
                    if act["op"] == "write_memory":
                        txt = str(act.get("text", "") or "")
                        if txt.strip():
                            _append_md(mem_path, title=f"{_now_ts()} (night day {st.day})", body=txt)
                            _truncate_to_max(mem_path, max_chars=cfg.limits.max_memory_chars)
                    elif act["op"] == "edit_instructions":
                        new_txt = str(act.get("text", "") or "")
                        if len(new_txt) <= cfg.limits.max_instruction_chars:
                            _append_instruction_history(
                                hist_path=hist_path,
                                title=f"{_now_ts()} (night day {st.day}) — edit",
                                instructions_text=new_txt,
                                max_chars=cfg.limits.max_instruction_history_chars,
                            )
                            ins_path.write_text(new_txt, encoding="utf-8")
                    elif act["op"] == "self_talk":
                        st.last_private[aid] = str(act.get("text", "") or "")[:1000]
                    persist()

                time.sleep(0.05)

            # Context reset and day rollover.
            if cfg.night.context_reset:
                st.inbox = {aid: [] for aid in st.agents.keys()}
                st.last_private = {}
                st.conversation_partner = {}
            st.day_remaining_s = float(realtime.day_seconds)
            st.step_in_day = 1

            if telemetry is not None:
                try:
                    telemetry.log_day(cfg=cfg, st=st, day_stats=day_stats, day_index=int(st.day), world_root=paths.root)
                except Exception:
                    pass
            if on_progress:
                on_progress({"type": "day_end", "day": st.day, "stats": dict(day_stats)})

            advance_day(st)
            persist()

    finally:
        stop_evt.set()
        for q in in_queues.values():
            try:
                q.put({"type": "shutdown"})
            except Exception:
                pass
        for p in procs.values():
            try:
                p.join(timeout=2.0)
            except Exception:
                pass
