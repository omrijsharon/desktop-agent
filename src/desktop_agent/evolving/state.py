from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .schemas import Gender


@dataclass
class AgentRecord:
    agent_id: str
    gender: Gender
    age_days: int
    energy: float
    alive: bool = True
    parents: tuple[str, str] | None = None  # (mother_id, father_id)
    is_child: bool = False
    edits_today: int = 0


@dataclass
class WorldState:
    phase: str = "day"  # "day" | "night"
    day: int = 1
    step_in_day: int = 1
    day_remaining_s: float = 0.0  # real-time remaining seconds in the current day (pause-resumable)
    agents: dict[str, AgentRecord] = field(default_factory=dict)
    pending_mate_requests: dict[str, set[str]] = field(default_factory=dict)  # target_id -> {from_id}
    children_by_pair: dict[str, int] = field(default_factory=dict)  # "mother|father" -> count
    relationship: dict[str, dict[str, float]] = field(default_factory=dict)  # i -> j -> score
    family_tree: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Ephemeral-but-resumable state (persisted so restart continues seamlessly).
    inbox: dict[str, list[str]] = field(default_factory=dict)  # agent_id -> messages
    last_private: dict[str, str] = field(default_factory=dict)  # agent_id -> last self_talk
    conversation_partner: dict[str, str] = field(default_factory=dict)  # agent_id -> partner_id (if any)
    # Resume cursor for the current step.
    step_agent_order: list[str] = field(default_factory=list)
    step_next_index: int = 0
    # Parallel-step resume (apply precomputed actions).
    current_step_actions: list[dict[str, Any]] = field(default_factory=list)  # [{"actor_id":..., "action":...}, ...]
    current_step_apply_index: int = 0
    # RNG state for reproducible continuation.
    rng_state_b64: str = ""

    def alive_agents(self) -> list[AgentRecord]:
        return [a for a in self.agents.values() if a.alive]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_population(*, n: int, start_age_days: int, start_energy: float, rng: random.Random) -> WorldState:
    st = WorldState()
    for i in range(1, n + 1):
        agent_id = f"a{i:03d}"
        gender: Gender = "male" if rng.random() < 0.5 else "female"
        st.agents[agent_id] = AgentRecord(
            agent_id=agent_id,
            gender=gender,
            age_days=int(start_age_days),
            energy=float(start_energy),
            alive=True,
            parents=None,
            is_child=False,
        )
    _init_relationship_matrix(st)
    _init_family_tree(st)
    st.inbox = {aid: [] for aid in st.agents.keys()}
    st.last_private = {}
    st.conversation_partner = {}
    return st


def _init_relationship_matrix(st: WorldState) -> None:
    ids = list(st.agents.keys())
    st.relationship = {i: {j: 0.0 for j in ids if j != i} for i in ids}


def _init_family_tree(st: WorldState) -> None:
    st.family_tree = {}
    for a in st.agents.values():
        st.family_tree[a.agent_id] = {"parents": list(a.parents) if a.parents else [], "children": []}


def clamp_energy(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return float(max(0.0, min(200.0, x)))


def advance_day(st: WorldState) -> None:
    st.day += 1
    st.step_in_day = 1
    for a in st.agents.values():
        a.edits_today = 0
        if a.alive:
            a.age_days += 1


def advance_step(st: WorldState) -> None:
    st.step_in_day += 1


def pair_key(*, mother_id: str, father_id: str) -> str:
    return f"{mother_id}|{father_id}"


def add_child(
    *,
    st: WorldState,
    mother_id: str,
    father_id: str,
    child_id: str,
    gender: Gender,
    start_energy: float,
) -> AgentRecord:
    rec = AgentRecord(
        agent_id=child_id,
        gender=gender,
        age_days=0,
        energy=float(start_energy),
        alive=True,
        parents=(mother_id, father_id),
        is_child=True,
    )
    st.agents[child_id] = rec

    # Expand relationship matrix with new row/col.
    for i in list(st.relationship.keys()):
        st.relationship[i][child_id] = 0.0
    st.relationship[child_id] = {j: 0.0 for j in st.agents.keys() if j != child_id}

    # Family tree.
    st.family_tree.setdefault(mother_id, {"parents": [], "children": []})
    st.family_tree.setdefault(father_id, {"parents": [], "children": []})
    st.family_tree.setdefault(child_id, {"parents": [mother_id, father_id], "children": []})
    st.family_tree[mother_id]["children"].append(child_id)
    st.family_tree[father_id]["children"].append(child_id)

    return rec


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def state_to_jsonable(st: WorldState) -> dict[str, Any]:
    return {
        "phase": st.phase,
        "day": st.day,
        "step_in_day": st.step_in_day,
        "day_remaining_s": float(getattr(st, "day_remaining_s", 0.0) or 0.0),
        "agents": {k: asdict(v) for k, v in st.agents.items()},
        "pending_mate_requests": {k: sorted(list(v)) for k, v in st.pending_mate_requests.items()},
        "children_by_pair": dict(st.children_by_pair),
        "relationship": st.relationship,
        "family_tree": st.family_tree,
        "inbox": st.inbox,
        "last_private": st.last_private,
        "conversation_partner": st.conversation_partner,
        "step_agent_order": list(st.step_agent_order),
        "step_next_index": int(st.step_next_index),
        "current_step_actions": list(st.current_step_actions),
        "current_step_apply_index": int(st.current_step_apply_index),
        "rng_state_b64": st.rng_state_b64,
    }


def load_state(path: Path) -> Optional[WorldState]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    st = WorldState(
        phase=str(data.get("phase", "day") or "day"),
        day=int(data.get("day", 1)),
        step_in_day=int(data.get("step_in_day", 1)),
        day_remaining_s=float(data.get("day_remaining_s", 0.0) or 0.0),
    )
    agents_obj = data.get("agents", {})
    if isinstance(agents_obj, dict):
        for aid, rec in agents_obj.items():
            if not isinstance(rec, dict):
                continue
            gender = rec.get("gender", "male")
            if gender not in {"male", "female"}:
                gender = "male"
            parents = rec.get("parents")
            parents_t: tuple[str, str] | None = None
            if isinstance(parents, (list, tuple)) and len(parents) == 2 and all(isinstance(x, str) for x in parents):
                parents_t = (parents[0], parents[1])
            st.agents[aid] = AgentRecord(
                agent_id=str(rec.get("agent_id", aid)),
                gender=gender,  # type: ignore[arg-type]
                age_days=int(rec.get("age_days", 0)),
                energy=float(rec.get("energy", 0.0)),
                alive=bool(rec.get("alive", True)),
                parents=parents_t,
                is_child=bool(rec.get("is_child", False)),
                edits_today=int(rec.get("edits_today", 0)),
            )
    pmr = data.get("pending_mate_requests", {})
    if isinstance(pmr, dict):
        st.pending_mate_requests = {
            str(k): set(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else set() for k, v in pmr.items()
        }
    cbp = data.get("children_by_pair", {})
    if isinstance(cbp, dict):
        st.children_by_pair = {str(k): int(v) for k, v in cbp.items() if isinstance(v, int)}
    rel = data.get("relationship", {})
    if isinstance(rel, dict):
        st.relationship = rel  # best-effort
    ft = data.get("family_tree", {})
    if isinstance(ft, dict):
        st.family_tree = ft  # best-effort

    inbox = data.get("inbox", {})
    if isinstance(inbox, dict):
        st.inbox = {str(k): list(v) if isinstance(v, list) else [] for k, v in inbox.items()}
    lp = data.get("last_private", {})
    if isinstance(lp, dict):
        st.last_private = {str(k): str(v or "") for k, v in lp.items()}
    cp = data.get("conversation_partner", {})
    if isinstance(cp, dict):
        st.conversation_partner = {str(k): str(v or "") for k, v in cp.items() if str(v or "").strip()}

    sao = data.get("step_agent_order", [])
    if isinstance(sao, list) and all(isinstance(x, str) for x in sao):
        st.step_agent_order = list(sao)
    st.step_next_index = int(data.get("step_next_index", 0))

    csa = data.get("current_step_actions", [])
    if isinstance(csa, list):
        st.current_step_actions = [x for x in csa if isinstance(x, dict)]
    st.current_step_apply_index = int(data.get("current_step_apply_index", 0))

    r = data.get("rng_state_b64", "")
    if isinstance(r, str):
        st.rng_state_b64 = r
    return st
