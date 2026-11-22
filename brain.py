"""Mech Royale bot brain (v2).

Score = 16 - place. Practice house bots never fight each other, so in practice
the score is roughly "how many of them die before we do". That means:
  * we must WIN fights (kills), not just hide,
  * every fight should be taken only with an edge (HP x DPS), otherwise break
    line of sight behind a building and regenerate (45 HP/s after 5 s),
  * snowball early: weapons (laser, lightning) and cubes (+4% HP and damage each).

Layers:
  1. world model     map raster, line of sight, nav grid + resumable A*
  2. memory          enemy tracks, known loot / crates / NPCs, damage history
  3. evaluation      expected DPS both ways -> time-to-kill advantage
  4. intent          ZONE / FIGHT / RETREAT / LOOT / DROP / EXPLORE (utility based)
  5. motor           path following, strafing, dodging, unsticking
  6. gunnery         lead aim, per-weapon fire control, laser charge timing
"""
import math
import os
import random
import time

from mechio import log
from model import CombatModel

TAU = 2.0 * math.pi
# aggression knobs
RET_HP = float(os.environ.get("RET_HP", "0.12"))
RET_ADV = float(os.environ.get("RET_ADV", "0.3"))
REGEN_HP = float(os.environ.get("REGEN_HP", "0.3"))
FFA_THRESHOLD = 12.0
FFA_STYLE = os.environ.get("FFA_STYLE", "survivor")   # opportunist | survivor | aggressor
# free-for-all knobs (placement matters, everyone shoots)
FFA_RET_HP = 0.4
FFA_RET_ADV = 0.9
FFA_REGEN_HP = 0.75
# weapon values: gun + laser beats anything with lightning (lightning pulls us into close fights)
WVAL = {"laser": 1.7, "lightning": float(os.environ.get("W_LIGHT", "0.9")), "gun": float(os.environ.get("W_GUN", "1.6")),
        "mortar": float(os.environ.get("W_MORTAR", "0.55"))}
CUBE_VAL = float(os.environ.get("CUBE_VAL", "13"))
HOSTILE_BONUS = float(os.environ.get("HOSTILE_BONUS", "5"))
KITE = os.environ.get("KITE", "0") == "1"
FIGHT_BOOST = os.environ.get("FIGHT_BOOST", "0") == "1"
STRAFE_LO = float(os.environ.get("STRAFE_LO", "0.35"))
STRAFE_HI = float(os.environ.get("STRAFE_HI", "0.95"))
RANGE_MIN = float(os.environ.get("RANGE_MIN", "0"))
# experiment knobs (defaults = submitted behaviour)
HUNT_WHEN_HOSTILE = os.environ.get("HUNT_WHEN_HOSTILE", "1") == "1"
TRAVEL_BOOST = os.environ.get("TRAVEL_BOOST", "0") == "1"
HUNT_HI = float(os.environ.get("HUNT_HI", "30"))
UPG_STYLE = os.environ.get("UPG_STYLE", "ehp")
FLANK = os.environ.get("FLANK", "1") == "1"
NPC_FIRST = os.environ.get("NPC_FIRST", "0") == "1"
FARM_EARLY = float(os.environ.get("FARM_EARLY", "0"))
ISO = os.environ.get("ISO", "1") == "1"
FIGHT_RANGE_SCALE = float(os.environ.get("FIGHT_RANGE_SCALE", "1.0"))
TEMPO = int(os.environ.get("TEMPO", "0"))
PEEK = int(os.environ.get("PEEK", "0"))
BLITZ = int(os.environ.get("BLITZ", "0"))
PF_W = float(os.environ.get("PF_W", "1.0"))
LOOT_PLUS = int(os.environ.get("LOOT_PLUS", "0"))
CANOPY = os.environ.get("CANOPY", "0") == "1"   # tree crowns block sight (vision config canopy_blocks)
LIGHT_FIRST = int(os.environ.get("LIGHT_FIRST", "0"))
AGGRO = int(os.environ.get("AGGRO", "0"))
AGGRO_QUIET = float(os.environ.get("AGGRO_QUIET", "6"))
RET2 = int(os.environ.get("RET2", "0"))
REGEN_MULT = float(os.environ.get("REGEN_MULT", "1.0"))
UPG_DMG_W = float(os.environ.get("UPG_DMG_W", "1.0"))
PREY_AWAY = float(os.environ.get("PREY_AWAY", "0"))
HUNT_D_MIN = float(os.environ.get("HUNT_D_MIN", "0"))
CUBE_URG = float(os.environ.get("CUBE_URG", "90"))
PEEK_R = float(os.environ.get("PEEK_R", "14"))
TEMPO_HP = float(os.environ.get("TEMPO_HP", "0.85"))
TEMPO_QUIET = float(os.environ.get("TEMPO_QUIET", "8"))
TEMPO_UNLOCK = float(os.environ.get("TEMPO_UNLOCK", "9999"))


def angn(a):
    return (a + math.pi) % TAU - math.pi


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(v):
    m = math.hypot(v[0], v[1])
    if m < 1e-9:
        return (0.0, 0.0)
    return (v[0] / m, v[1] / m)


def cget(d, path, default):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur


class Track:
    __slots__ = ("id", "kind", "pos", "vel", "t_seen", "t_first", "hp", "max_hp", "cubes",
                 "weapons", "aim", "charging", "charge_since", "dmg_to_us", "last_hit_us",
                 "hostile", "approach", "estimated", "dist_hist", "vhist")

    def __init__(self, m, t):
        self.id = m["id"]
        self.kind = m.get("kind", "mech")
        self.t_first = t
        self.charging = False
        self.charge_since = None
        self.dmg_to_us = 0.0
        self.last_hit_us = -99.0
        self.hostile = False
        self.approach = 0.0
        self.estimated = False
        self.dist_hist = []
        self.vhist = []
        self.update(m, t)

    def update(self, m, t):
        self.estimated = False
        self.pos = list(m.get("pos") or [0, 0, 0])
        self.vel = list(m.get("vel") or [0, 0, 0])
        vh = self.vhist
        vh.append((t, self.vel[0], self.vel[1]))
        while vh and t - vh[0][0] > 1.2:
            vh.pop(0)
        self.t_seen = t
        self.hp = m.get("hp", 1400) if m.get("hp") is not None else 1400
        self.max_hp = m.get("max_hp", 1400) or 1400
        self.cubes = m.get("cubes", 0) or 0
        self.weapons = list(m.get("weapons") or ["gun"])
        self.aim = m.get("aim") or [0, 0]
        ch = bool(m.get("charging"))
        if ch and not self.charging:
            self.charge_since = t
        if not ch:
            self.charge_since = None
        self.charging = ch

    def avg_vel(self, window):
        """Mean horizontal velocity over the last `window` seconds (zig-zag averages out)."""
        vh = self.vhist
        if not vh:
            return self.vel[0], self.vel[1]
        t1 = vh[-1][0]
        sx = sy = 0.0
        n = 0
        for (tt, vx, vy) in vh:
            if t1 - tt <= window:
                sx += vx
                sy += vy
                n += 1
        return (sx / n, sy / n) if n else (self.vel[0], self.vel[1])

    def pred(self, t, horizon=2.0):
        dt = min(max(0.0, t - self.t_seen), horizon)
        return [self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt, self.pos[2]]


class Brain:
    def __init__(self, world_factory):
        self.world_factory = world_factory
        self.world = None
        self.cfg = {}
        self.rng = random.Random(1)
        self.tracks = {}
        self.npcs = {}
        self.crates = {}
        self.pickups = {}
        self.cubes_seen = {}
        self.cube_bad = set()
        self.npc_attackers = {}
        self.cube_near = {}
        # practice squad vs free-for-all detection
        self.mode = "practice"
        self.ffa_evidence = 0.0
        self.ffa_log = []
        self.ffa_kinds = {}
        self.ffa_seen_proj = set()
        self.ffa_seen_beam = set()
        self.hp_seen = {}
        self.hit_by_us = {}
        self.n_mechs = 16
        self.dead_ids = set()
        self.last_offer = None
        self.my_id = None
        self.t = 0.0
        self.last_combat_t = -99.0
        self.last_damaged_t = -99.0
        self.dmg_events = []
        self.sound_marks = []
        self.intent = "EXPLORE"
        self.intent_since = 0.0
        self.retreating = False
        self.cover = None
        self.cover_t = -99.0
        self.fight_cover = None
        self.fight_cover_t = -99.0
        self.hunt_boost = False
        self.kite_boost = False
        self.iso_spot = None
        self.iso_t = -99.0
        self.iso_for = None
        self.canopy = False
        self.pk_cache = (None, None)
        self.pk_t = -99.0
        self.pk_for = None
        self.pk_state = "expose"
        self.range_d = 90.0
        self.range_t = -99.0
        self.range_for = None
        self.strafe_sign = 1
        self.strafe_until = 0.0
        self.path_goal = None
        self.path = None
        self.path_i = 0
        self.path_t = -99.0
        self.astar = None
        self.pos_hist = []
        self.unstick_until = -1.0
        self.unstick_dir = (0.0, 0.0)
        self.unstick_n = 0
        self.unstick_last = -99.0
        self.scan_phase = 0.0
        self.last_target_id = None
        self.explore_goal = None
        self.explore_t = -99.0
        self.kills = 0
        self.model = CombatModel({})
        self.prev_t = None
        self.hp_lost = 0.0
        self.last_kill_hostile = {}
        self.prev_ammo = None
        self.prev_charge = 0.0
        self.fire_tgt = None      # (is_mech, distance, hostile) of what we aimed at last tick
        self.deadline = 1e18
        self.loot_kind = None
        self.adv = 99.0
        self.debug_label = ""
        self.tick_ms = []

    # ================================================================ hello
    def on_hello(self, msg):
        self.cfg = msg.get("config") or {}
        self.model = CombatModel(self.cfg)
        self.n_mechs = cget(self.cfg, "match.mechs", 16)
        self.rng = random.Random(msg.get("seed", 1))
        self.slot = msg.get("slot", 0)
        mp = msg.get("map") or {}
        mdir = mp.get("dir") or "/arena"
        try:
            self.world = self.world_factory(mdir, self.cfg)
            from world import AStar
            self.astar = AStar(self.world)
            log("world ok in %.2fs" % self.world.build_s)
        except Exception as e:
            log("world load failed:", repr(e))
            self.world = None
        self.canopy = bool(cget(self.cfg, "vision.canopy_blocks", False)) and hasattr(self.world, "vis") \
            and getattr(self.world, "kb", None) is not None
        return "Hunter-v2"

    def on_end(self, msg):
        log("end", msg, "kills", self.kills)

    # ================================================================ helpers
    def c(self, path, default):
        return cget(self.cfg, path, default)

    def los(self, a, b):
        if self.world is None:
            return True
        return self.world.los(a[0], a[1], a[2], b[0], b[1], b[2])

    def vis(self, a, b):
        """Can an eye at a see point b (terrain, buildings and, in the real game, tree crowns)."""
        if self.world is None:
            return True
        if CANOPY and self.canopy:
            return self.world.vis(a[0], a[1], a[2], b[0], b[1], b[2])
        return self.world.los(a[0], a[1], a[2], b[0], b[1], b[2])

    def eye(self, p, scale=1.0):
        return (p[0], p[1], p[2] + 8.0 * scale)

    # ================================================================ main
    def on_obs(self, obs):
        t0 = time.perf_counter()
        self.deadline = t0 + 0.020
        act = self._decide(obs)
        dt = (time.perf_counter() - t0) * 1000
        if dt > 30:
            log("slow tick %.1f ms" % dt)
        return act

    def _decide(self, obs):
        t = obs.get("t", 0.0)
        self.t = t
        me = obs.get("self") or {}
        self.me = me
        self.my_id = me.get("id", self.my_id)
        pos = me.get("pos") or [0, 0, 0]
        self.pos = pos
        self.vel = me.get("vel") or [0, 0, 0]
        self.hp = me.get("hp", 1400)
        self.max_hp = me.get("max_hp", 1400) or 1400
        self.energy = me.get("energy", 100)
        self.max_energy = me.get("max_energy", 100) or 100
        self.stats = me.get("stats") or {}
        self.weapons = me.get("weapons") or [None, None]
        self.my_cubes = me.get("cubes", 0) or 0
        self.scale = 1.0 + self.c("cubes.size_per_cube", 0.05) * self.my_cubes
        self.zone = obs.get("zone") or {}
        self.alive = obs.get("alive", 16)
        self.obs = obs
        act = {}

        self._update_memory(obs)
        self._update_mode(obs)
        self._learn(obs)

        # ---------------- upgrades
        offer = obs.get("offer")
        if offer and offer.get("id") != self.last_offer:
            act["pick"] = self._choose_upgrade(offer)
            self.last_offer = offer.get("id")

        # ---------------- situation
        threats = self._threats()
        self.threats = threats
        target = self._pick_target(threats)
        incoming = sum(e["dps"] for e in threats)
        self.incoming = incoming

        # ---------------- intent
        goal, style = self._choose_intent(threats, target)

        # ---------------- motor
        mv, boost, jump = self._motor(goal, style, threats, target)
        act["move"] = [mv[0], mv[1]]
        if boost:
            act["boost"] = True
        if jump:
            act["jump"] = True

        # ---------------- loot actions
        tk = self._take_action()
        if tk is not None:
            act["take"] = tk[0]
            if tk[1] is not None:
                act["drop"] = tk[1]

        # ---------------- gunnery
        aim, fire, reload = self._gunnery(target, threats)
        act["aim"] = [aim[0], aim[1]]
        act["fire"] = fire
        if reload:
            act["reload"] = True
        return act

    # ================================================================ memory
    def _update_memory(self, obs):
        t = self.t
        seen_ids = set()
        for m in obs.get("mechs") or []:
            mid = m.get("id")
            if mid is None or mid == self.my_id:
                continue
            if m.get("kind") == "npc":
                self.npcs[mid] = (list(m["pos"]), list(m.get("vel") or [0, 0, 0]), m.get("hp", 350), t)
                continue
            seen_ids.add(mid)
            tr = self.tracks.get(mid)
            if tr is None:
                tr = self.tracks[mid] = Track(m, t)
            else:
                tr.update(m, t)
            dd = d2(tr.pos, self.pos)
            tr.dist_hist.append((t, dd))
            while tr.dist_hist and t - tr.dist_hist[0][0] > 4.0:
                tr.dist_hist.pop(0)
            if len(tr.dist_hist) >= 8:
                t0, d0 = tr.dist_hist[0]
                rate = (d0 - dd) / max(t - t0, 0.1)
                tr.approach = rate
                hv = math.hypot(tr.vel[0], tr.vel[1])
                head_ok = False
                if hv > 3.0:
                    bto = math.atan2(self.pos[1] - tr.pos[1], self.pos[0] - tr.pos[0])
                    head_ok = abs(angn(math.atan2(tr.vel[1], tr.vel[0]) - bto)) < 0.45
                aim_ok = False
                if tr.aim:
                    bto2 = math.atan2(self.pos[1] - tr.pos[1], self.pos[0] - tr.pos[0])
                    aim_ok = abs(angn(tr.aim[0] - bto2)) < 0.25
                if rate > 4.5 and dd < 220 and t - t0 > 3.0 and head_ok and aim_ok:
                    tr.hostile = True
            if tr.charging and dd < 290:
                b = math.atan2(self.pos[1] - tr.pos[1], self.pos[0] - tr.pos[0])
                if abs(angn(b - (tr.aim[0] if tr.aim else 0))) < 0.15:
                    tr.hostile = True
        for ev in obs.get("events") or []:
            ty = ev.get("type")
            if ty == "damaged":
                self.last_damaged_t = t
                self.last_combat_t = t
                self.hp_lost += ev.get("amount", 0.0)
                if ev.get("weapon") == "npc" and ev.get("by") is not None:
                    self.npc_attackers[ev["by"]] = t
                self.dmg_events.append((t, ev.get("amount", 0), ev.get("by"), ev.get("bearing"), ev.get("weapon")))
                by = ev.get("by")
                if by is not None and by in self.tracks:
                    tr = self.tracks[by]
                    tr.dmg_to_us += ev.get("amount", 0)
                    tr.last_hit_us = t
                    tr.hostile = True
                    if t - tr.t_seen > 0.3 and ev.get("bearing") is not None:
                        # unseen shooter: move its estimate onto the bearing line
                        b = ev["bearing"]
                        dist = max(20.0, min(200.0, d2(tr.pos, self.pos)))
                        tr.pos = [self.pos[0] + math.cos(b) * dist, self.pos[1] + math.sin(b) * dist, self.pos[2]]
                        tr.vel = [0.0, 0.0, 0.0]
                        tr.t_seen = t - 0.5
                        tr.estimated = True
                elif by is not None and by < 100 and ev.get("bearing") is not None:
                    # attacker unseen: make a ghost track along the bearing
                    b = ev["bearing"]
                    dist = 120.0 if ev.get("weapon") in ("laser", "gun") else 40.0
                    gp = [self.pos[0] + math.cos(b) * dist, self.pos[1] + math.sin(b) * dist, self.pos[2]]
                    tr = Track({"id": by, "kind": "mech", "pos": gp, "vel": [0, 0, 0],
                                "weapons": [ev.get("weapon") or "gun"]}, t - 1.0)
                    tr.dmg_to_us = ev.get("amount", 0)
                    tr.last_hit_us = t
                    tr.hostile = True
                    tr.estimated = True
                    tr.t_seen = t - 0.5
                    self.tracks[by] = tr
            elif ty == "hit":
                self.last_combat_t = t
                tg = ev.get("target")
                if tg is not None:
                    self.hit_by_us[tg] = t
            elif ty == "kill":
                self.kills += 1
                tid = ev.get("target")
                if tid in self.tracks:
                    self.last_kill_hostile[tid] = self.tracks[tid].hostile
                self.dead_ids.add(tid)
                self.tracks.pop(tid, None)
            elif ty == "took":
                self.pickups.pop(ev.get("id"), None)
        self.dmg_events = [e for e in self.dmg_events if t - e[0] < 6.0]
        # sounds -> rough marks (only used for looking around)
        for s in obs.get("sounds") or []:
            if s.get("kind") in ("death",):
                continue
            self.sound_marks.append((t, s.get("bearing", 0.0), s.get("dist", "far"), s.get("kind")))
        self.sound_marks = [s for s in self.sound_marks if t - s[0] < 3.0]
        # forget stale tracks
        for mid in list(self.tracks):
            tr = self.tracks[mid]
            if t - tr.t_seen > 25.0 or mid in self.dead_ids:
                del self.tracks[mid]
        # loot memory
        vis_crates = set()
        for cr in obs.get("crates") or []:
            self.crates[cr["id"]] = (list(cr["pos"]), cr.get("hp", 300), t)
            vis_crates.add(cr["id"])
        vis_pick = set()
        for p in obs.get("pickups") or []:
            self.pickups[p["id"]] = (p.get("kind"), list(p["pos"]), t)
            vis_pick.add(p["id"])
        vis_cubes = set()
        for cu in obs.get("cubes") or []:
            if cu["id"] in self.cube_bad:
                continue
            self.cubes_seen[cu["id"]] = (list(cu["pos"]), t)
            vis_cubes.add(cu["id"])
            # standing right next to it for a while without getting it: unreachable (roof, air)
            if d2(cu["pos"], self.pos) < 5.0:
                n = self.cube_near.get(cu["id"], 0) + 1
                self.cube_near[cu["id"]] = n
                if n > 25:
                    self.cube_bad.add(cu["id"])
                    self.cubes_seen.pop(cu["id"], None)
        # remove remembered things that should be visible now but are not
        for store, vis in ((self.crates, vis_crates), (self.pickups, vis_pick), (self.cubes_seen, vis_cubes)):
            for k in list(store):
                if k in vis:
                    continue
                p = store[k][1] if store is self.pickups else store[k][0]
                if d2(p, self.pos) < 35.0 or t - store[k][-1] > 90.0:
                    del store[k]
        vis_npc = {m["id"] for m in obs.get("mechs") or [] if m.get("kind") == "npc"}
        for k in list(self.npcs):
            if k not in vis_npc:
                p = self.npcs[k][0]
                if d2(p, self.pos) < 35.0 or t - self.npcs[k][3] > 60.0:
                    del self.npcs[k]

    # ================================================================ learning
    def _learn(self, obs):
        """Feed the combat model with what happened since the last observation."""
        t = self.t
        dt = 0.1 if self.prev_t is None else max(0.0, min(0.5, t - self.prev_t))
        self.prev_t = t
        m = self.model
        ft = self.fire_tgt
        # our gun shots = ammo spent (only shots at mechs teach accuracy)
        gun = self._weapon("gun")
        if gun is not None:
            ammo = gun.get("ammo")
            if ammo is not None and self.prev_ammo is not None and ammo < self.prev_ammo and ft and ft[0]:
                m.on_shots("gun", ft[1], self.prev_ammo - ammo, t)
            self.prev_ammo = ammo
        else:
            self.prev_ammo = None
        laser = self._weapon("laser")
        ch = (laser.get("charge") or 0.0) if laser else 0.0
        if laser is not None and self.prev_charge > 0 and ch <= 0 and (laser.get("cooldown") or 0) > 0.3 and ft and ft[0]:
            m.on_shots("laser", ft[1], 1, t)
        self.prev_charge = ch
        if ft and ft[0]:
            m.on_fight_time("duel" if ft[2] else "hunt", ft[1], dt)
        for ev in obs.get("events") or []:
            ty = ev.get("type")
            if ty == "hit" and ev.get("target") is not None and ev.get("target") < 100:
                m.on_hit(ev.get("weapon"), t)
            elif ty == "damaged" and ev.get("by") is not None and ev.get("by") < 100:
                tr = self.tracks.get(ev["by"])
                if tr is not None:
                    d = d2(tr.pos, self.pos)
                    mult = 1.0 + self.c("cubes.damage_per_cube", 0.04) * min(tr.cubes, self.c("cubes.stat_cap", 10))
                    if ev.get("rear"):
                        mult *= self.c("mech.rear_mult", 1.5)
                    m.on_damage(ev.get("weapon"), d, ev.get("amount", 0.0), mult)
        # how many enemies shoot at us at once while we duel, and stray fire while we hunt
        shooters = [e for e in (getattr(self, "threats", []) or []) if e["los"] and e["age"] < 0.6 and e["tr"].hostile]
        if ft and ft[0] and ft[2]:
            m.on_shooters(len(shooters), dt)
        if self.intent == "HUNT":
            dmg_now = sum(ev.get("amount", 0.0) for ev in (obs.get("events") or []) if ev.get("type") == "damaged")
            m.on_hunt_exposure(dt, dmg_now)
        for ev in obs.get("events") or []:
            if ev.get("type") == "kill":
                m.on_kill(not self.last_kill_hostile.get(ev.get("target"), False))
        # exposure: enemies that can shoot us right now
        for e in getattr(self, "threats", []) or []:
            if e["age"] > 0.6 or not e["los"]:
                continue
            tr = e["tr"]
            if not (tr.hostile or (self.mode == "ffa" and self._aiming_at_us(tr))):
                continue
            m.on_exposure(tr.weapons, e["d"], dt)

    def _aiming_at_us(self, tr):
        if not tr.aim:
            return False
        bto = math.atan2(self.pos[1] - tr.pos[1], self.pos[0] - tr.pos[0])
        return abs(angn(tr.aim[0] - bto)) < 0.35

    # ================================================================ mode
    def _update_mode(self, obs):
        """Practice squad or free-for-all?

        In practice the house bots never fight each other: a death we did not cause almost
        never happens (measured: 0.04 per match with a bot that never fires). In a
        free-for-all they happen within seconds. Two such deaths decide it.
        """
        if self.mode == "ffa":
            return
        dead_others = (self.n_mechs - self.alive) - self.kills
        self.ffa_evidence = float(dead_others)
        if dead_others >= 2:
            self.mode = "ffa"
            log("mode -> ffa at t=%.1f" % self.t)

    # ================================================================ evaluation
    def _wstats(self, kind):
        for w in self.weapons:
            if w and w.get("kind") == kind:
                return w.get("stats") or {}
        return cget(self.cfg, "weapons." + kind, {})

    def _held(self):
        return [w.get("kind") for w in self.weapons if w]

    def dps_kind(self, kind, d, stats, energy=100.0, mover=8.0, us=True):
        """Expected damage per second for one weapon at distance d."""
        if kind == "gun":
            rng = stats.get("range", 230)
            if d > rng:
                return 0.0
            mag = stats.get("mag", 12)
            cd = stats.get("cooldown", 0.24)
            rl = stats.get("reload", 2.4)
            sustain = (mag * cd) / (mag * cd + rl)
            if us:
                # our lead aim against strafing targets
                hit = 0.9 if d <= 100 else max(0.45, 0.9 - (d - 100) * 0.0035)
            else:
                # their bullets against our random strafing: falls off fast with flight time
                hit = 0.85 * math.exp(-d / 60.0)
            return stats.get("damage", 45) / cd * sustain * hit
        if kind == "laser":
            if d > stats.get("range", 280):
                return 0.0
            return stats.get("damage", 315) / (stats.get("charge", 0.75) + stats.get("cooldown", 1.65)) * 0.85
        if kind == "lightning":
            if d > stats.get("range", 45) - 1.0:
                return 0.0
            if energy < 8:
                return 0.0
            return stats.get("damage", 60) / stats.get("cooldown", 0.25) * 0.9
        if kind == "mortar":
            if d < 25 or d > 230:
                return 0.0
            return (stats.get("damage", 200) * 0.25 + stats.get("splash", 160) * 0.35) / stats.get("cooldown", 2.6)
        return 0.0

    def our_mult(self):
        cap = self.c("cubes.stat_cap", 10)
        return 1.0 + self.c("cubes.damage_per_cube", 0.04) * min(self.my_cubes, cap)

    def our_dps(self, d, tgt_speed=8.0, los=True):
        if not los:
            return 0.0
        return self.model.our_dps(self.weapons, d, self.our_mult(), self.energy)

    def enemy_dps(self, tr, d):
        cap = self.c("cubes.stat_cap", 10)
        mult = 1.0 + self.c("cubes.damage_per_cube", 0.04) * min(tr.cubes, cap)
        armor = self.stats.get("armor", 0.0) or 0.0
        return self.model.enemy_dps(tr.weapons, d, mult, armor)

    def _threats(self):
        out = []
        my_eye = self.eye(self.pos, self.scale)
        my_c = (self.pos[0], self.pos[1], self.pos[2] + 4.5 * self.scale)
        budget = 10
        for tr in self.tracks.values():
            age = self.t - tr.t_seen
            p = tr.pred(self.t)
            d = d2(p, self.pos)
            if d > 330:
                continue
            sc = 1.0 + self.c("cubes.size_per_cube", 0.05) * tr.cubes
            their_eye = (p[0], p[1], p[2] + 8.0 * sc)
            if age < 0.15 and budget > 0:
                budget -= 1
                los = self.vis(their_eye, my_c) or self.vis(their_eye, (my_c[0], my_c[1], my_c[2] + 3.5 * self.scale))
            else:
                los = age < 6.0 and d < 60
            dps = self.enemy_dps(tr, d) if (los or d < 50) else self.enemy_dps(tr, d) * 0.25
            if age > 6.0:
                dps *= 0.3
            if not tr.hostile:
                dps *= 0.8 if self.mode == "ffa" else 0.35
            out.append({"tr": tr, "pos": p, "d": d, "los": los, "dps": dps, "age": age,
                        "hp": tr.hp, "scale": sc, "hostile": tr.hostile})
        out.sort(key=lambda e: e["d"])
        return out

    def _distracted(self, tr):
        if not tr.aim:
            return False
        bto = math.atan2(self.pos[1] - tr.pos[1], self.pos[0] - tr.pos[0])
        looking_away = abs(angn(tr.aim[0] - bto)) > 0.6
        hp_prev = self.hp_seen.get(tr.id)
        losing = tr.hp < tr.max_hp * 0.9
        return looking_away and (losing or tr.charging)

    def _prey_ok(self):
        """May we start on a passive (non-hunting) mech right now?"""
        if self.mode == "ffa" or TEMPO == 0 or self.t >= TEMPO_UNLOCK:
            return True
        if TEMPO == 1:
            return False
        if self.hp < TEMPO_HP * self.max_hp:
            return False
        for tr in self.tracks.values():
            if tr.hostile and self.t - tr.t_seen < TEMPO_QUIET:
                return False
        return True

    def _pick_target(self, threats):
        """Choose the target with the best net kill rate.

        value  = 1 kill
               + kills saved by silencing it (its dps over the rest of the fight, in kills)
               - kills lost to the damage we take while killing it
        rate   = value / time to kill (including the walk when it is out of sight)
        """
        best = None
        my_m = (self.pos[0], self.pos[1], self.pos[2] + self.stats.get("muzzle_height", 7.0))
        kappa = self._kills_per_hp()
        hostiles = [e for e in threats if e["hostile"] and e["age"] < 3.0 and (e["los"] or e["d"] < 60)]
        inc = sum(e["dps"] for e in hostiles)
        walk = self.stats.get("walk_speed", 11.7)
        prey_ok = self._prey_ok()
        cur = self.last_target_id
        for e in threats:
            if e["age"] > 1.0:
                continue
            tr = e["tr"]
            if not tr.hostile and not prey_ok and tr.id != cur:
                continue
            d = e["d"]
            c = (e["pos"][0], e["pos"][1], e["pos"][2] + 4.5 * e["scale"])
            los = self.los(my_m, c) if self.world else True
            e["myl"] = los
            spd = math.hypot(tr.vel[0], tr.vel[1])
            mydps = self.our_dps(d, spd, True)
            if mydps <= 0 and not los:
                continue
            ttk = tr.hp / max(mydps, 1.0)
            t_reach = 0.0 if los else min(d, 80.0) / walk
            if not los and mydps <= 0:
                t_reach = d / walk
            t_tot = ttk + t_reach
            # damage we eat meanwhile (everyone already shooting at us)
            cost = inc * t_tot * kappa
            # silencing a shooter: its dps no longer hits us during the rest of the fight
            others_ttk = sum(o["tr"].hp / max(self.our_dps(o["d"], 8.0, True), 1.0) for o in hostiles if o is not e)
            relief = e["dps"] * others_ttk * kappa if tr.hostile else 0.0
            value = 1.0 + relief - cost
            # in a free-for-all a kill is only worth it when we come out ahead; weak/busy targets are cheaper
            if self.mode == "ffa" and not tr.hostile and not (tr.hp < 0.5 * tr.max_hp or self._distracted(tr)):
                value -= 0.6
            score = value / max(t_tot, 0.5)
            if PREY_AWAY > 0 and not tr.hostile and self.mode != "ffa":
                # a passive next to the hunters drags us into their crossfire
                hn = [o for o in threats if o["hostile"] and o["age"] < 12.0]
                if hn:
                    near_h = min(d2(o["pos"], e["pos"]) for o in hn)
                    if near_h < 140.0:
                        score *= max(0.2, 1.0 - PREY_AWAY * (140.0 - near_h) / 140.0)
            if tr.hostile and self.mode != "ffa" and PF_W != 1.0:
                # practice squad releases hunters weakest first: a dead hunter is replaced by a stronger one,
                # a dead passive removes a strong mech from the pool
                score *= PF_W
            if tr.id == self.last_target_id:
                score *= 1.15   # switching costs aim time
            if best is None or score > best[0]:
                best = (score, e)
        if best is None:
            self.last_target_id = None
            return None
        self.last_target_id = best[1]["tr"].id
        return best[1]

    # ================================================================ zone
    def zone_circle(self, t):
        z = self.zone
        if not z:
            return (0.0, 0.0), 1e9
        cc = z.get("center", [0, 0])
        cr = z.get("radius", 1e4)
        nc = z.get("next_center", cc)
        nr = z.get("next_radius", cr)
        s0 = z.get("shrink_start", 1e9)
        s1 = z.get("shrink_end", 1e9)
        if t <= s0 or s1 <= s0:
            return (cc[0], cc[1]), cr
        f = min(1.0, (t - s0) / (s1 - s0))
        return (cc[0] + (nc[0] - cc[0]) * f, cc[1] + (nc[1] - cc[1]) * f), cr + (nr - cr) * f

    def zone_ok(self, p, eta=0.0, margin=6.0):
        """Is point p safely inside the zone when we would get there (and a bit after)?"""
        t = self.t + eta
        for dt in (0.0, 8.0):
            c, r = self.zone_circle(t + dt)
            if d2(p, c) > r - margin:
                return False
        z = self.zone
        if z and self.t > z.get("shrink_start", 1e9) - 25:
            nc = z.get("next_center", [0, 0])
            nr = z.get("next_radius", 1e4)
            if d2(p, nc) > max(nr - margin, nr * 0.85):
                return False
        return True

    def zone_pressure(self):
        """(urgency 0..3, safe point) for staying inside the zone."""
        z = self.zone
        if not z:
            return 0.0, None
        walk = self.stats.get("walk_speed", 11.7)
        c_now, r_now = self.zone_circle(self.t)
        d_now = d2(self.pos, c_now)
        nc = z.get("next_center", c_now)
        nr = z.get("next_radius", r_now)
        s0 = z.get("shrink_start", 1e9)
        s1 = z.get("shrink_end", 1e9)
        if d_now > r_now - 3:
            return 3.0, c_now
        # where must we be by the end of this shrink
        need = d2(self.pos, nc) - max(nr * 0.7, nr - 25.0)
        if need <= 0:
            # inside the safe part of the next circle
            if d_now > r_now - 15:
                return 1.5, c_now
            return 0.0, None
        travel = need / (walk * 0.75)
        slack = s1 - self.t - travel
        if slack < 6:
            return 2.6, nc
        if slack < 18 or self.t > s0 - 10:
            return 1.6, nc
        if slack < 40:
            return 0.8, nc
        return 0.2, nc

    # ================================================================ intent
    def _choose_intent(self, threats, target):
        t = self.t
        pos = self.pos
        urg, zpt = self.zone_pressure()
        hp_frac = self.hp / self.max_hp
        near = [e for e in threats if e["age"] < 6.0 and e["d"] < 230]
        hostile = [e for e in near if e["hostile"]]
        hostile_los = [e for e in hostile if e["los"] or e["d"] < 60]

        # ---- fight evaluation (only hostiles shoot back)
        adv = 99.0
        if target is not None:
            tr = target["tr"]
            spd = math.hypot(tr.vel[0], tr.vel[1])
            mydps = self.our_dps(target["d"], spd, True)
            ttk_them = tr.hp / max(mydps, 1.0)
            inc = sum(e["dps"] for e in near if e["los"] or e["d"] < 60)
            ttk_us = self.hp / max(inc, 1.0)
            adv = ttk_us / max(ttk_them, 0.1)
        self.adv = adv
        recent_dmg = sum(e[1] for e in self.dmg_events if t - e[0] < 3.0)

        ffa = self.mode == "ffa"
        ret_hp = FFA_RET_HP if ffa else RET_HP
        ret_adv = FFA_RET_ADV if ffa else RET_ADV
        regen_hp = FFA_REGEN_HP if ffa else REGEN_HP

        # ---- fight or flee: project the exchange with everyone who can shoot us now
        engaged = [e for e in hostile if (e["los"] or e["d"] < 60) and e["age"] < 3.0]
        proj = None
        inc_now = 0.0
        if engaged:
            foes = []
            for e in engaged:
                tr_e = e["tr"]
                spd_e = math.hypot(tr_e.vel[0], tr_e.vel[1])
                od = self.our_dps(e["d"], spd_e, True)
                ttk = tr_e.hp / max(od, 1.0) + (0.0 if e["los"] else 2.0)
                foes.append((ttk, e["dps"]))
                inc_now += e["dps"]
            proj = CombatModel.project(self.hp, foes) + (len(foes),)
        self.proj = proj
        safety = (0.25 if ffa else 0.06) * self.max_hp
        if AGGRO and not ffa:
            # the practice squad never loses track of us: hiding only hands them time
            self.retreating = False
            proj = None
        if RET2 and not ffa and len(engaged) >= RET2 and not self.retreating:
            self.retreating = True
        if self.retreating:
            if not engaged or hp_frac > 0.9 or (proj is not None and proj[0] >= proj[3] and proj[1] > safety * 2):
                self.retreating = False
        elif proj is not None:
            kills_p, hp_left, t_p, n_p = proj
            # value everything in kills: HP is worth kappa kills per point (measured loadout model)
            kappa = self._kills_per_hp()
            if ffa:
                kappa *= 1.5   # in a free-for-all, staying alive is the score itself
            fight_val = kills_p + hp_left * kappa
            if kills_p < n_p:
                cv = self._find_cover(hostile)
                if cv is not None:
                    spd = self.stats.get("boost_speed", 19.5) if self.energy > 30 else self.stats.get("walk_speed", 11.7)
                    t_c = d2(cv, pos) / max(spd, 1.0)
                    cost = inc_now * t_c * 0.7
                    flee_val = max(0.0, self.hp - cost) * kappa * 0.85   # chased retreats do not always work
                    if cost < self.hp * 0.85 and flee_val > fight_val + 0.3:
                        self.retreating = True

        # ---- zone emergency
        if urg >= 2.6 and zpt is not None:
            self.intent = "ZONE"
            return zpt, "fast"

        if self.retreating and hostile:
            cv = self._find_cover(hostile)
            if cv is not None:
                self.intent = "RETREAT"
                return cv, "retreat"
            if target is not None:
                # cover too far to reach under fire: fight back instead of running
                self.intent = "FIGHT"
                return None, "fight"

        # ---- supply drop being captured: hold the ring unless someone is shooting at us
        cap = self._capturing_drop()
        if cap is not None and not hostile_los:
            self.intent = "CAPTURE"
            return cap, ("fast" if d2(cap, pos) > 3.0 else "stop")

        # ---- hostile engagement
        if target is not None and target["hostile"] and (target["d"] < 230 or t - target["tr"].last_hit_us < 3):
            self.intent = "FIGHT"
            return None, "fight"

        if urg >= 1.6 and zpt is not None:
            self.intent = "ZONE"
            return zpt, "fast"

        # ---- regenerate after a fight before starting anything new
        in_combat = t - self.last_combat_t < 4.0
        quiet = True
        if AGGRO and not ffa:
            quiet = not any(tr.hostile and t - tr.t_seen < AGGRO_QUIET for tr in self.tracks.values())
        if self.hp < self._hp_needed(hostile, ffa) and not hostile_los and quiet:
            cv = self._regen_spot(hostile or near)
            self.intent = "REGEN"
            # grab very close cubes on the way
            cu = self._nearest_cube(12.0)
            if cu is not None and not in_combat:
                return cu, "fast"
            return cv, ("fast" if cv is not None and d2(cv, pos) > 4 else "stop")

        # ---- loot first only when it is close (or a real weapon upgrade / supply drop)
        g = self._loot_goal(near, urg)
        if LOOT_PLUS and g is not None and self.loot_kind in ("npc", "crate") and self.my_cubes < 10 \
                and d2(g, pos) < 70 and not hostile_los:
            self.intent = "LOOT"
            return g, "fast"
        if g is not None and self.loot_kind in ("pick", "cube", "drop"):
            gd = d2(g, pos)
            if LOOT_PLUS and not ffa:
                urgent_loot = True
            elif BLITZ and not ffa:
                urgent_loot = ((self.loot_kind == "drop" and gd < 80) or (self.loot_kind == "pick" and gd < 45)
                               or (self.loot_kind == "cube" and gd < 18) or target is None)
            else:
                urgent_loot = (self.loot_kind == "drop" or (self.loot_kind == "pick" and gd < 120)
                               or (self.loot_kind == "cube" and gd < (CUBE_URG if self.my_cubes < 10 else 35))
                               or target is None or target["hostile"] or ffa)
            if urgent_loot:
                self.intent = "LOOT"
                return g, "fast"

        # ---- passive targets: free kills, go close (practice squad only)
        farm_window = t < FARM_EARLY and self.my_cubes < 6 and g is not None and target is not None and target["d"] > 60
        if target is not None and not target["hostile"] and t >= 2.6 and not farm_window:
            if not ffa and (HUNT_WHEN_HOSTILE or not hostile):
                self.intent = "HUNT"
                return None, "hunt"
            # free-for-all: only take fights we are likely to win (weak or busy targets)
            tr = target["tr"]
            if FFA_STYLE == "aggressor":
                self.intent = "HUNT"
                return None, "hunt"
            if FFA_STYLE == "survivor":
                finish = tr.hp < 0.3 * tr.max_hp and target["d"] < 90 and hp_frac > 0.6
                if finish:
                    self.intent = "FIGHT"
                    return None, "fight"
            else:
                weak = tr.hp < 0.5 * tr.max_hp or self._distracted(tr)
                if (weak and adv > 1.2 and hp_frac > 0.5) or adv > 2.5:
                    self.intent = "FIGHT"
                    return None, "fight"

        # look for prey: remembered passive mechs
        prey = None
        hn = [o for o in threats if o["hostile"] and o["age"] < 12.0] if PREY_AWAY > 0 else []
        best_pd = None
        for e in threats:
            if not e["hostile"] and e["age"] < 20 and self.zone_ok(e["pos"], e["d"] / 11.0, 10.0):
                pd = e["d"]
                if hn:
                    near_h = min(d2(o["pos"], e["pos"]) for o in hn)
                    pd += PREY_AWAY * max(0.0, 140.0 - near_h)
                if prey is None or pd < best_pd:
                    prey, best_pd = e, pd
        held = self._held()
        armed = ("laser" in held or "lightning" in held or self.my_cubes >= 4)
        if ffa or not self._prey_ok():
            prey = None
        if prey is not None and armed and prey["d"] < 220 and prey["age"] < 15:
            # points come only from kills: a known victim beats farming crates
            near_farm = g is not None and d2(g, pos) < 30.0
            if not near_farm and self.loot_kind not in ("npc", "crate") or g is not None and d2(g, pos) > 30.0:
                self.intent = "PROWL"
                return prey["pos"], "fast"
        if g is not None or self.loot_kind in ("npc", "crate", "cube", "pick"):
            self.intent = "LOOT"
            return g, ("fast" if g is not None else "stop")
        if urg > 0.5 and zpt is not None:
            self.intent = "ZONE"
            return zpt, "fast"
        if prey is not None:
            self.intent = "PROWL"
            return prey["pos"], "fast"
        self.intent = "EXPLORE"
        if ffa:
            return self._ffa_hold_spot(), "walk"
        return self._explore_goal(), "walk"

    def _ffa_hold_spot(self):
        """Free-for-all idle position: a wall-side spot well inside the next circle."""
        t = self.t
        if getattr(self, "_hold", None) is not None and t - self._hold_t < 10.0 and self.zone_ok(self._hold, 3.0, 12.0):
            return self._hold
        z = self.zone
        nc = z.get("next_center", [self.pos[0], self.pos[1]]) if z else [self.pos[0], self.pos[1]]
        nr = z.get("next_radius", 200) if z else 200
        best = None
        if self.world is not None:
            for dd, x, y in self.world.anchors_near(nc[0], nc[1], max(30.0, nr * 0.55))[:300]:
                if not self.zone_ok((x, y), d2((x, y), self.pos) / 11.0, 10.0):
                    continue
                sc = d2((x, y), self.pos) * 0.5 + math.sqrt(dd) * 0.8
                if best is None or sc < best[0]:
                    best = (sc, [x, y, 0.0])
        self._hold = best[1] if best else [nc[0], nc[1], 0.0]
        self._hold_t = t
        return self._hold

    def _capturing_drop(self):
        rad = self.c("drops.radius", 8.0)
        for dr in self.obs.get("drops") or []:
            dp = dr.get("pos")
            if not dp:
                continue
            holder = dr.get("holder")
            if holder is not None and holder != self.my_id:
                continue
            if dr.get("lands_in", 99) > 1.0:
                continue
            if d2(dp, self.pos) < rad + 6.0 and self.zone_ok(dp, 2.0, 4.0):
                return [dp[0], dp[1], dp[2]]
        return None

    def _cube_clusters(self):
        pts = [p for (p, ts) in self.cubes_seen.values()]
        out = []
        used = [False] * len(pts)
        for i, p in enumerate(pts):
            if used[i]:
                continue
            grp = [p]
            used[i] = True
            for j in range(i + 1, len(pts)):
                if not used[j] and d2(pts[j], p) < 7.0 and abs(pts[j][2] - p[2]) < 3:
                    used[j] = True
                    grp.append(pts[j])
            # go to the cube nearest to us in the group
            gp = min(grp, key=lambda q: d2(q, self.pos))
            out.append((gp, len(grp)))
        return out

    def _kills_per_hp(self):
        """Kills per HP point, measured in this match (prior: one kill per 300 HP lost)."""
        k0 = 2.0
        return (self.kills + k0) / (self.hp_lost + k0 * 300.0)

    def _kills_per_hp_model(self):
        held = self._held()
        stats = {w.get("kind"): (w.get("stats") or {}) for w in self.weapons if w}
        return self.model.loadout_value(held, stats, self.our_mult(), self._enemy_kinds_seen())

    def _hp_needed(self, hostile, ffa):
        """HP we want before starting anything new: enough to win a duel against a typical hunter."""
        m = self.model
        best = None
        for bd in (35.0, 52.0, 90.0, 135.0):
            ours = self.our_dps(bd, 8.0, True)
            theirs = m.rate("gun", bd) + 0.5 * m.rate("laser", bd)
            if ours <= 1.0:
                continue
            cost = theirs * (1400.0 / ours)
            if best is None or cost < best:
                best = cost
        if best is None:
            return 0.6 * self.max_hp
        need = best * self.model.shooters_avg() * REGEN_MULT
        if ffa:
            need *= 1.6
        return min(0.9 * self.max_hp, max(0.15 * self.max_hp, need))

    def _nearest_cube(self, rmax):
        best = None
        for cid, (p, ts) in self.cubes_seen.items():
            d = d2(p, self.pos)
            if d < rmax and abs(p[2] - self.pos[2]) < 3.5 and (best is None or d < best[0]):
                best = (d, p)
        return best[1] if best else None

    def _regen_spot(self, threats):
        """Where to wait while HP regenerates: out of sight of known threats."""
        if threats:
            cv = self._find_cover(threats)
            if cv is not None:
                return cv
        # no known threats: stay put next to the nearest wall-ish spot (cheap: current position)
        return [self.pos[0], self.pos[1], self.pos[2]]

    # ---------------------------------------------------------------- loot
    def _enemy_kinds_seen(self):
        kinds = []
        for tr in self.tracks.values():
            if tr.hostile or self.mode == "ffa":
                kinds.extend(tr.weapons)
        return kinds or ["gun"]

    def _loadout_value(self, kinds):
        stats = {}
        for w in self.weapons:
            if w:
                stats[w.get("kind")] = w.get("stats") or {}
        v = self.model.loadout_value(kinds, stats, self.our_mult(), self._enemy_kinds_seen())
        # carrying a mortar makes enemy lasers hit 60% harder (cook-off)
        if "mortar" in kinds:
            ek = self._enemy_kinds_seen()
            laser_share = ek.count("laser") / max(len(ek), 1)
            v *= 1.0 - 0.3 * laser_share
        return v

    def _loadout_gain(self, kind):
        """Relative improvement in expected damage throughput if we take this weapon."""
        held = self._held()
        if kind in held:
            return 0.0, None
        slots = [w.get("kind") if w else None for w in self.weapons]
        if LIGHT_FIRST and self.mode != "ffa":
            if kind == "lightning":
                if None in slots:
                    return 1.0, None
                for wi, k in enumerate(slots):
                    if k not in ("gun", "lightning"):
                        return 1.0, wi
                return 1.0, (1 if len(slots) > 1 else 0)
            if "lightning" in held:
                return 0.0, None
        cur = self._loadout_value(held)
        if None in slots:
            new = self._loadout_value(held + [kind])
            return (new - cur) / max(cur, 1e-9), None
        best = (0.0, None)
        for wi, k in enumerate(slots):
            w = self.weapons[wi]
            if w and w.get("kind") == "laser" and (w.get("charge") or 0) > 0:
                continue  # never drop a charging laser
            rest = [x for j, x in enumerate(slots) if j != wi]
            new = self._loadout_value(rest + [kind])
            g = (new - cur) / max(cur, 1e-9)
            if g > best[0]:
                best = (g, wi)
        return best

    def _loot_goal(self, near, urg):
        t = self.t
        pos = self.pos
        walk = self.stats.get("walk_speed", 11.7)
        hp_frac = self.hp / self.max_hp
        danger = any(e["d"] < 120 and e["age"] < 4 and e["hostile"] for e in near)
        best = None
        self.loot_kind = None
        w = self.world
        my_n = w.nearest_walkable(pos[0], pos[1], 4) if w is not None else -1
        my_comp = w.comp[my_n] if my_n >= 0 else -1

        def consider(p, value, kind, stay=False, ttk=0.0):
            nonlocal best
            if stay:
                u = value / (ttk + 2.5)
                if best is None or u > best[0]:
                    best = (u, None, kind)
                return
            if w is not None and p[2] > w.ground_z(p[0], p[1]) + 3.0:
                return  # on a roof
            d = d2(p, pos)
            eta = d / walk
            if not self.zone_ok(p, eta, 8.0):
                return
            if w is not None and d > 6.0:
                n = w.nearest_walkable(p[0], p[1], 3)
                if n < 0:
                    return
                if my_comp >= 0 and w.comp[n] != my_comp:
                    return
            u = value / (eta + 2.5)
            if danger:
                u *= 0.5
            if best is None or u > best[0]:
                best = (u, p, kind)

        for pid, (kind, p, ts) in self.pickups.items():
            gain, _ = self._loadout_gain(kind)
            if gain > 0.06:
                consider(p, 40.0 * min(gain, 1.0), "pick")
        for cp, cnt in self._cube_clusters():
            if abs(cp[2] - pos[2]) > 6 and d2(cp, pos) < 8:
                continue
            per = CUBE_VAL if self.my_cubes < 10 else CUBE_VAL * 0.46
            consider(cp, per * min(cnt, 4) ** 0.85, "cube")
        if not danger or hp_frac > 0.7:
            for nid, (p, v, hp, ts) in self.npcs.items():
                if t - ts > 20:
                    continue
                val = 9.0 if self.my_cubes < 10 else 3.0
                d = d2(p, pos)
                if d < 45 and t - ts < 1.0 and self._has_los_to(p, 1.75):
                    consider(p, val, "npc", stay=True, ttk=hp / max(self.our_dps(d, 3.0), 1.0))
                    continue
                consider(self._approach_point(p, 32.0), val * 0.8, "npc")
            for cid, (p, hp, ts) in self.crates.items():
                val = 5.0 if self.my_cubes < 10 else 2.0
                if len(self._held()) < 2:
                    val += 3.0
                d = d2(p, pos)
                if d < 40 and self._has_los_to(p, 1.2):
                    consider(p, val, "crate", stay=True, ttk=hp / max(self.our_dps(d, 0.0), 1.0))
                    continue
                consider(self._approach_point(p, 18.0), val, "crate")
        for dr in self.obs.get("drops") or []:
            dp = dr.get("pos")
            if not dp:
                continue
            d = d2(dp, pos)
            eta = d / walk
            holder = dr.get("holder")
            if holder is not None and holder != self.my_id:
                continue
            if eta + 2 < dr.get("lands_in", 0) - 6:
                continue
            if danger:
                continue
            consider(dp, 16.0 if hp_frac < 0.8 else 11.0, "drop")
        if best is None:
            return None
        self.loot_kind = best[2]
        return best[1]

    def _approach_point(self, p, dist):
        v = unit((self.pos[0] - p[0], self.pos[1] - p[1]))
        dd = d2(self.pos, p)
        if dd <= dist:
            return [self.pos[0], self.pos[1], self.pos[2]]
        return [p[0] + v[0] * dist, p[1] + v[1] * dist, p[2]]

    def _has_los_to(self, p, h):
        m = (self.pos[0], self.pos[1], self.pos[2] + self.stats.get("muzzle_height", 7.0))
        return self.los(m, (p[0], p[1], p[2] + h))

    def _explore_goal(self):
        t = self.t
        if self.explore_goal is not None and t - self.explore_t < 12.0 and d2(self.explore_goal, self.pos) > 12:
            return self.explore_goal
        # pick a random point inside the (next) safe circle, biased toward open areas
        z = self.zone
        if z:
            nc = z.get("next_center", [0, 0])
            nr = z.get("next_radius", 300)
        else:
            nc, nr = [0, 0], 300
        best = None
        for _ in range(12):
            a = self.rng.uniform(-math.pi, math.pi)
            r = nr * math.sqrt(self.rng.uniform(0.0, 0.35))
            p = [nc[0] + math.cos(a) * r, nc[1] + math.sin(a) * r, 0.0]
            if self.world is not None and not self.world.walkable(p[0], p[1]):
                continue
            d = d2(p, self.pos)
            sc = -abs(d - 90.0)
            if best is None or sc > best[0]:
                best = (sc, p)
        if best is None:
            best = (0, [nc[0], nc[1], 0.0])
        self.explore_goal = best[1]
        self.explore_t = t
        return self.explore_goal

    # ---------------------------------------------------------------- cover
    def _find_cover(self, threats):
        """Nearest wall-side spot that no current threat can see."""
        t = self.t
        if self.world is None:
            return None
        w = self.world
        pos = self.pos
        sees = [e for e in threats if e["age"] < 8.0][:4]
        if not sees:
            return None
        if self.cover is not None and t - self.cover_t < 1.0:
            return self.cover
        if self.cover is not None and d2(self.cover, pos) > 3:
            # keep the current cover while it still hides us from everyone
            if self._hidden_at(self.cover, sees):
                self.cover_t = t
                return self.cover
        walk = self.stats.get("walk_speed", 11.7)
        tlim = min(time.perf_counter() + 0.008, self.deadline)
        best = None
        cands = w.anchors_near(pos[0], pos[1], 130.0)
        for k, (dd, x, y) in enumerate(cands):
            if k > 150 or time.perf_counter() > tlim:
                break
            dist = math.sqrt(dd)
            if best is not None and dist * 0.8 > best[0] + 60:
                break
            if not self.zone_ok((x, y), dist / walk, 5.0):
                continue
            if abs(w.ground_z(x, y) - w.ground_z(pos[0], pos[1])) > 4.0 and dist > 15:
                continue
            p = (x, y)
            if not self._hidden_at(p, sees):
                continue
            # moving toward a threat to reach cover is bad
            closer = 0.0
            for e in sees:
                dn = d2(p, e["pos"])
                if dn < 40:
                    closer += (40 - dn) * 2.0
                closer += max(0.0, e["d"] - dn) * 0.8
            score = dist * 0.8 + closer
            if best is None or score < best[0]:
                best = (score, [x, y, w.ground_z(x, y)])
        if best is not None:
            self.cover = best[1]
            self.cover_t = t
            return self.cover
        self.cover = None
        return None

    def _hidden_at(self, p, sees):
        w = self.world
        gz = w.ground_z(p[0], p[1])
        for e in sees:
            sc = e["scale"]
            te = (e["pos"][0], e["pos"][1], e["pos"][2] + 8.0 * sc)
            if self.vis(te, (p[0], p[1], gz + 4.5 * self.scale)) or \
                    self.vis(te, (p[0], p[1], gz + 8.6 * self.scale)):
                return False
        return True

    # ================================================================ motor
    def _motor(self, goal, style, threats, target):
        t = self.t
        pos = self.pos
        walk_spd = self.stats.get("walk_speed", 11.7)
        mv = (0.0, 0.0)
        speed = 1.0
        boost = False
        jump = False
        held = self._held()
        has_light = "lightning" in held
        energy_reserve = 40 if has_light else 15

        if style in ("fight", "hunt") and target is not None:
            self.hunt_boost = False
            self.kite_boost = False
            mv, speed = self._fight_move(target, threats, passive=(style == "hunt"))
            boost = self.hunt_boost or self.kite_boost
            if FIGHT_BOOST and style == "fight" and not has_light and self.energy > 40:
                charging = any(w and w.get("kind") == "laser" and (w.get("charge") or 0) > 0 for w in self.weapons)
                if not charging and any(e["hostile"] and e["los"] for e in threats):
                    boost = True
        elif style == "stop":
            mv, speed = (0.0, 0.0), 0.0
        elif style == "retreat" and goal is not None:
            mv = self._steer_to(goal)
            dist = d2(goal, pos)
            exposed = any(e["los"] and e["hostile"] for e in threats)
            if exposed and dist > 4.0:
                if t > self.strafe_until:
                    self.strafe_sign = -self.strafe_sign
                    self.strafe_until = t + self.rng.uniform(0.35, 0.8)
                perp = (-mv[1] * self.strafe_sign, mv[0] * self.strafe_sign)
                cand = unit((mv[0] + perp[0] * 0.7, mv[1] + perp[1] * 0.7))
                if self.world is None or self.world.walkable(pos[0] + cand[0] * 5, pos[1] + cand[1] * 5):
                    mv = cand
            if dist < 2.0:
                speed = 0.0
            elif dist > 12 and self.energy > energy_reserve + 10:
                boost = True
        elif goal is not None:
            mv = self._steer_to(goal)
            dist = d2(goal, pos)
            if style == "walk":
                speed = 1.0
            if style == "fast" and dist > 25 and self.energy > energy_reserve + 20:
                boost = self.intent in ("ZONE", "RETREAT") or (TRAVEL_BOOST and self.intent in ("LOOT", "PROWL", "EXPLORE"))
            if dist < 2.0:
                speed = 0.0
        else:
            mv = (0.0, 0.0)
            speed = 0.0

        # dodge mortar shells and close bullets
        dv = self._dodge()
        if dv[0] or dv[1]:
            mv = unit((mv[0] * max(speed, 0.3) + dv[0] * 1.6, mv[1] * max(speed, 0.3) + dv[1] * 1.6))
            speed = 1.0

        # stuck handling
        mv, jump2, boost2 = self._unstick(mv, speed)
        jump = jump or jump2
        boost = boost or boost2
        if boost and self.energy < 12:
            boost = False
        return (mv[0] * speed, mv[1] * speed), boost, jump

    def _fight_move(self, target, threats, passive=False):
        t = self.t
        pos = self.pos
        tr = target["tr"]
        d = target["d"]
        held = self._held()
        their = tr.weapons
        if passive:
            # it is not shooting back: get close and behind it (rear hits deal x1.5)
            hi = self._hunt_distance()
            lo = max(8.0, hi - 18.0)
            face = tr.aim[0] if tr.aim else 0.0
            back = (tr.pos[0] - math.cos(face) * (lo + hi) * 0.5, tr.pos[1] - math.sin(face) * (lo + hi) * 0.5)
            rel = math.atan2(pos[1] - tr.pos[1], pos[0] - tr.pos[0])
            in_rear = abs(angn(rel - face)) > math.radians(105) or not FLANK
            if d > hi + 25 or not target.get("myl", True):
                self.hunt_boost = d > 60 and self.energy > (70 if "lightning" in held else 35)
                goal = back if (FLANK and (self.world is None or self.world.walkable(back[0], back[1]))) else tr.pos
                return self._steer_to(goal), 1.0
            if FLANK and not in_rear and (self.world is None or self.world.walkable(back[0], back[1])):
                # swing around behind it, keeping our distance
                return self._steer_to(back), 1.0
            to = unit((tr.pos[0] - pos[0], tr.pos[1] - pos[1]))
            radial = 1.0 if d > hi else (-0.6 if d < lo else 0.0)
            # follow it and stay in its rear arc
            bv = unit((back[0] - pos[0], back[1] - pos[1])) if FLANK else (0.0, 0.0)
            mv = (to[0] * radial + bv[0] * 0.8, to[1] * radial + bv[1] * 0.8)
            if math.hypot(*mv) < 0.1:
                return (0.0, 0.0), 0.0
            return unit(mv), 1.0
        if PEEK and self.world is not None:
            hs = [e for e in threats if e["hostile"] and e["age"] < 4.0 and e["d"] < 240]
            if any(e["tr"].id == tr.id for e in hs):
                expose, hide = self._peek_spots(tr, hs)
                ready = self._shot_ready(tr, d)
                goal = expose if ready else hide
                self.pk_state = "expose" if ready else "hide"
                if goal is not None:
                    if d2(goal, pos) > 1.5:
                        return self._steer_to(goal), 1.0
                    if not ready:
                        return (0.0, 0.0), 0.0
        best_d = self._best_range(tr, threats) * FIGHT_RANGE_SCALE
        if RANGE_MIN > 0 and "lightning" not in held and not ("lightning" in their):
            best_d = max(best_d, RANGE_MIN)
        lo, hi = best_d - 18.0, best_d + 18.0
        # weak target and we are healthy: close in to finish
        if tr.hp < 250 and self.hp > 0.5 * self.max_hp:
            hi = min(hi, max(lo + 10, 90.0))
        to = unit((tr.pos[0] - pos[0], tr.pos[1] - pos[1]))
        radial = 0.0
        if d > hi:
            radial = 1.0
        elif d < lo:
            radial = -1.0
        self.kite_boost = False
        if KITE and d < lo - 5 and "lightning" not in held:
            hs = [e for e in threats if e["hostile"] and e["age"] < 4.0 and e["d"] < 250]
            if hs:
                cx = sum(e["pos"][0] for e in hs) / len(hs)
                cy = sum(e["pos"][1] for e in hs) / len(hs)
                away = unit((pos[0] - cx, pos[1] - cy))
                if self.world is None or self.world.walkable(pos[0] + away[0] * 8, pos[1] + away[1] * 8):
                    to = (-away[0], -away[1])
                if self.energy > 30:
                    self.kite_boost = True
        if t > self.strafe_until:
            self.strafe_sign = -self.strafe_sign if self.rng.random() < 0.75 else self.strafe_sign
            self.strafe_until = t + self.rng.uniform(STRAFE_LO, STRAFE_HI)
        perp = (-to[1] * self.strafe_sign, to[0] * self.strafe_sign)
        mv = (to[0] * radial * 1.0 + perp[0] * 0.9, to[1] * radial * 1.0 + perp[1] * 0.9)
        # walls: if the strafe direction is blocked, flip
        if self.world is not None:
            ahead = (pos[0] + mv[0] * 6, pos[1] + mv[1] * 6)
            if not self.world.walkable(ahead[0], ahead[1]):
                self.strafe_sign = -self.strafe_sign
                perp = (-to[1] * self.strafe_sign, to[0] * self.strafe_sign)
                mv = (to[0] * radial + perp[0] * 0.9, to[1] * radial + perp[1] * 0.9)
                ahead = (pos[0] + mv[0] * 6, pos[1] + mv[1] * 6)
                if not self.world.walkable(ahead[0], ahead[1]):
                    mv = self._steer_to(tr.pos) if radial >= 0 else (-to[0], -to[1])
        # several shooters: slide to a nearby spot where only our target can see us
        shooters = [e for e in threats if e["hostile"] and e["los"] and e["age"] < 2.0 and e["d"] < 230]
        if ISO and len(shooters) >= 2:
            spot = self._isolation_spot(tr, shooters)
            if spot is not None and d2(spot, pos) > 2.0:
                sv = unit((spot[0] - pos[0], spot[1] - pos[1]))
                mv = (mv[0] * 0.5 + sv[0] * 1.6, mv[1] * 0.5 + sv[1] * 1.6)
                return unit(mv), 1.0
        # stay close to cover from the other hunters while fighting this one
        others = [e for e in threats if e["hostile"] and e["tr"].id != tr.id and e["age"] < 8.0 and e["d"] < 260]
        anc = None
        if others:
            if self.t - self.fight_cover_t > 0.6:
                self.fight_cover = self._find_cover(others)
                self.fight_cover_t = self.t
            anc = self.fight_cover
            wgt = 1.4
        else:
            if self.t - self.fight_cover_t > 1.5 and self.world is not None:
                a = self.world.anchors_near(pos[0], pos[1], 35.0)
                self.fight_cover = [a[0][1], a[0][2], 0.0] if a else None
                self.fight_cover_t = self.t
            anc = self.fight_cover
            wgt = 0.5
        if anc is not None:
            da = d2(anc, pos)
            if 4.0 < da < 60.0:
                av = unit((anc[0] - pos[0], anc[1] - pos[1]))
                mv = (mv[0] + av[0] * wgt, mv[1] + av[1] * wgt)
        # drift through nearby cubes while fighting
        cu = self._nearest_cube(22.0)
        if cu is not None:
            cv = unit((cu[0] - pos[0], cu[1] - pos[1]))
            mv = (mv[0] + cv[0] * 1.3, mv[1] + cv[1] * 1.3)
        # stay inside the zone
        urg, zpt = self.zone_pressure()
        if urg > 0.7 and zpt is not None:
            zv = unit((zpt[0] - pos[0], zpt[1] - pos[1]))
            mv = (mv[0] + zv[0] * urg, mv[1] + zv[1] * urg)
        # no line of sight and target close: go get it via the path
        if not target.get("myl", True) and d < 120 and self.hp > 0.45 * self.max_hp:
            mv = self._steer_to(tr.pos)
        return unit(mv), 1.0

    def _hunt_distance(self):
        """Farthest distance where our damage output is still close to its best (no return fire)."""
        cands = (14.0, 22.0, 30.0, 40.0, 52.0, 70.0, 90.0)
        vals = [(dd, self.our_dps(dd, 3.0, True)) for dd in cands]
        top = max(v for _, v in vals) if vals else 0.0
        if top <= 0:
            return 30.0
        ok = [dd for dd, v in vals if v >= 0.92 * top]
        hd = max(ok) if ok else 30.0
        if HUNT_D_MIN > 0 and "laser" in self._held() and self.mode != "ffa":
            hd = max(hd, HUNT_D_MIN)
        return hd

    def _shot_ready(self, tr, d):
        """Is our main weapon ready to deliver a hit now (so exposing ourselves pays)?"""
        laser = self._weapon("laser")
        if laser is not None and d < 275:
            st = laser.get("stats") or {}
            ch = laser.get("charge") or 0.0
            if ch >= st.get("charge", 0.75) * 0.9:
                return True
            # laser recharging: hide unless the gun alone is clearly winning this exchange
            gun = self._weapon("gun")
            if gun is not None and gun.get("reload", 0) <= 0 and gun.get("ammo", 1) > 0 and (self.adv > 1.6 or tr.hp < 320):
                return True
            return False
        lt = self._weapon("lightning")
        if lt is not None and d < 55 and self.energy > 14:
            return True
        gun = self._weapon("gun")
        if gun is not None and gun.get("reload", 0) <= 0 and gun.get("ammo", 1) > 0 and d < 228:
            return True
        return False

    def _peek_spots(self, tr, hs):
        """Nearby spots: one that shows us to the target only, one hidden from every known hostile."""
        t = self.t
        if t - self.pk_t < 0.3 and self.pk_for == tr.id:
            return self.pk_cache
        w = self.world
        pos = self.pos
        gz0 = w.ground_z(pos[0], pos[1])
        tlim = min(time.perf_counter() + 0.007, self.deadline)
        cands = [(pos[0], pos[1])]
        for r in (4.0, 8.0, 13.0):
            for k in range(8):
                a = k * (TAU / 8) + (0.39 if r == 8.0 else 0.0)
                cands.append((pos[0] + math.cos(a) * r, pos[1] + math.sin(a) * r))
        for (dd, x, y) in w.anchors_near(pos[0], pos[1], PEEK_R)[:8]:
            cands.append((x, y))
        best_e = None
        best_h = None
        for (x, y) in cands:
            if time.perf_counter() > tlim:
                break
            if (x, y) != (pos[0], pos[1]):
                if not w.walkable(x, y) or not w.straight_walkable(pos[0], pos[1], x, y):
                    continue
                if not self.zone_ok((x, y), 1.5, 4.0):
                    continue
            gz = w.ground_z(x, y)
            if abs(gz - gz0) > 3.0:
                continue
            body = (x, y, gz + 4.5 * self.scale)
            head = (x, y, gz + 8.0 * self.scale)
            expo = 0.0
            seen_t = False
            for e in hs:
                sc = e["scale"]
                ee = (e["pos"][0], e["pos"][1], e["pos"][2] + 8.0 * sc)
                if self.vis(ee, body) or self.vis(ee, head):
                    if e["tr"].id == tr.id:
                        seen_t = True
                    else:
                        expo += max(e["dps"], 20.0)
            dist = d2((x, y), pos)
            if seen_t:
                sc_e = expo + dist * 0.8
                if best_e is None or sc_e < best_e[0]:
                    best_e = (sc_e, [x, y, gz])
            elif expo <= 0.0:
                if best_h is None or dist < best_h[0]:
                    best_h = (dist, [x, y, gz])
        self.pk_cache = (best_e[1] if best_e else None, best_h[1] if best_h else None)
        self.pk_t = t
        self.pk_for = tr.id
        return self.pk_cache

    def _isolation_spot(self, tr, shooters):
        """Nearby walkable spot that keeps the target in view and hides us from the other shooters."""
        t = self.t
        if self.iso_spot is not None and t - self.iso_t < 0.4 and self.iso_for == tr.id:
            return self.iso_spot
        w = self.world
        if w is None:
            return None
        pos = self.pos
        gz0 = w.ground_z(pos[0], pos[1])
        tp = tr.pos
        best = None
        tlim = min(time.perf_counter() + 0.006, self.deadline)
        cands = [(pos[0], pos[1])]
        for r in (8.0, 16.0):
            for k in range(8):
                a = k * (TAU / 8) + (0.39 if r > 10 else 0.0)
                cands.append((pos[0] + math.cos(a) * r, pos[1] + math.sin(a) * r))
        for (x, y) in cands:
            if time.perf_counter() > tlim:
                break
            if (x, y) != (pos[0], pos[1]):
                if not w.walkable(x, y) or not w.straight_walkable(pos[0], pos[1], x, y):
                    continue
                if not self.zone_ok((x, y), 1.5, 4.0):
                    continue
            gz = w.ground_z(x, y)
            if abs(gz - gz0) > 3.0:
                continue
            body = (x, y, gz + 4.5 * self.scale)
            head = (x, y, gz + 8.0 * self.scale)
            exposure = 0.0
            for e in shooters:
                if e["tr"].id == tr.id:
                    continue
                sc = e["scale"]
                ee = (e["pos"][0], e["pos"][1], e["pos"][2] + 8.0 * sc)
                if self.vis(ee, body) or self.vis(ee, head):
                    exposure += max(e["dps"], 20.0)
            sees_target = self.vis((head[0], head[1], gz + self.stats.get("muzzle_height", 7.0)),
                                   (tp[0], tp[1], tp[2] + 4.5))
            score = exposure + (0.0 if sees_target else 60.0) + d2((x, y), pos) * 0.8
            if best is None or score < best[0]:
                best = (score, [x, y, gz])
        self.iso_spot = best[1] if best else None
        self.iso_t = t
        self.iso_for = tr.id
        return self.iso_spot

    def _best_range(self, tr, threats):
        """Distance where our expected DPS beats theirs by the widest margin."""
        if self.t - self.range_t < 0.5 and self.range_for == tr.id:
            return self.range_d
        spd = math.hypot(tr.vel[0], tr.vel[1])
        others = 0.0
        for e in threats:
            if e["hostile"] and e["tr"].id != tr.id and (e["los"] or e["d"] < 60):
                others += e["dps"]
        best = None
        for d in (20.0, 32.0, 45.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 185.0, 210.0):
            ours = self.our_dps(d, spd, True)
            if ours <= 1.0:
                continue
            theirs = self.enemy_dps(tr, d) + others
            sc = ours / (theirs + 25.0)
            # time to kill also matters a little: prefer faster kills when close to equal
            sc *= (1.0 + 0.002 * ours)
            if best is None or sc > best[0]:
                best = (sc, d)
        d = best[1] if best else 90.0
        self.range_d, self.range_t, self.range_for = d, self.t, tr.id
        return d

    def _steer_to(self, goal):
        pos = self.pos
        w = self.world
        if w is None:
            return unit((goal[0] - pos[0], goal[1] - pos[1]))
        dist = d2(goal, pos)
        if dist < 1.0:
            return (0.0, 0.0)
        # on a roof or not in a nav cell: go straight
        gz = w.ground_z(pos[0], pos[1])
        if pos[2] > gz + 3.0:
            return unit((goal[0] - pos[0], goal[1] - pos[1]))
        if dist < 60 and w.straight_walkable(pos[0], pos[1], goal[0], goal[1]):
            self.path = None
            return unit((goal[0] - pos[0], goal[1] - pos[1]))
        # path planning
        need_new = (self.path_goal is None or d2(self.path_goal, goal) > 10.0
                    or self.t - self.path_t > 4.0)
        if need_new and (self.astar.open is None or d2(self.path_goal or goal, goal) > 10.0 or self.t - self.path_t > 4.0):
            s = w.nearest_walkable(pos[0], pos[1], 4)
            g = w.nearest_walkable(goal[0], goal[1], 6)
            if s >= 0 and g >= 0:
                self.astar.begin(s, g)
                self.path_goal = list(goal)
                self.path_t = self.t
        if self.astar.open is not None:
            self.astar.step(max(0.0008, min(0.005, self.deadline - time.perf_counter())))
            if self.astar.done:
                self.path = [w.nav_pos(n) for n in self.astar.path]
                self.path_i = 0
        if self.path:
            # advance along path: farthest node (within 30 m) that is straight-walkable
            best = None
            i0 = self.path_i
            # move index forward to nearest node
            nearest_i, nd = i0, 1e9
            for i in range(i0, min(len(self.path), i0 + 30)):
                dd = d2(self.path[i], pos)
                if dd < nd:
                    nd, nearest_i = dd, i
            self.path_i = nearest_i
            for i in range(min(len(self.path) - 1, nearest_i + 18), nearest_i, -2):
                p = self.path[i]
                if d2(p, pos) < 32 and w.straight_walkable(pos[0], pos[1], p[0], p[1]):
                    best = p
                    break
            if best is None:
                best = self.path[min(len(self.path) - 1, nearest_i + 1)]
            if nearest_i >= len(self.path) - 2:
                return unit((goal[0] - pos[0], goal[1] - pos[1]))
            return unit((best[0] - pos[0], best[1] - pos[1]))
        return unit((goal[0] - pos[0], goal[1] - pos[1]))

    def _dodge(self):
        out = [0.0, 0.0]
        pos = self.pos
        me = (pos[0], pos[1], pos[2] + 4.5 * self.scale)
        g = self.c("weapons.mortar.gravity", 22.0)
        splash = self.c("weapons.mortar.splash_radius", 7.0)
        for p in self.obs.get("projectiles") or []:
            pp, pv = p.get("pos"), p.get("vel")
            if not pp or not pv:
                continue
            if p.get("kind") == "mortar":
                # predict landing: solve z(t) = ground
                gz = pos[2]
                a = -0.5 * g
                b = pv[2]
                cc = pp[2] - gz
                disc = b * b - 4 * a * cc
                if disc < 0:
                    continue
                tl = (-b - math.sqrt(disc)) / (2 * a)
                if tl <= 0 or tl > 6:
                    continue
                lx = pp[0] + pv[0] * tl
                ly = pp[1] + pv[1] * tl
                # where will we be
                fx = pos[0] + self.vel[0] * min(tl, 1.0)
                fy = pos[1] + self.vel[1] * min(tl, 1.0)
                dd = math.hypot(fx - lx, fy - ly)
                if dd < splash + 4.0:
                    v = unit((fx - lx, fy - ly)) if dd > 0.5 else unit((-pv[1], pv[0]))
                    out[0] += v[0] * 1.5
                    out[1] += v[1] * 1.5
                continue
            rel = (me[0] - pp[0], me[1] - pp[1], me[2] - pp[2])
            vv = pv[0] ** 2 + pv[1] ** 2 + pv[2] ** 2
            if vv < 1:
                continue
            tc = (rel[0] * pv[0] + rel[1] * pv[1] + rel[2] * pv[2]) / vv
            if tc <= 0.05 or tc > 1.2:
                continue
            cx = pp[0] + pv[0] * tc - me[0]
            cy = pp[1] + pv[1] * tc - me[1]
            miss = math.hypot(cx, cy)
            if miss < 3.5 * self.scale:
                perp = unit((-pv[1], pv[0]))
                side = -1.0 if (perp[0] * cx + perp[1] * cy) > 0 else 1.0
                out[0] += perp[0] * side
                out[1] += perp[1] * side
        return out

    def _unstick(self, mv, speed):
        t = self.t
        pos = self.pos
        self.pos_hist.append((t, pos[0], pos[1]))
        while self.pos_hist and t - self.pos_hist[0][0] > 1.0:
            self.pos_hist.pop(0)
        if t < self.unstick_until:
            return self.unstick_dir, self.unstick_n <= 1, self.unstick_n >= 2
        if t - self.unstick_last > 4.0:
            self.unstick_n = 0
        if speed > 0.5 and (mv[0] or mv[1]) and len(self.pos_hist) >= 7:
            t0, x0, y0 = self.pos_hist[0]
            if t - t0 > 0.6 and math.hypot(pos[0] - x0, pos[1] - y0) < 1.2:
                self.unstick_n += 1
                self.unstick_last = t
                base = math.atan2(mv[1], mv[0])
                turn = self.rng.choice((1, -1)) * self.rng.uniform(1.2, 2.4)
                self.unstick_dir = (math.cos(base + turn), math.sin(base + turn))
                self.unstick_until = t + 0.6
                self.pos_hist = []
                self.path_t = -99.0
                self.path_goal = None
                return self.unstick_dir, True, self.unstick_n >= 2
        return mv, False, False

    # ================================================================ loot actions
    def _take_action(self):
        rad = self.c("loot.pickup_radius", 4.0)
        best = None
        for p in self.obs.get("pickups") or []:
            pp = p.get("pos")
            if not pp:
                continue
            dh = d2(pp, self.pos)
            if dh > rad - 0.3 or abs(pp[2] - self.pos[2]) > 3.8:
                continue
            gain, drop = self._loadout_gain(p.get("kind"))
            if gain <= 0.06:
                continue
            # do not swap while an enemy is close and we would lose our only gun
            if drop is not None and any(e["d"] < 60 and e["age"] < 1 for e in self.threats):
                continue
            if best is None or gain > best[0]:
                best = (gain, p["id"], drop)
        if best is None:
            return None
        return best[1], best[2]

    # ================================================================ gunnery
    def _gunnery(self, target, threats):
        pos = self.pos
        me = self.me
        stats = self.stats
        mh = stats.get("muzzle_height", 7.0)
        muzzle = (pos[0], pos[1], pos[2] + mh)
        cur = me.get("aim") or [0.0, 0.0]
        fire = [False, False]
        reload = False
        weapons = self.weapons

        # choose what to shoot: enemy target, else npc/crate when farming
        tgt = None
        if (self.intent == "REGEN" or self.t < 2.6) and target is not None and not target["hostile"]:
            target = None
        if self.mode == "ffa" and FFA_STYLE == "survivor" and target is not None and not target["hostile"] \
                and self.intent != "FIGHT":
            target = None
        npc_tgt = self._npc_shooting_us() if NPC_FIRST else None
        if npc_tgt is not None and (target is None or not (target["hostile"] and target.get("myl", True))):
            tgt = npc_tgt
        elif target is not None and target["age"] < 1.0:
            tr = target["tr"]
            sc = target["scale"]
            d_ = target["d"]
            # long flight times: aim at the centre of a zig-zag rather than its current leg
            fl = d_ / 252.0
            if fl > 0.2 and len(tr.vhist) >= 5:
                ax, ay = tr.avg_vel(min(1.0, 1.6 * fl + 0.3))
                wb = min(1.0, (fl - 0.2) / 0.35)
                lv = [tr.vel[0] * (1 - wb) + ax * wb, tr.vel[1] * (1 - wb) + ay * wb, tr.vel[2]]
            else:
                lv = tr.vel
            tgt = {"pos": tr.pred(self.t, 0.3), "vel": lv, "h": 4.5 * sc, "r": 2.5 * sc,
                   "kind": "mech", "d": d_, "id": tr.id, "airborne": abs(tr.vel[2]) > 0.5}
        elif self.intent != "REGEN" and self.t >= 2.6:
            tgt = self._farm_target()

        if tgt is None:
            self.fire_tgt = None
            aim = self._idle_aim()
            # ambush: pre-charge the laser while a known hunter is close but out of sight
            lurking = any(e["hostile"] and e["age"] < 6.0 and e["d"] < 140 for e in threats)
            for i, w in enumerate(weapons):
                if w and w.get("kind") == "laser":
                    st = w.get("stats") or {}
                    ch = w.get("charge") or 0
                    full = st.get("charge", 0.75)
                    hold = st.get("hold_s", 2.0)
                    if ch > 0 and ch < full + hold * 0.85 and (lurking or self.t - self.last_combat_t < 4.0):
                        fire[i] = True  # keep the charge for the peek
                    elif ch <= 0 and lurking and w.get("cooldown", 0) <= 0.05 and self.intent not in ("RETREAT", "ZONE"):
                        fire[i] = True  # start charging
            gun = self._weapon("gun")
            if gun is not None:
                st = gun.get("stats") or {}
                if gun.get("reload", 0) == 0 and gun.get("ammo", st.get("mag", 12)) < st.get("mag", 12) * 0.6:
                    reload = True
            return aim, fire, reload

        tp = tgt["pos"]
        aimpt = (tp[0], tp[1], tp[2] + tgt["h"])
        tv = tgt["vel"]
        d = math.hypot(aimpt[0] - muzzle[0], aimpt[1] - muzzle[1])
        los = self.los(muzzle, aimpt) or self.los(muzzle, (aimpt[0], aimpt[1], aimpt[2] + tgt["h"] * 0.7))

        gun = self._weapon("gun")
        gspd = (gun.get("stats") or {}).get("speed", 252.0) if gun else 252.0
        g_yaw, g_pitch = self._lead(muzzle, aimpt, tv, gspd, tgt.get("airborne", False))
        l_yaw, l_pitch = self._lead(muzzle, aimpt, tv, 1e9, False, extra=0.1)

        laser = self._weapon("laser")
        laser_ready = False
        if laser is not None:
            lst = laser.get("stats") or {}
            ch = laser.get("charge") or 0.0
            laser_ready = ch >= lst.get("charge", 0.75) * 0.97
        # primary aim
        if laser_ready and los:
            yaw, pitch = l_yaw, l_pitch
        elif gun is not None and los:
            yaw, pitch = g_yaw, g_pitch
        else:
            yaw, pitch = l_yaw, l_pitch

        mortar_aim = None
        err_y = abs(angn(cur[0] - yaw))
        err_p = abs(cur[1] - pitch)
        tol_y = math.atan2(tgt["r"] * 0.8, max(d, 1.0)) + 0.004
        tol_p = math.atan2(tgt["h"] * 0.7, max(d, 1.0)) + 0.004

        for i, w in enumerate(weapons):
            if not w:
                continue
            kind = w.get("kind")
            st = w.get("stats") or {}
            if kind == "gun":
                if not los or d > st.get("range", 230) - 2:
                    continue
                if w.get("reload", 0) > 0 or w.get("ammo", 1) <= 0:
                    continue
                ey = abs(angn(cur[0] - g_yaw))
                ep = abs(cur[1] - g_pitch)
                if ey < tol_y * 1.1 and ep < tol_p:
                    fire[i] = True
            elif kind == "laser":
                ch = w.get("charge") or 0.0
                full = st.get("charge", 0.75)
                if w.get("cooldown", 0) > 0.05 and ch <= 0:
                    continue
                if d > st.get("range", 280) - 3:
                    continue
                if ch < full * 0.97:
                    # charge when the target is (or soon will be) in view
                    if los or tgt["kind"] != "mech" or d < 90:
                        fire[i] = True
                    elif ch > 0:
                        fire[i] = True
                else:
                    ey = abs(angn(cur[0] - l_yaw))
                    ep = abs(cur[1] - l_pitch)
                    held_for = ch - full
                    hold_s = st.get("hold_s", 2.0)
                    if los and ey < tol_y and ep < tol_p:
                        fire[i] = False  # release -> beam
                    elif held_for > hold_s * 0.85:
                        fire[i] = False  # it would auto fire soon anyway
                    else:
                        fire[i] = True
            elif kind == "lightning":
                rng = st.get("range", 45)
                cone = math.radians(st.get("cone_deg", 30))
                if d > rng - 1.0 or self.energy < st.get("energy", 3) + 1:
                    continue
                if not los:
                    continue
                ey = abs(angn(cur[0] - math.atan2(tp[1] - pos[1], tp[0] - pos[0])))
                if ey < cone * 0.45:
                    fire[i] = True
            elif kind == "mortar":
                if tgt["kind"] == "crate" or d < 30 or d > 225:
                    continue
                if los and (gun is not None or laser is not None) and d < 150:
                    continue  # direct weapons are better in sight
                sol = self._mortar_solution(tgt, st)
                if sol is None:
                    continue
                mortar_aim = sol
                if w.get("cooldown", 0) <= 0.05:
                    ey = abs(angn(cur[0] - sol[0]))
                    ep = abs(cur[1] - sol[1])
                    if ey < 0.03 and ep < 0.03:
                        fire[i] = True
        if mortar_aim is not None and not any(fire):
            yaw, pitch = mortar_aim
        # gun reload during lulls
        if gun is not None and not fire[weapons.index(gun)]:
            st = gun.get("stats") or {}
            if gun.get("reload", 0) == 0 and gun.get("ammo", 12) < 3 and not los:
                reload = True
        if tgt["kind"] == "mech" and los:
            hostile_t = bool(target is not None and target.get("hostile"))
            self.fire_tgt = (True, d, hostile_t)
        else:
            self.fire_tgt = None
        pmin = self.c("mech.min_pitch", -0.7)
        pmax = self.c("mech.max_pitch", 1.3)
        return (yaw, max(pmin, min(pmax, pitch))), fire, reload

    def _weapon(self, kind):
        for w in self.weapons:
            if w and w.get("kind") == kind:
                return w
        return None

    def _lead(self, muzzle, tgt, tv, speed, airborne, extra=0.0):
        p = list(tgt)
        g = self.c("mech.gravity", 22.0)
        for _ in range(3):
            dx, dy, dz = p[0] - muzzle[0], p[1] - muzzle[1], p[2] - muzzle[2]
            tt = math.sqrt(dx * dx + dy * dy + dz * dz) / max(speed, 1.0) + extra
            tt = min(tt, 1.5)
            pz = tgt[2] + tv[2] * tt
            if airborne:
                pz -= 0.5 * g * tt * tt
                if self.world is not None:
                    gz = self.world.ground_z(tgt[0] + tv[0] * tt, tgt[1] + tv[1] * tt) + 4.0
                    pz = max(pz, gz)
            p = [tgt[0] + tv[0] * tt, tgt[1] + tv[1] * tt, pz]
        dx, dy, dz = p[0] - muzzle[0], p[1] - muzzle[1], p[2] - muzzle[2]
        return math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))

    def _mortar_solution(self, tgt, st):
        v = st.get("speed", 72.0)
        g = st.get("gravity", 22.0)
        pos = self.pos
        tube = (pos[0], pos[1], pos[2] + self.stats.get("height", 9.0) + 0.5)
        tp = tgt["pos"]
        tv = tgt["vel"]
        p = [tp[0], tp[1], tp[2] + 0.5]
        sol = None
        for _ in range(3):
            dx, dy = p[0] - tube[0], p[1] - tube[1]
            x = math.hypot(dx, dy)
            y = p[2] - tube[2]
            disc = v ** 4 - g * (g * x * x + 2 * y * v * v)
            if disc < 0 or x < 1:
                return None
            # try the low arc first, then the high arc if blocked
            best = None
            for th in (math.atan2(v * v - math.sqrt(disc), g * x), math.atan2(v * v + math.sqrt(disc), g * x)):
                if th > self.c("mech.max_pitch", 1.3):
                    continue
                if self.world is not None and not self._arc_clear(tube, math.atan2(dy, dx), th, v, g, x):
                    continue
                best = th
                break
            if best is None:
                return None
            tt = x / max(v * math.cos(best), 1.0)
            sol = (math.atan2(dy, dx), best)
            p = [tp[0] + tv[0] * tt, tp[1] + tv[1] * tt, tp[2] + 0.5]
        return sol

    def _arc_clear(self, o, yaw, th, v, g, x):
        vx = v * math.cos(th)
        vz = v * math.sin(th)
        T = x / max(vx, 1e-3)
        cy, sy = math.cos(yaw), math.sin(yaw)
        n = 10
        prev = o
        for k in range(1, n):
            tt = T * k / n
            px = o[0] + cy * vx * tt
            py = o[1] + sy * vx * tt
            pz = o[2] + vz * tt - 0.5 * g * tt * tt
            if not self.world.los(prev[0], prev[1], prev[2], px, py, pz, step=2.0):
                return False
            prev = (px, py, pz)
        return True

    def _npc_shooting_us(self):
        t = self.t
        pos = self.pos
        mh = self.stats.get("muzzle_height", 7.0)
        muzzle = (pos[0], pos[1], pos[2] + mh)
        best = None
        for m in self.obs.get("mechs") or []:
            if m.get("kind") != "npc":
                continue
            if t - self.npc_attackers.get(m.get("id"), -99) > 2.5:
                continue
            d = d2(m["pos"], pos)
            if d > 80:
                continue
            c = (m["pos"][0], m["pos"][1], m["pos"][2] + 1.75)
            if not self.los(muzzle, c):
                continue
            if best is None or d < best[0]:
                best = (d, {"pos": m["pos"], "vel": m.get("vel") or [0, 0, 0], "h": 1.75, "r": 1.6,
                            "kind": "npc", "d": d, "id": m["id"], "airborne": False})
        return best[1] if best else None

    def _farm_target(self):
        pos = self.pos
        best = None
        mh = self.stats.get("muzzle_height", 7.0)
        muzzle = (pos[0], pos[1], pos[2] + mh)
        for m in self.obs.get("mechs") or []:
            if m.get("kind") != "npc":
                continue
            d = d2(m["pos"], pos)
            if d > 150:
                continue
            c = (m["pos"][0], m["pos"][1], m["pos"][2] + 1.75)
            if not self.los(muzzle, c):
                continue
            sc = -d
            if best is None or sc > best[0]:
                best = (sc, {"pos": m["pos"], "vel": m.get("vel") or [0, 0, 0], "h": 1.75, "r": 1.6,
                             "kind": "npc", "d": d, "id": m["id"], "airborne": False})
        if best is not None:
            return best[1]
        for cr in self.obs.get("crates") or []:
            d = d2(cr["pos"], pos)
            if d > 70:
                continue
            c = (cr["pos"][0], cr["pos"][1], cr["pos"][2] + 1.2)
            if not self.los(muzzle, c):
                continue
            sc = -d
            if best is None or sc > best[0]:
                best = (sc, {"pos": cr["pos"], "vel": [0, 0, 0], "h": 1.2, "r": 1.3,
                             "kind": "crate", "d": d, "id": cr["id"], "airborne": False})
        return best[1] if best else None

    def _idle_aim(self):
        t = self.t
        pos = self.pos
        if t < 2.6:
            # opening look-around: find weapons, crates and drones before choosing a plan
            cur = (self.me.get("aim") or [0.0, 0.0])[0]
            return (angn(cur + 1.2), -0.08)
        # recently damaged from a direction we can't see: look there
        for ev in reversed(self.dmg_events):
            if t - ev[0] < 2.5 and ev[3] is not None:
                return (ev[3], 0.0)
        # remembered enemies
        best = None
        for e in self.threats:
            if e["age"] < 15:
                if best is None or e["d"] < best["d"]:
                    best = e
        if best is not None:
            p = best["pos"]
            return (math.atan2(p[1] - pos[1], p[0] - pos[0]), 0.0)
        for s in reversed(self.sound_marks):
            if s[2] in ("near", "mid") and s[3] in ("gun", "laser", "laser_charge", "lightning", "step", "boost", "jet", "mortar"):
                return (s[1], 0.0)
        # sweep to discover loot and threats
        self.scan_phase += 0.32
        base = math.atan2(self.vel[1], self.vel[0]) if math.hypot(self.vel[0], self.vel[1]) > 1 else 0.0
        return (angn(base + 2.2 * math.sin(self.scan_phase * 0.5)), -0.05)

    # ================================================================ upgrades
    def _choose_upgrade(self, offer):
        """Pick the upgrade that raises (our damage output x our effective HP) the most.

        A fight's margin scales with the product of how fast we kill and how long we last,
        so every upgrade is scored by its relative effect on that product, using this
        match's measured hit rates for the weapon shares.
        """
        held = self._held()
        taken = {}
        for u in self.me.get("upgrades") or []:
            k = u if isinstance(u, str) else (u.get("id") if isinstance(u, dict) else str(u))
            taken[k] = taken.get(k, 0) + 1
        # each weapon's share of our damage, over the fights we actually have
        m = self.model
        hs = m.mode_share()
        share = {"gun": 0.0, "laser": 0.0, "lightning": 0.0, "mortar": 0.0}
        tot = 0.0
        for w in self.weapons:
            if not w:
                continue
            k = w.get("kind")
            st = w.get("stats") or {}
            v = 0.0
            for dist, wt in ((22.0, hs), (90.0, 1.0 - hs)):
                if m.in_range(k, dist, st):
                    v += wt * m.base_dps(k, st) * m.hit_p(k, dist)
            share[k] = v
            tot += v
        for k in share:
            share[k] = share[k] / tot if tot > 0 else 0.0
        armor = self.stats.get("armor", 0.0) or 0.0
        ffa = self.mode == "ffa"
        ehp_w = 1.3 if ffa else 1.0
        st_gun = (self._weapon("gun") or {}).get("stats") or self.c("weapons.gun", {})
        mag, cd, rl = st_gun.get("mag", 12), st_gun.get("cooldown", 0.24), st_gun.get("reload", 2.4)
        sus0 = (mag * cd) / (mag * cd + rl)
        sus_mag = ((mag + 8) * cd) / ((mag + 8) * cd + rl * 0.8)
        cdr = cd * 0.87
        sus_rate = (mag * cdr) / (mag * cdr + rl)
        base = {
            "plating": 0.15 * ehp_w,
            "armor": ((1 - armor) / max(0.05, 1 - armor - 0.08) - 1.0) * ehp_w,
            "nanites": 0.06 * ehp_w,
            "gyros": 0.03, "servos": 0.03, "sensors": 0.02, "jets": 0.01,
            "capacitor": 0.02 + 0.08 * share["lightning"], "reactor": 0.02 + 0.08 * share["lightning"],
            "laser_damage": 0.18 * share["laser"],
            "laser_charge": 0.085 * share["laser"],
            "laser_pierce": 0.01 * share["laser"],
            "gun_damage": 0.15 * share["gun"],
            "gun_rate": ((sus_rate / cdr) / (sus0 / cd) - 1.0) * share["gun"],
            "gun_mag": (sus_mag / sus0 - 1.0) * share["gun"],
            "gun_velocity": 0.05 * share["gun"],
            "lightning_efficiency": 0.12 * share["lightning"],
            "lightning_range": 0.05 * share["lightning"],
            "lightning_chain": 0.02 * share["lightning"],
            "mortar_splash": 0.05 * share["mortar"], "mortar_reload": 0.18 * share["mortar"],
            "mortar_velocity": 0.03 * share["mortar"],
        }
        if UPG_DMG_W != 1.0 and not ffa:
            for k in list(base):
                if k.startswith(("gun_", "laser_", "lightning_", "mortar_")):
                    base[k] *= UPG_DMG_W
        # we may still swap the starting gun away: discount upgrades for weapons we might drop
        if len(held) < 2:
            for k in list(base):
                if k.startswith("gun_"):
                    base[k] *= 0.7
        best_i, best_s = 0, -1e9
        for i, ch in enumerate(offer.get("choices") or []):
            cid = ch.get("id", "")
            s = base.get(cid, 0.01)
            if cid == "plating":
                s *= 0.97 ** taken.get(cid, 0)
            if s > best_s:
                best_i, best_s = i, s
        return best_i
