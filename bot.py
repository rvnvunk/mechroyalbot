"""Mech Royale survival bot.

Goal: place as high as possible (placement points, not kills).
Main ideas:
  1. Always stay inside the zone, and move toward the next circle early.
  2. Face threats (rear hits do x1.5) and back away while shooting (legs and torso are independent).
  3. Farm safe cubes early (crates, NPC drones, loose cubes) to get more HP.
  4. Grab a second weapon, pick defensive upgrades.
  5. Walk slowly when nothing is going on (under 5 m/s makes no footstep sound).
  6. Dodge incoming projectiles, get unstuck from walls.
"""
import math
import random

import mechio
from mechio import log

TAU = 2 * math.pi


# ---------------------------------------------------------------- helpers
def cfg(d, path, default):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def angn(a):
    return (a + math.pi) % TAU - math.pi


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm2(v):
    m = math.hypot(v[0], v[1])
    if m < 1e-9:
        return [0.0, 0.0]
    return [v[0] / m, v[1] / m]


def add(a, b, w=1.0):
    return [a[0] + b[0] * w, a[1] + b[1] * w]


def yaw_to(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def unit_yaw(y):
    return [math.cos(y), math.sin(y)]


# ---------------------------------------------------------------- bot
class Bot:
    def __init__(self):
        self.config = {}
        self.slot = 0
        self.rng = random.Random(0)
        self.last_offer = None
        self.strafe_sign = 1
        self.strafe_until = 0.0
        self.threat_bearing = None
        self.threat_until = -1.0
        self.pos_hist = []
        self.stuck_until = -1.0
        self.stuck_dir = None
        self.last_move = [0.0, 0.0]
        self.scan_phase = 0.0
        self.home_angle = None

    # ------------------------------------------------------------ lifecycle
    def on_hello(self, msg):
        self.config = msg.get("config") or {}
        self.slot = msg.get("slot", 0)
        self.rng = random.Random(msg.get("seed", 0))
        self.home_angle = self.rng.uniform(-math.pi, math.pi)
        return "SurvivorBot"

    def on_end(self, msg):
        log("end", msg)

    # ------------------------------------------------------------ main
    def on_obs(self, obs):
        t = obs.get("t", 0.0)
        me = obs.get("self") or {}
        pos = me.get("pos") or [0, 0, 0]
        vel = me.get("vel") or [0, 0, 0]
        hp = me.get("hp", 1400)
        max_hp = me.get("max_hp", 1400) or 1400
        energy = me.get("energy", 100)
        stats = me.get("stats") or {}
        walk = stats.get("walk_speed", 11.7)
        hp_frac = hp / max_hp
        weapons = me.get("weapons") or [None, None]
        held = {w.get("kind") for w in weapons if w}

        act = {}

        # ---------- remember damage direction
        for ev in obs.get("events") or []:
            if ev.get("type") == "damaged" and ev.get("bearing") is not None:
                self.threat_bearing = ev["bearing"]
                self.threat_until = t + 3.0

        # ---------- upgrade offers
        offer = obs.get("offer")
        if offer and offer.get("id") != self.last_offer:
            act["pick"] = self.choose_upgrade(offer)
            self.last_offer = offer.get("id")

        # ---------- classify what we see
        mechs, npcs = [], []
        for m in obs.get("mechs") or []:
            if m.get("id") == me.get("id"):
                continue
            (mechs if m.get("kind") == "mech" else npcs).append(m)
        for m in mechs + npcs:
            m["_d"] = dist2(pos, m["pos"])
        mechs.sort(key=lambda m: m["_d"])
        npcs.sort(key=lambda m: m["_d"])
        near_mechs = [m for m in mechs if m["_d"] < 160]

        zone = obs.get("zone") or {}
        alive = obs.get("alive", 16)

        # ---------- decide mode
        fleeing = hp_frac < 0.4 and bool(near_mechs)
        fighting = bool(mechs) and mechs[0]["_d"] < 90
        farming = (not near_mechs) and hp_frac > 0.5 and alive > 3

        move = [0.0, 0.0]
        speed = 0.45          # slow = silent by default
        boost = False

        # zone pull (always)
        zv, urgency = self.zone_vector(pos, t, zone, walk)
        move = add(move, zv, urgency)
        if urgency >= 2.0:
            speed = 1.0
            boost = urgency >= 3.0

        # threats: push away, stronger when weak
        for m in near_mechs:
            d = max(m["_d"], 1.0)
            away = norm2([pos[0] - m["pos"][0], pos[1] - m["pos"][1]])
            keep = 55.0 if not fleeing else 150.0
            if d < keep:
                w = (keep - d) / keep * (3.0 if fleeing else 1.5)
                move = add(move, away, w)
        if fleeing:
            speed = 1.0
            boost = energy > 25

        # strafing while fighting (harder to hit)
        if fighting:
            if t > self.strafe_until:
                self.strafe_sign = -self.strafe_sign
                self.strafe_until = t + self.rng.uniform(0.8, 1.8)
            to_e = norm2([mechs[0]["pos"][0] - pos[0], mechs[0]["pos"][1] - pos[1]])
            perp = [-to_e[1] * self.strafe_sign, to_e[0] * self.strafe_sign]
            move = add(move, perp, 1.2)
            speed = 1.0

        # dodge projectiles
        dodge = self.dodge_vector(pos, obs.get("projectiles") or [])
        if dodge[0] or dodge[1]:
            move = add(move, dodge, 2.5)
            speed = 1.0

        # goals (loot / supply drop) when calm
        goal = None
        take_id = None
        if not fighting and not fleeing and urgency < 2.0:
            goal, take_id = self.pick_goal(obs, pos, held, weapons, zone, t, farming)
            if goal is not None:
                gv = norm2([goal[0] - pos[0], goal[1] - pos[1]])
                gd = dist2(pos, goal)
                move = add(move, gv, 2.0 if gd > 2 else 0.0)
                speed = 1.0 if gd > 6 else 0.5
        if take_id is not None:
            act["take"] = take_id

        # idle wander inside the safe area (slow, silent)
        if goal is None and not fighting and not fleeing and math.hypot(*move) < 0.2:
            move = add(move, self.idle_vector(pos, zone, t), 0.6)

        # stuck handling
        move, jump, boost2 = self.unstick(t, pos, vel, move, speed)
        boost = boost or boost2

        mv = norm2(move)
        move_out = [mv[0] * speed, mv[1] * speed]
        self.last_move = move_out
        act["move"] = move_out
        act["boost"] = bool(boost and energy > 10)
        if jump:
            act["jump"] = True

        # ---------- aiming and firing
        target = None
        if mechs:
            target = mechs[0]
        elif farming or (npcs and npcs[0]["_d"] < 40):
            target = self.farm_target(obs, pos, npcs)

        aim_yaw, aim_pitch, fire = self.aim_and_fire(me, pos, target, weapons, t, obs)
        act["aim"] = [aim_yaw, aim_pitch]
        act["fire"] = fire

        # reload when calm
        gun = next((w for w in weapons if w and w.get("kind") == "gun"), None)
        if gun and not mechs and gun.get("reload", 0) == 0:
            mag = (gun.get("stats") or {}).get("mag", 12)
            if gun.get("ammo", mag) < mag * 0.5:
                act["reload"] = True

        return act

    # ------------------------------------------------------------ zone
    def zone_vector(self, pos, t, zone, walk):
        if not zone:
            return [0.0, 0.0], 0.0
        cc = zone.get("center", [0, 0])
        cr = zone.get("radius", 500)
        nc = zone.get("next_center", cc)
        nr = zone.get("next_radius", cr)
        s0 = zone.get("shrink_start", 0)
        s1 = zone.get("shrink_end", 1)

        # current circle right now (interpolated during shrink)
        if t >= s0 and s1 > s0:
            f = min(1.0, (t - s0) / (s1 - s0))
            cc = [cc[0] + (nc[0] - cc[0]) * f, cc[1] + (nc[1] - cc[1]) * f]
            cr = cr + (nr - cr) * f

        d_cur = dist2(pos, cc)
        d_next = dist2(pos, nc)

        # outside or near the edge of the current circle: very urgent
        if d_cur > cr - 8:
            return norm2([cc[0] - pos[0], cc[1] - pos[1]]), 4.0

        safe_r = max(nr * 0.65, min(nr, 12.0))
        if d_next <= safe_r:
            return [0.0, 0.0], 0.0
        need = d_next - safe_r
        time_left = max(0.1, s1 - t)
        travel = need / max(walk * 0.8, 1.0)
        v = norm2([nc[0] - pos[0], nc[1] - pos[1]])
        if travel > time_left - 8:
            return v, 3.0
        if travel > time_left * 0.5 or t > s0 - 15:
            return v, 2.0
        return v, 0.6

    def idle_vector(self, pos, zone, t):
        if not zone:
            return [0.0, 0.0]
        nc = zone.get("next_center", [0, 0])
        nr = zone.get("next_radius", 100)
        # hold a spot at 40% radius on our own side of the next circle
        ang = self.home_angle + 0.15 * math.sin(t * 0.05)
        spot = [nc[0] + math.cos(ang) * nr * 0.4, nc[1] + math.sin(ang) * nr * 0.4]
        if dist2(pos, spot) < 6:
            return [0.0, 0.0]
        return norm2([spot[0] - pos[0], spot[1] - pos[1]])

    def in_safe(self, p, zone, margin=0.85):
        if not zone:
            return True
        nc = zone.get("next_center", [0, 0])
        nr = zone.get("next_radius", 500)
        return dist2(p, nc) < nr * margin

    # ------------------------------------------------------------ goals
    def pick_goal(self, obs, pos, held, weapons, zone, t, farming):
        z = pos[2]
        # 1. weapon pickups (only if we have an empty slot)
        empty = any(w is None for w in weapons)
        if empty:
            best = None
            for p in obs.get("pickups") or []:
                if p.get("kind") in held:
                    continue
                pp = p["pos"]
                if pp[2] > z + 4 or not self.in_safe(pp, zone):
                    continue
                d = dist2(pos, pp)
                if d < 90 and (best is None or d < best[0]):
                    best = (d, p)
            if best:
                d, p = best
                return p["pos"], (p["id"] if d < 3.8 else None)

        # 2. supply drop: full HP + 3 cubes, only when nobody is around
        for dr in obs.get("drops") or []:
            dp = dr["pos"]
            d = dist2(pos, dp)
            if d < 140 and (dr.get("holder") in (None, (obs.get("self") or {}).get("id"))):
                if dr.get("lands_in", 99) * 11 + 10 > d or dr.get("lands_in", 99) < 5:
                    return dp, None

        if not farming:
            return None, None

        # 3. loose cubes
        best = None
        for c in obs.get("cubes") or []:
            cp = c["pos"]
            if cp[2] > z + 4 or not self.in_safe(cp, zone):
                continue
            d = dist2(pos, cp)
            if d < 70 and (best is None or d < best[0]):
                best = (d, cp)
        if best:
            return best[1], None

        # 4. crates: stand about 15 m away and shoot them (aiming does the shooting)
        best = None
        for c in obs.get("crates") or []:
            cp = c["pos"]
            if cp[2] > z + 4 or not self.in_safe(cp, zone):
                continue
            d = dist2(pos, cp)
            if d < 80 and (best is None or d < best[0]):
                best = (d, cp)
        if best:
            d, cp = best
            if d > 18:
                return cp, None
        return None, None

    def farm_target(self, obs, pos, npcs):
        if npcs and npcs[0]["_d"] < 110:
            return npcs[0]
        best = None
        for c in obs.get("crates") or []:
            d = dist2(pos, c["pos"])
            if d < 60 and c["pos"][2] < pos[2] + 4 and (best is None or d < best[0]):
                best = (d, c)
        if best:
            c = dict(best[1])
            c["vel"] = [0, 0, 0]
            c["_crate"] = True
            c["_d"] = best[0]
            return c
        return None

    # ------------------------------------------------------------ movement safety
    def dodge_vector(self, pos, projectiles):
        out = [0.0, 0.0]
        me = [pos[0], pos[1], pos[2] + 4.5]
        for p in projectiles:
            pp, pv = p.get("pos"), p.get("vel")
            if not pp or not pv:
                continue
            rel = [me[0] - pp[0], me[1] - pp[1], me[2] - pp[2]]
            vv = pv[0] ** 2 + pv[1] ** 2 + pv[2] ** 2
            if vv < 1:
                continue
            tc = (rel[0] * pv[0] + rel[1] * pv[1] + rel[2] * pv[2]) / vv
            if tc <= 0 or tc > 1.5:
                continue
            cx = pp[0] + pv[0] * tc - me[0]
            cy = pp[1] + pv[1] * tc - me[1]
            if math.hypot(cx, cy) < 5.0:
                perp = norm2([-pv[1], pv[0]])
                # step to the side the projectile is NOT passing on
                side = -1 if (perp[0] * cx + perp[1] * cy) > 0 else 1
                if p.get("kind") == "mortar":
                    side_v = norm2([-cx, -cy]) if math.hypot(cx, cy) > 0.5 else perp
                    out = add(out, side_v, 1.5)
                else:
                    out = add(out, perp, side)
        return out

    def unstick(self, t, pos, vel, move, speed):
        self.pos_hist.append((t, pos[0], pos[1]))
        self.pos_hist = [h for h in self.pos_hist if t - h[0] < 1.0]
        jump, boost = False, False
        if t < self.stuck_until and self.stuck_dir is not None:
            return self.stuck_dir, True, True
        wanted = math.hypot(*move) > 0.3 and speed > 0.4
        if wanted and len(self.pos_hist) >= 8:
            t0, x0, y0 = self.pos_hist[0]
            moved = math.hypot(pos[0] - x0, pos[1] - y0)
            if moved < 1.5 and t - t0 > 0.7:
                base = math.atan2(move[1], move[0])
                turn = self.rng.choice([1, -1]) * self.rng.uniform(1.2, 2.0)
                self.stuck_dir = unit_yaw(base + turn)
                self.stuck_until = t + 1.0
                self.pos_hist = []
                return self.stuck_dir, True, True
        return move, jump, boost

    # ------------------------------------------------------------ upgrades
    def choose_upgrade(self, offer):
        choices = offer.get("choices") or []
        prefs = [
            ("armor", 10), ("hp", 9), ("health", 9), ("regen", 8), ("repair", 8),
            ("shield", 8), ("speed", 6), ("energy", 5), ("jet", 4), ("sensor", 5),
            ("rate", 4), ("damage", 4), ("mag", 3), ("reload", 3),
        ]
        best_i, best_s = 0, -1
        for i, c in enumerate(choices):
            text = " ".join(str(c.get(k, "")) for k in ("id", "name", "desc")).lower()
            s = 0
            for key, val in prefs:
                if key in text:
                    s = max(s, val)
            if s > best_s:
                best_i, best_s = i, s
        return best_i

    # ------------------------------------------------------------ aim / fire
    def aim_and_fire(self, me, pos, target, weapons, t, obs):
        stats = me.get("stats") or {}
        muzzle_h = stats.get("muzzle_height", 7.0)
        eye_h = stats.get("eye_height", 8.0)
        cur = me.get("aim") or [0.0, 0.0]
        fire = [False, False]
        muzzle = [pos[0], pos[1], pos[2] + muzzle_h]

        if target is None:
            # look toward last damage, or a near sound, or scan
            if self.threat_bearing is not None and t < self.threat_until:
                return self.threat_bearing, 0.0, self.laser_release(weapons, fire)
            snd = None
            for s in obs.get("sounds") or []:
                if s.get("kind") in ("step", "boost", "gun", "laser_charge", "lightning", "jet") \
                        and s.get("dist") in ("near", "mid"):
                    snd = s
                    break
            if snd:
                return snd.get("bearing", cur[0]), 0.0, self.laser_release(weapons, fire)
            self.scan_phase += 0.12
            zone = obs.get("zone") or {}
            nc = zone.get("next_center", [pos[0], pos[1]])
            base = yaw_to(pos, nc) if dist2(pos, nc) > 5 else cur[0]
            return angn(base + 1.1 * math.sin(self.scan_phase)), 0.0, self.laser_release(weapons, fire)

        tp = target["pos"]
        tv = target.get("vel") or [0, 0, 0]
        is_crate = target.get("_crate", False)
        h = 1.0 if is_crate else (1.75 if target.get("kind") == "npc" else 4.5)
        aim_pt = [tp[0], tp[1], tp[2] + h]
        d = dist2(pos, aim_pt)

        # base aim with gun lead
        gun_speed = 252.0
        for w in weapons:
            if w and w.get("kind") == "gun":
                gun_speed = (w.get("stats") or {}).get("speed", 252.0)
        yaw, pitch = self.lead(muzzle, aim_pt, tv, gun_speed)

        yaw_err = abs(angn(cur[0] - yaw))
        tol = math.atan2(2.0, max(d, 1.0)) + 0.02

        for i, w in enumerate(weapons):
            if not w:
                continue
            kind = w.get("kind")
            ws = w.get("stats") or {}
            if kind == "gun":
                rng = ws.get("range", 230)
                if d < rng and yaw_err < tol and w.get("reload", 0) == 0 and w.get("ammo", 1) > 0:
                    fire[i] = True
            elif kind == "lightning":
                rng = ws.get("range", 45)
                cone = math.radians(ws.get("cone_deg", 30)) * 0.4
                if d < rng - 2 and yaw_err < cone and me.get("energy", 0) > 15:
                    fire[i] = True
            elif kind == "laser":
                rng = ws.get("range", 280)
                charge = w.get("charge", 0) or 0
                full = ws.get("charge", 0.75)
                if d < rng and w.get("cooldown", 0) <= 0:
                    if charge < full * 0.9:
                        fire[i] = True          # keep charging
                    elif yaw_err < tol:
                        fire[i] = False         # release = shoot
                    else:
                        fire[i] = True          # hold until on target
            elif kind == "mortar":
                if not is_crate and 30 < d < 225:
                    sol = self.mortar_pitch(pos, me, aim_pt, tv, ws)
                    if sol is not None:
                        m_yaw, m_pitch = sol
                        # mortar needs its own pitch; only fire if gun is not using the aim now
                        if not fire[0] or d > 120:
                            yaw, pitch = m_yaw, m_pitch
                            if abs(angn(cur[0] - m_yaw)) < 0.05 and abs(cur[1] - m_pitch) < 0.05:
                                fire[i] = True
        return yaw, pitch, fire

    def laser_release(self, weapons, fire):
        # with no target, never hold a charging laser (it would auto-fire anyway)
        return fire

    def lead(self, muzzle, tgt, tv, speed):
        p = list(tgt)
        for _ in range(3):
            dx, dy, dz = p[0] - muzzle[0], p[1] - muzzle[1], p[2] - muzzle[2]
            tt = math.sqrt(dx * dx + dy * dy + dz * dz) / max(speed, 1.0)
            p = [tgt[0] + tv[0] * tt, tgt[1] + tv[1] * tt, tgt[2] + tv[2] * tt * 0.5]
        dx, dy, dz = p[0] - muzzle[0], p[1] - muzzle[1], p[2] - muzzle[2]
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, math.hypot(dx, dy))
        return yaw, pitch

    def mortar_pitch(self, pos, me, tgt, tv, ws):
        v = ws.get("speed", 72.0)
        g = ws.get("gravity", cfg(self.config, "weapons.mortar.gravity", 22.0))
        stats = me.get("stats") or {}
        tube = [pos[0], pos[1], pos[2] + stats.get("height", 9.0) + 0.5]
        p = list(tgt)
        sol = None
        for _ in range(3):
            dx, dy = p[0] - tube[0], p[1] - tube[1]
            x = math.hypot(dx, dy)
            y = (p[2] - 4.0) - tube[2]   # aim at the feet area for splash
            disc = v ** 4 - g * (g * x * x + 2 * y * v * v)
            if disc < 0 or x < 1:
                return None
            th = math.atan2(v * v - math.sqrt(disc), g * x)
            tt = x / max(v * math.cos(th), 1.0)
            sol = (math.atan2(dy, dx), th)
            p = [tgt[0] + tv[0] * tt, tgt[1] + tv[1] * tt, tgt[2]]
        return sol


if __name__ == "__main__":
    mechio.run(Bot())
