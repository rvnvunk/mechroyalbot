"""Online combat model: learns, during the match, how well we hit and how hard enemies hit us.

Every number the brain uses for fight decisions comes from here. Each estimate starts at a
physics-based prior and moves toward what actually happens in this match (Bayesian averaging),
so the same logic adapts to whatever opponents we meet (practice house bots or other teams).

Distance bands (m): [0,35) [35,70) [70,110) [110,160) [160,+)
"""
import math

BAND_EDGES = (35.0, 70.0, 110.0, 160.0)
BAND_MID = (22.0, 52.0, 90.0, 135.0, 190.0)
NB = 5
KINDS = ("gun", "laser", "lightning", "mortar")


def band(d):
    for i, e in enumerate(BAND_EDGES):
        if d < e:
            return i
    return NB - 1


class CombatModel:
    SHOT_PRIOR_N = 8.0      # pseudo-shots behind each prior hit rate
    DMG_PRIOR_T = 5.0       # pseudo-seconds behind each prior damage rate

    def __init__(self, cfg):
        self.cfg = cfg or {}
        w = (self.cfg.get("weapons") or {})
        self.wcfg = {k: dict(w.get(k) or {}) for k in KINDS}
        # our accuracy (shots, hits) per weapon per band, mech targets only
        self.shots = {k: [0.0] * NB for k in KINDS}
        self.hits = {k: [0.0] * NB for k in KINDS}
        self.pending = {k: [] for k in KINDS}   # recent shots waiting for a hit event: (t, band)
        # damage we take per enemy weapon per band, and how long we were exposed to it
        self.exposure = {k: [0.0] * NB for k in KINDS}
        self.taken = {k: [0.0] * NB for k in KINDS}
        # where our fights happen (seconds spent shooting at mechs per band), split by target type
        self.fight_time = {"hunt": [0.0] * NB, "duel": [0.0] * NB}
        self.mode_time = {"hunt": 0.0, "duel": 0.0}
        self.shooter_time = 0.0
        self.shooter_sum = 0.0
        self.hunt_time = 0.0
        self.hunt_dmg = 0.0
        self.kills_passive = 0
        self.kills_hostile = 0

    # ------------------------------------------------------------ priors
    def prior_hit(self, kind, b):
        d = BAND_MID[b]
        if kind == "gun":
            return 0.9 if d <= 100 else max(0.4, 0.9 - (d - 100) * 0.004)
        if kind == "laser":
            return 0.85
        if kind == "lightning":
            return 0.9 if d < self.wcfg["lightning"].get("range", 45) else 0.0
        if kind == "mortar":
            return 0.35
        return 0.5

    def prior_rate(self, kind, b):
        """Expected damage per second an enemy deals us with one weapon at band b (no cubes)."""
        d = BAND_MID[b]
        st = self.wcfg.get(kind) or {}
        if kind == "gun":
            if d > st.get("range", 230):
                return 0.0
            cd = st.get("cooldown", 0.24)
            mag = st.get("mag", 12)
            rl = st.get("reload", 2.4)
            sustain = (mag * cd) / (mag * cd + rl)
            return st.get("damage", 45) / cd * sustain * 0.85 * math.exp(-d / 60.0)
        if kind == "laser":
            if d > st.get("range", 280):
                return 0.0
            return st.get("damage", 315) / (st.get("charge", 0.75) + st.get("cooldown", 1.65)) * 0.7
        if kind == "lightning":
            if d > st.get("range", 45):
                return 0.0
            return st.get("damage", 60) / st.get("cooldown", 0.25) * 0.75
        if kind == "mortar":
            if d < 25 or d > 230:
                return 0.0
            return (st.get("damage", 200) * 0.2 + st.get("splash", 160) * 0.3) / st.get("cooldown", 2.6)
        return 0.0

    # ------------------------------------------------------------ learning: our shots
    def on_shots(self, kind, d, n, t):
        b = band(d)
        self.shots[kind][b] += n
        pend = self.pending[kind]
        for _ in range(int(n)):
            pend.append((t, b))
        while pend and t - pend[0][0] > 1.5:
            pend.pop(0)

    def on_hit(self, kind, t):
        pend = self.pending.get(kind)
        if not pend:
            return
        while pend and t - pend[0][0] > 1.5:
            pend.pop(0)
        if pend:
            _, b = pend.pop(0)
            self.hits[kind][b] += 1.0

    def hit_p(self, kind, d):
        b = band(d)
        p0 = self.prior_hit(kind, b)
        if kind == "lightning" and d > self.wcfg["lightning"].get("range", 45):
            return 0.0
        n0 = self.SHOT_PRIOR_N
        return (self.hits[kind][b] + p0 * n0) / (self.shots[kind][b] + n0)

    # ------------------------------------------------------------ learning: damage we take
    def on_exposure(self, kinds, d, dt):
        b = band(d)
        for k in kinds:
            if k in self.exposure:
                self.exposure[k][b] += dt

    def on_damage(self, kind, d, amount, their_mult):
        if kind not in self.taken:
            return
        self.taken[kind][band(d)] += amount / max(their_mult, 0.5)

    def rate(self, kind, d):
        b = band(d)
        r0 = self.prior_rate(kind, b)
        if r0 <= 0 and self.exposure[kind][b] < 1.0:
            return 0.0
        t0 = self.DMG_PRIOR_T
        return (self.taken[kind][b] + r0 * t0) / (self.exposure[kind][b] + t0)

    # ------------------------------------------------------------ where we fight
    def on_fight_time(self, mode, d, dt):
        self.fight_time[mode][band(d)] += dt
        self.mode_time[mode] += dt

    def band_weights(self, mode):
        """Fraction of fight time per band for this kind of fight (prior + what we did so far)."""
        prior = {"hunt": (0.55, 0.3, 0.1, 0.05, 0.0), "duel": (0.15, 0.3, 0.3, 0.17, 0.08)}[mode]
        ft = self.fight_time[mode]
        tot = sum(ft)
        k = 15.0
        return [(ft[i] + prior[i] * k) / (tot + k) for i in range(NB)]

    def mode_share(self):
        """Share of fight time spent on passive targets (hunt) versus shooting enemies (duel)."""
        h, du = self.mode_time["hunt"], self.mode_time["duel"]
        k = 20.0
        return (h + 0.4 * k) / (h + du + k)

    # ------------------------------------------------------------ derived quantities
    def base_dps(self, kind, stats, energy=100.0):
        st = stats or self.wcfg.get(kind) or {}
        if kind == "gun":
            cd = st.get("cooldown", 0.24)
            mag = st.get("mag", 12)
            rl = st.get("reload", 2.4)
            return st.get("damage", 45) / cd * (mag * cd) / (mag * cd + rl)
        if kind == "laser":
            return st.get("damage", 315) / (st.get("charge", 0.75) + st.get("cooldown", 1.65))
        if kind == "lightning":
            if energy < 8:
                return 0.0
            return st.get("damage", 60) / st.get("cooldown", 0.25)
        if kind == "mortar":
            return (st.get("damage", 200) * 0.5 + st.get("splash", 160) * 0.5) / st.get("cooldown", 2.6)
        return 0.0

    def in_range(self, kind, d, stats=None):
        st = stats or self.wcfg.get(kind) or {}
        if kind == "gun":
            return d <= st.get("range", 230)
        if kind == "laser":
            return d <= st.get("range", 280)
        if kind == "lightning":
            return d <= st.get("range", 45) - 1.0
        if kind == "mortar":
            return 25 <= d <= 230
        return False

    def our_dps(self, weapons, d, mult, energy=100.0):
        tot = 0.0
        for w in weapons:
            if not w:
                continue
            k = w.get("kind") if isinstance(w, dict) else w
            st = (w.get("stats") if isinstance(w, dict) else None) or self.wcfg.get(k) or {}
            if not self.in_range(k, d, st):
                continue
            tot += self.base_dps(k, st, energy) * self.hit_p(k, d)
        return tot * mult

    def enemy_dps(self, kinds, d, mult, armor=0.0):
        tot = 0.0
        for k in kinds:
            tot += self.rate(k, d)
        return tot * mult * (1.0 - min(0.6, armor))

    def loadout_value(self, kinds, stats_by_kind, mult, enemy_kinds_seen):
        """Kills we expect per HP we lose with this loadout (higher is better).

        We die when our HP runs out, not when the clock does, so a loadout is worth the
        number of kills it buys per point of HP:
          - against an enemy that shoots back we pick our distance, and HP lost per kill is
            enemy_hp / (our_dps / their_dps) at the best distance (Lanchester);
          - against a passive target we lose HP only to background fire while we kill it,
            so HP lost per kill is background_dps * enemy_hp / our_dps.
        All rates are this match's measured values (with priors).
        """
        ek = enemy_kinds_seen or ("gun",)
        n_eff = self.shooters_avg()
        best_r = 0.0
        for b in range(NB):
            d = BAND_MID[b]
            ours = self._dps_kinds(kinds, stats_by_kind, d, mult)
            if ours <= 0:
                continue
            theirs = n_eff * sum(self.rate(k, d) for k in ek) / max(len(ek), 1)
            r = ours / (theirs + 5.0)
            if r > best_r:
                best_r = r
        hp_e = 1400.0
        hp_duel = hp_e / max(best_r, 0.05)
        hw = self.band_weights("hunt")
        d_hunt = sum(hw[b] * self._dps_kinds(kinds, stats_by_kind, BAND_MID[b], mult) for b in range(NB))
        hp_hunt = self.background_dps() * hp_e / max(d_hunt, 1.0)
        fp = self.passive_kill_share()
        hp_per_kill = (1.0 - fp) * hp_duel + fp * hp_hunt
        return 1.0 / max(hp_per_kill, 1.0)

    # ------------------------------------------------------------ fight context (measured)
    def on_shooters(self, n, dt):
        self.shooter_time += dt
        self.shooter_sum += n * dt

    def shooters_avg(self):
        k = 20.0
        return (self.shooter_sum + 1.6 * k) / (self.shooter_time + k)

    def on_hunt_exposure(self, dt, dmg):
        self.hunt_time += dt
        self.hunt_dmg += dmg

    def background_dps(self):
        k = 10.0
        return (self.hunt_dmg + 18.0 * k) / (self.hunt_time + k)

    def on_kill(self, passive):
        if passive:
            self.kills_passive += 1
        else:
            self.kills_hostile += 1

    def passive_kill_share(self):
        k = 4.0
        return (self.kills_passive + 0.5 * k) / (self.kills_passive + self.kills_hostile + k)

    def hp_per_kill(self, kinds, stats_by_kind, mult, enemy_kinds_seen):
        return 1.0 / max(self.loadout_value(kinds, stats_by_kind, mult, enemy_kinds_seen), 1e-6)

    def _dps_kinds(self, kinds, stats_by_kind, d, mult):
        tot = 0.0
        for k in kinds:
            if not k:
                continue
            st = stats_by_kind.get(k) or self.wcfg.get(k) or {}
            if self.in_range(k, d, st):
                tot += self.base_dps(k, st) * self.hit_p(k, d)
        return tot * mult

    # ------------------------------------------------------------ exchange projection
    @staticmethod
    def project(our_hp, foes):
        """Sequential fight: kill foes in order of time-to-kill while all living foes shoot us.

        foes: list of (ttk_seconds, incoming_dps). Returns (kills, hp_left, time).
        Fractional kills count partial progress on the foe we die fighting.
        """
        order = sorted(foes, key=lambda f: f[0])
        hp = our_hp
        t = 0.0
        kills = 0.0
        inc = sum(f[1] for f in order)
        for ttk, dps in order:
            if inc <= 1e-6:
                kills += 1.0
                t += ttk
                continue
            need = inc * ttk
            if need >= hp:
                frac = (hp / inc) / max(ttk, 1e-6)
                return kills + min(frac, 0.99), 0.0, t + hp / inc
            hp -= need
            t += ttk
            kills += 1.0
            inc -= dps
        return kills, hp, t
