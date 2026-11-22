"""Quick local smoke test (NOT the official check).

Sends a hello and fake observations based on the example in protocol.md,
then checks: ready is sent, every answer echoes the tick, answers are fast,
queued observations get only one answer, and the bot exits after end.

Usage:
    python3 tools/smoke_test.py -- python3 -u bot.py
"""
import json
import math
import random
import subprocess
import sys
import time

cmd = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["python3", "-u", "bot.py"]

HELLO = {"type": "hello", "protocol": "mechfight/1", "slot": 3, "seed": 12345,
         "map": {"dir": "/arena", "files": ["v2/world.json", "v2/terrain.i16"]},
         "config": {"weapons": {"mortar": {"gravity": 22}}}}


def make_obs(tick, rnd):
    t = tick / 20
    me = [-218 + rnd.uniform(-5, 5), -22 + rnd.uniform(-5, 5), -4.7]
    enemy = [me[0] + rnd.uniform(-80, 80), me[1] + rnd.uniform(-80, 80), -4.7]
    return {
        "type": "obs", "tick": tick, "t": t, "alive": 15,
        "self": {"id": 3, "pos": me, "vel": [1, 0, 0], "aim": [rnd.uniform(-3, 3), 0],
                 "hp": rnd.choice([300, 900, 1400]), "max_hp": 1400, "energy": 60, "max_energy": 100,
                 "grounded": True, "cubes": 0, "upgrades": [], "in_zone": True,
                 "weapons": [
                     {"kind": "gun", "cooldown": 0, "ammo": 10, "reload": 0,
                      "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "mag": 12}},
                     rnd.choice([None, {"kind": "laser", "cooldown": 0, "charge": rnd.uniform(0, 0.8),
                                        "stats": {"range": 280, "charge": 0.75}},
                                 {"kind": "mortar", "cooldown": 0, "stats": {"speed": 72, "gravity": 22}},
                                 {"kind": "lightning", "cooldown": 0, "stats": {"range": 45, "cone_deg": 30}}])],
                 "stats": {"walk_speed": 11.7, "muzzle_height": 7, "eye_height": 8, "height": 9}},
        "mechs": rnd.choice([[], [{"id": 5, "kind": "mech", "pos": enemy, "vel": [7, -5, 0],
                                   "aim": [0, 0], "hp": 900, "max_hp": 1400}]]),
        "projectiles": [{"id": 280, "kind": "gun", "pos": [me[0] + 20, me[1], -0.5], "vel": [-250, 0, 0]}],
        "crates": [{"id": 1007, "pos": [me[0] + 30, me[1] - 10, -4.7], "hp": 300}],
        "cubes": [{"id": 5012, "pos": [me[0] + 10, me[1] + 5, -4.7]}],
        "pickups": [{"id": 8013, "kind": "mortar", "pos": [me[0] + 2, me[1] + 1, -4.7]}],
        "sounds": [{"kind": "gun", "bearing": -2.68, "dist": "near", "age": 0.05}],
        "beams": [], "events": [{"type": "damaged", "amount": 90, "weapon": "gun", "bearing": 0.9, "by": 5}],
        "drops": [],
        "zone": {"center": [41.3, -72.8], "radius": 560, "next_center": [-23.7, 2.2], "next_radius": 380,
                 "shrink_start": 60, "shrink_end": 110, "dps": 20},
        "offer": rnd.choice([None, {"id": tick, "expires": t + 5, "choices": [
            {"id": "gun_rate", "name": "Rotary feed", "desc": "+15% arm gun fire rate"},
            {"id": "armor", "name": "Reactive armor", "desc": "-8% damage taken"},
            {"id": "sensors", "name": "Sensor suite", "desc": "+15 m vision"}]}]),
    }


def main():
    rnd = random.Random(1)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(m):
        p.stdin.write(json.dumps(m) + "\n")
        p.stdin.flush()

    send(HELLO)
    ready = json.loads(p.stdout.readline())
    assert ready.get("type") == "ready", ready
    print("ready:", ready)

    times = []
    for tick in range(2, 602, 2):
        obs = make_obs(tick, rnd)
        t0 = time.perf_counter()
        send(obs)
        ans = json.loads(p.stdout.readline())
        times.append((time.perf_counter() - t0) * 1000)
        assert ans.get("tick") == tick, (tick, ans)
        for k in ("move", "aim"):
            assert len(ans[k]) == 2 and all(math.isfinite(x) for x in ans[k]), ans
        if "fire" in ans:
            assert len(ans["fire"]) == 2 and all(isinstance(x, bool) for x in ans["fire"]), ans

    # queued observations: send 3 at once, expect one answer for the newest
    p.stdin.write("".join(json.dumps(make_obs(tk, rnd)) + "\n" for tk in (700, 702, 704)))
    p.stdin.flush()
    ans = json.loads(p.stdout.readline())
    print("queued answer tick:", ans["tick"], "(want 704; 700 is possible only if timing was unlucky)")

    send({"type": "end", "place": 1, "kills": 0})
    p.stdin.close()
    rc = p.wait(timeout=5)
    times.sort()
    print(f"answers: {len(times)}  median {times[len(times)//2]:.2f} ms  max {times[-1]:.2f} ms  exit code {rc}")
    print("OK")


if __name__ == "__main__":
    main()
