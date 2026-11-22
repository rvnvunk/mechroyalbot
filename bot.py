"""Entry point: python3 -u bot.py"""
import mechio
from brain import Brain
from world import World


def make_world(map_dir, cfg):
    mech = (cfg or {}).get("mech") or {}
    return World(map_dir, mech.get("radius", 2.5), mech.get("height", 9.0), mech.get("step_up", 1.2))


if __name__ == "__main__":
    mechio.run(Brain(make_world))
