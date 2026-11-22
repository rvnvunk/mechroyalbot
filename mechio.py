"""Minimal mechfight/1 stdin/stdout loop (standard library only).

- Reads JSON lines from stdin with raw os.read so we can see queued data.
- When several observations are queued, only the newest one is answered.
- Logs go to stderr. Only protocol messages go to stdout.
"""
import json
import math
import os
import select
import sys

_fd = sys.stdin.fileno()
_buf = b""
_eof = False


def log(*args):
    try:
        print(*args, file=sys.stderr, flush=True)
    except Exception:
        pass


def _parse_lines():
    global _buf
    out = []
    while b"\n" in _buf:
        line, _buf = _buf.split(b"\n", 1)
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception as e:
            log("bad line from server:", e)
    return out


def _read_chunk(timeout):
    """Read one chunk. timeout=None blocks. Returns False on EOF/no data."""
    global _buf, _eof
    if _eof:
        return False
    if timeout is not None:
        r, _, _ = select.select([_fd], [], [], timeout)
        if not r:
            return False
    chunk = os.read(_fd, 1 << 20)
    if not chunk:
        _eof = True
        return False
    _buf += chunk
    return True


def next_messages():
    """Block until at least one message, then drain everything queued.
    Returns a list (empty list means stdin closed)."""
    msgs = _parse_lines()
    while not msgs:
        if not _read_chunk(None):
            return _parse_lines()
        msgs = _parse_lines()
    while _read_chunk(0):
        msgs.extend(_parse_lines())
    return msgs


def _clean(x):
    if isinstance(x, float):
        if not math.isfinite(x):
            return 0.0
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def send(msg):
    sys.stdout.write(json.dumps(_clean(msg), separators=(",", ":")) + "\n")
    sys.stdout.flush()


def run(bot):
    """bot must have on_hello(msg) -> name, on_obs(obs) -> action dict, on_end(msg)."""
    while True:
        msgs = next_messages()
        if not msgs:
            return
        latest_obs = None
        for m in msgs:
            kind = m.get("type")
            if kind == "hello":
                try:
                    name = bot.on_hello(m)
                except Exception as e:
                    log("hello error:", repr(e))
                    name = "bot"
                send({"type": "ready", "name": str(name)[:32]})
            elif kind == "obs":
                latest_obs = m
            elif kind == "end":
                try:
                    bot.on_end(m)
                except Exception:
                    pass
                return
        if latest_obs is not None:
            tick = latest_obs.get("tick", 0)
            try:
                act = bot.on_obs(latest_obs) or {}
            except Exception as e:
                log("obs error:", repr(e))
                act = {}
            act["tick"] = tick
            send(act)
