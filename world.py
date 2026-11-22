"""Static map model built from /arena (world.json + terrain.i16).

Everything here is pure standard-library Python.

Raster (1 m, same grid as the terrain vertex grid):
    ground[c]  terrain height in cm
    cbase[c], ctop[c]  lowest base / highest top of prisms covering the cell, in cm
                       (32767 / -32768 when no prism)
    multi[c]   list of disjoint (base, top) cm intervals when a cell has more than one
Nav grid (2 m): walk[n] = 1 when a mech (radius 2.5) can stand there at ground level.
"""
import array
import heapq
import json
import math
import os
import sys
import time

NOB = 32767
NOT = -32768


class World:
    def __init__(self, map_dir, mech_radius=2.5, mech_height=9.0, step_up=1.2):
        t0 = time.time()
        wj = json.load(open(os.path.join(map_dir, "v2", "world.json")))
        T = wj["terrain"]
        self.minX = float(T["minX"])
        self.minY = float(T["minY"])
        self.nx = int(T["nx"])
        self.ny = int(T["ny"])
        self.step = float(T.get("step", 1.0))
        b = wj.get("bounds") or {}
        self.bminX = b.get("minX", self.minX)
        self.bmaxX = b.get("maxX", self.minX + self.nx - 1)
        self.bminY = b.get("minY", self.minY)
        self.bmaxY = b.get("maxY", self.minY + self.ny - 1)
        tf = os.path.join(map_dir, "v2", T.get("file", "terrain.i16"))
        g = array.array("h")
        with open(tf, "rb") as f:
            g.frombytes(f.read())
        if sys.byteorder == "big":
            g.byteswap()
        self.ground = g
        N = self.nx * self.ny
        self.N = N
        self._raster_prisms(wj.get("prisms") or [])
        self._build_trees(wj.get("trees") or {})
        self._build_nav(mech_radius, mech_height, step_up)
        self.build_s = time.time() - t0

    # ------------------------------------------------------------ raster
    def _raster_prisms(self, prisms):
        nx, ny, minX, minY = self.nx, self.ny, self.minX, self.minY
        cells = {}
        self.prisms = []
        for p in prisms:
            ring = p.get("ring") or []
            if len(ring) < 6:
                continue
            base = float(p.get("base", 0.0))
            top = float(p.get("top", 0.0))
            polys = [ring] + [h for h in (p.get("holes") or []) if len(h) >= 6]
            edges = []
            for poly in polys:
                m = len(poly) // 2
                for k in range(m):
                    x0, y0 = poly[2 * k], poly[2 * k + 1]
                    x1, y1 = poly[2 * ((k + 1) % m)], poly[2 * ((k + 1) % m) + 1]
                    if y0 != y1:
                        edges.append((x0, y0, x1, y1))
            ys = ring[1::2]
            xs = ring[0::2]
            self.prisms.append((min(xs), min(ys), max(xs), max(ys), base, top, p.get("kind", "building")))
            j0 = max(0, int(math.ceil(min(ys) - minY)))
            j1 = min(ny - 1, int(math.floor(max(ys) - minY)))
            bc = int(round(base * 100))
            tc = int(round(top * 100))
            for j in range(j0, j1 + 1):
                y = minY + j
                xi = []
                for (x0, y0, x1, y1) in edges:
                    if (y0 <= y < y1) or (y1 <= y < y0):
                        xi.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
                if len(xi) < 2:
                    continue
                xi.sort()
                row = j * nx
                for k in range(0, len(xi) - 1, 2):
                    ia = max(0, int(math.ceil(xi[k] - minX)))
                    ib = min(nx - 1, int(math.floor(xi[k + 1] - minX)))
                    for i in range(ia, ib + 1):
                        c = row + i
                        lst = cells.get(c)
                        if lst is None:
                            cells[c] = [(bc, tc)]
                        else:
                            lst.append((bc, tc))
        N = self.N
        cbase = array.array("h", [NOB]) * N
        ctop = array.array("h", [NOT]) * N
        multi = {}
        for c, lst in cells.items():
            if len(lst) == 1:
                cbase[c], ctop[c] = lst[0]
                continue
            lst.sort()
            merged = [list(lst[0])]
            for b_, t_ in lst[1:]:
                if b_ <= merged[-1][1] + 30:
                    if t_ > merged[-1][1]:
                        merged[-1][1] = t_
                else:
                    merged.append([b_, t_])
            cbase[c] = merged[0][0]
            ctop[c] = max(m[1] for m in merged)
            if len(merged) > 1:
                multi[c] = [tuple(m) for m in merged]
        self.cbase = cbase
        self.ctop = ctop
        self.multi = multi

    def _build_trees(self, trees):
        # trunks only (they block sight, not shots); crowns ignored here
        self.trees = []
        f = trees.get("fields") or []
        d = trees.get("data") or []
        if not f:
            return
        nf = len(f)
        ix, iy = f.index("x"), f.index("y")
        ih = f.index("height") if "height" in f else None
        icr = f.index("crownR") if "crownR" in f else None
        icb = f.index("crownBase") if "crownBase" in f else None
        icx = f.index("crownX") if "crownX" in f else None
        icy = f.index("crownY") if "crownY" in f else None
        iz = f.index("z") if "z" in f else None
        for k in range(0, len(d) - nf + 1, nf):
            row = d[k:k + nf]
            x, y = row[ix], row[iy]
            cx = row[icx] if icx is not None and row[icx] is not None else x
            cy = row[icy] if icy is not None and row[icy] is not None else y
            self.trees.append((x, y, row[iz] if iz is not None else 0.0,
                               row[ih] if ih is not None else 10.0,
                               row[icr] if icr is not None else 3.0,
                               row[icb] if icb is not None else 3.0, cx, cy))

        self._build_canopy()

    def _build_canopy(self):
        """Raster of tree crowns (they block sight in the real game): per 1 m cell the lowest
        and highest crown height in cm, crowns modelled as ellipsoids."""
        N = self.N
        kb = array.array("h", [NOB]) * N
        kt = array.array("h", [NOT]) * N
        nx, ny, minX, minY = self.nx, self.ny, self.minX, self.minY
        ncell = 0
        for (x, y, z, h, cr, cbz, cx, cy) in self.trees:
            if h is None or cr is None or cr <= 0.3:
                continue
            if z is None or z == 0.0:
                z = self.ground_z(x, y)
            cbz = cbz if cbz is not None else 3.0
            if h <= cbz + 0.5:
                continue
            zc = z + 0.5 * (cbz + h)
            hh = 0.5 * (h - cbz)
            R = cr
            i0 = int(cx - R - minX); i1 = int(cx + R - minX) + 1
            j0 = int(cy - R - minY); j1 = int(cy + R - minY) + 1
            for j in range(max(0, j0), min(ny, j1 + 1)):
                py = minY + j
                for i in range(max(0, i0), min(nx, i1 + 1)):
                    px = minX + i
                    rho2 = ((px - cx) ** 2 + (py - cy) ** 2) / (R * R)
                    if rho2 >= 1.0:
                        continue
                    span = hh * math.sqrt(1.0 - rho2)
                    lo = int((zc - span) * 100.0)
                    hi = int((zc + span) * 100.0)
                    c = j * nx + i
                    if lo < kb[c]:
                        kb[c] = max(-32000, lo)
                    if hi > kt[c]:
                        kt[c] = min(32000, hi)
                    ncell += 1
        self.kb = kb
        self.kt = kt
        self.canopy_cells = ncell

    def vis(self, x0, y0, z0, x1, y1, z1, step=1.0):
        """Line of sight for vision: terrain, prisms and tree crowns."""
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        hd = math.hypot(dx, dy)
        n = int(max(hd, abs(dz)) / step) + 1
        if n < 2:
            return True
        minX = self.minX - 0.5
        minY = self.minY - 0.5
        nx, ny = self.nx, self.ny
        g, cb, ct, multi, kb, kt = self.ground, self.cbase, self.ctop, self.multi, self.kb, self.kt
        inv = 1.0 / n
        for k in range(1, n):
            t = k * inv
            i = int(x0 + dx * t - minX)
            j = int(y0 + dy * t - minY)
            if i < 0 or j < 0 or i >= nx or j >= ny:
                return False
            c = j * nx + i
            zc = (z0 + dz * t) * 100.0
            if zc < g[c]:
                return False
            if kb[c] <= zc <= kt[c]:
                return False
            if cb[c] <= zc <= ct[c]:
                m = multi.get(c)
                if m is None:
                    return False
                for b_, t_ in m:
                    if b_ <= zc <= t_:
                        return False
        return True

    # ------------------------------------------------------------ queries
    def cell(self, x, y):
        i = int(x - self.minX + 0.5)
        j = int(y - self.minY + 0.5)
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return -1
        return j * self.nx + i

    def ground_z(self, x, y):
        """Bilinear terrain height in meters (ignores prisms)."""
        fx = x - self.minX
        fy = y - self.minY
        i = int(fx)
        j = int(fy)
        if i < 0 or j < 0 or i >= self.nx - 1 or j >= self.ny - 1:
            c = self.cell(x, y)
            return self.ground[c] / 100.0 if c >= 0 else 0.0
        tx = fx - i
        ty = fy - j
        g = self.ground
        nx = self.nx
        c = j * nx + i
        a = g[c] * (1 - tx) + g[c + 1] * tx
        b = g[c + nx] * (1 - tx) + g[c + nx + 1] * tx
        return (a * (1 - ty) + b * ty) / 100.0

    def solid(self, x, y, z):
        c = self.cell(x, y)
        if c < 0:
            return True
        zc = z * 100.0
        if zc < self.ground[c]:
            return True
        if self.cbase[c] <= zc <= self.ctop[c]:
            m = self.multi.get(c)
            if m is None:
                return True
            for b_, t_ in m:
                if b_ <= zc <= t_:
                    return True
        return False

    def los(self, x0, y0, z0, x1, y1, z1, step=1.0, skip_end=0.0):
        """True when the straight segment is clear of terrain and prisms."""
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        hd = math.hypot(dx, dy)
        n = int(max(hd, abs(dz)) / step) + 1
        if n < 2:
            return True
        minX = self.minX - 0.5
        minY = self.minY - 0.5
        nx, ny = self.nx, self.ny
        g, cb, ct, multi = self.ground, self.cbase, self.ctop, self.multi
        end = n - int(skip_end / step) if skip_end > 0 else n
        inv = 1.0 / n
        for k in range(1, end):
            t = k * inv
            i = int(x0 + dx * t - minX)
            j = int(y0 + dy * t - minY)
            if i < 0 or j < 0 or i >= nx or j >= ny:
                return False
            c = j * nx + i
            zc = (z0 + dz * t) * 100.0
            if zc < g[c]:
                return False
            if cb[c] <= zc <= ct[c]:
                m = multi.get(c)
                if m is None:
                    return False
                for b_, t_ in m:
                    if b_ <= zc <= t_:
                        return False
        return True

    def blocked_dist(self, x0, y0, z0, x1, y1, z1, step=1.0):
        """Distance along the segment to the first solid sample (None when clear)."""
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = int(max(math.hypot(dx, dy), abs(dz)) / step) + 1
        for k in range(1, n + 1):
            t = k / n
            if self.solid(x0 + dx * t, y0 + dy * t, z0 + dz * t):
                return L * t
        return None

    # ------------------------------------------------------------ nav
    def _build_nav(self, R, H, step_up):
        nx, ny, N = self.nx, self.ny, self.N
        g, cb, ct, multi = self.ground, self.cbase, self.ctop, self.multi
        blk = bytearray(N)
        Hc = int(H * 100)
        Sc = int(step_up * 100)
        for c in range(N):
            if ct[c] == NOT:
                continue
            gc = g[c]
            m = multi.get(c)
            ivs = m if m is not None else ((cb[c], ct[c]),)
            for b_, t_ in ivs:
                if b_ < gc + Hc and t_ > gc + Sc:
                    blk[c] = 1
                    break
        # steep terrain (cannot walk up slopes over ~50 degrees)
        steep = bytearray(N)
        lim = 119
        for j in range(ny - 1):
            row = j * nx
            for i in range(nx - 1):
                c = row + i
                h = g[c]
                if abs(g[c + 1] - h) > lim or abs(g[c + nx] - h) > lim:
                    steep[c] = 1
        self.blk = blk
        self.steep = steep
        # nav grid 2 m
        NX, NY = nx // 2, ny // 2
        self.NX, self.NY = NX, NY
        walk = bytearray(b"\x01") * (NX * NY)
        # boundary blocked cells dilate by R
        Rr = R + 0.3
        rr2 = Rr * Rr
        for j in range(ny):
            row = j * nx
            for i in range(nx):
                c = row + i
                if not blk[c]:
                    continue
                # nav cells whose center lies inside this blocked cell neighbourhood
                boundary = (i == 0 or j == 0 or i == nx - 1 or j == ny - 1 or not blk[c - 1]
                            or not blk[c + 1] or not blk[c - nx] or not blk[c + nx])
                if not boundary:
                    if (i & 1) == 0 and (j & 1) == 0:
                        I, J = i >> 1, j >> 1
                        if I < NX and J < NY:
                            walk[J * NX + I] = 0
                    continue
                I0 = max(0, int(math.ceil((i - Rr) / 2)))
                I1 = min(NX - 1, int(math.floor((i + Rr) / 2)))
                J0 = max(0, int(math.ceil((j - Rr) / 2)))
                J1 = min(NY - 1, int(math.floor((j + Rr) / 2)))
                for J in range(J0, J1 + 1):
                    dj = 2 * J - j
                    for I in range(I0, I1 + 1):
                        di = 2 * I - i
                        if di * di + dj * dj <= rr2:
                            walk[J * NX + I] = 0
        # steep cells: block only their own nav cell
        for J in range(NY):
            for I in range(NX):
                n = J * NX + I
                if not walk[n]:
                    continue
                i, j = 2 * I, 2 * J
                c = j * nx + i
                if steep[c] or (c + 1 < N and steep[c + 1]) or (c + nx < N and steep[c + nx]):
                    walk[n] = 0
        # map edges
        mX0 = self.bminX + R + 0.5
        mX1 = self.bmaxX - R - 0.5
        mY0 = self.bminY + R + 0.5
        mY1 = self.bmaxY - R - 0.5
        for J in range(NY):
            y = self.minY + 2 * J
            for I in range(NX):
                x = self.minX + 2 * I
                if x < mX0 or x > mX1 or y < mY0 or y > mY1:
                    walk[J * NX + I] = 0
        self.walk = walk
        self._label_components()
        self._build_anchors()
        self._add_tree_anchors()

    def _add_tree_anchors(self):
        """Spots under tree crowns: hidden from sight when crowns block vision."""
        self.tree_spots = []
        if not getattr(self, "trees", None):
            return
        B = self.anchor_B
        for (x, y, z, h, cr, cbz, cx, cy) in self.trees:
            if cr is None or cr < 2.2 or h is None or h < 7.0:
                continue
            if not self.walkable(cx, cy):
                continue
            self.tree_spots.append((cx, cy))
            k = (int((cx - self.minX) // B), int((cy - self.minY) // B))
            self.anchor_buckets.setdefault(k, []).append((cx, cy))

    def _label_components(self):
        NX, NY, walk = self.NX, self.NY, self.walk
        comp = array.array("i", [-1]) * (NX * NY)
        sizes = []
        cid = 0
        for s in range(NX * NY):
            if not walk[s] or comp[s] >= 0:
                continue
            stack = [s]
            comp[s] = cid
            cnt = 0
            while stack:
                n = stack.pop()
                cnt += 1
                I = n % NX
                J = n // NX
                if I > 0 and walk[n - 1] and comp[n - 1] < 0:
                    comp[n - 1] = cid; stack.append(n - 1)
                if I < NX - 1 and walk[n + 1] and comp[n + 1] < 0:
                    comp[n + 1] = cid; stack.append(n + 1)
                if J > 0 and walk[n - NX] and comp[n - NX] < 0:
                    comp[n - NX] = cid; stack.append(n - NX)
                if J < NY - 1 and walk[n + NX] and comp[n + NX] < 0:
                    comp[n + NX] = cid; stack.append(n + NX)
            sizes.append(cnt)
            cid += 1
        self.comp = comp
        self.comp_sizes = sizes
        self.main_comp = max(range(len(sizes)), key=lambda k: sizes[k]) if sizes else -1

    def _build_anchors(self):
        """Walkable nav cells right next to walls (good cover spots), bucketed by 25 m."""
        NX, NY, walk = self.NX, self.NY, self.walk
        comp, mc = self.comp, self.main_comp
        anchors = []
        for J in range(2, NY - 2, 2):
            for I in range(2, NX - 2, 2):
                n = J * NX + I
                if not walk[n] or comp[n] != mc:
                    continue
                # a blocked nav cell within 2 cells (4 m) means a wall is right there
                near = False
                for dJ in (-2, -1, 0, 1, 2):
                    base = n + dJ * NX
                    for dI in (-2, -1, 0, 1, 2):
                        if not walk[base + dI]:
                            near = True
                            break
                    if near:
                        break
                if near:
                    anchors.append(n)
        self.anchors = anchors
        B = 25.0
        buckets = {}
        for n in anchors:
            x, y = self.nav_pos(n)
            k = (int((x - self.minX) // B), int((y - self.minY) // B))
            buckets.setdefault(k, []).append((x, y))
        self.anchor_buckets = buckets
        self.anchor_B = B

    def anchors_near(self, x, y, r):
        B = self.anchor_B
        k0 = int((x - self.minX) // B)
        k1 = int((y - self.minY) // B)
        rr = int(r // B) + 1
        out = []
        r2 = r * r
        for a in range(k0 - rr, k0 + rr + 1):
            for b in range(k1 - rr, k1 + rr + 1):
                for (px, py) in self.anchor_buckets.get((a, b), ()):
                    dx, dy = px - x, py - y
                    d = dx * dx + dy * dy
                    if d <= r2:
                        out.append((d, px, py))
        out.sort()
        return out

    def nav_index(self, x, y):
        I = int((x - self.minX) / 2.0 + 0.5)
        J = int((y - self.minY) / 2.0 + 0.5)
        if I < 0 or J < 0 or I >= self.NX or J >= self.NY:
            return -1
        return J * self.NX + I

    def nav_pos(self, n):
        return (self.minX + 2.0 * (n % self.NX), self.minY + 2.0 * (n // self.NX))

    def walkable(self, x, y):
        n = self.nav_index(x, y)
        return n >= 0 and self.walk[n] == 1

    def nearest_walkable(self, x, y, max_r=8, comp=None):
        n0 = self.nav_index(x, y)
        if n0 >= 0 and self.walk[n0] and (comp is None or self.comp[n0] == comp):
            return n0
        NX, NY = self.NX, self.NY
        I0 = int((x - self.minX) / 2.0 + 0.5)
        J0 = int((y - self.minY) / 2.0 + 0.5)
        best = None
        for r in range(1, max_r + 1):
            for dJ in range(-r, r + 1):
                for dI in (-r, r) if abs(dJ) != r else range(-r, r + 1):
                    I, J = I0 + dI, J0 + dJ
                    if 0 <= I < NX and 0 <= J < NY:
                        n = J * NX + I
                        if self.walk[n] and (comp is None or self.comp[n] == comp):
                            d = dI * dI + dJ * dJ
                            if best is None or d < best[0]:
                                best = (d, n)
            if best is not None:
                return best[1]
        return -1

    def straight_walkable(self, x0, y0, x1, y1, step=1.0):
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        n = int(L / step) + 1
        walk = self.walk
        for k in range(1, n + 1):
            t = k / n
            m = self.nav_index(x0 + dx * t, y0 + dy * t)
            if m < 0 or not walk[m]:
                return False
        return True


class AStar:
    """Resumable A* on the 2 m nav grid."""

    def __init__(self, world):
        self.w = world
        self.reset()

    def reset(self):
        self.start = -1
        self.goal = -1
        self.open = None
        self.g = None
        self.came = None
        self.done = False
        self.path = None
        self.expanded = 0

    def begin(self, start, goal):
        self.reset()
        self.start, self.goal = start, goal
        self.g = {start: 0.0}
        self.came = {start: -1}
        self.open = [(self.h(start), 0.0, start)]
        self.closed = set()
        self.best = (self.h(start), start)

    def h(self, n):
        NX = self.w.NX
        dx = abs(n % NX - self.goal % NX)
        dy = abs(n // NX - self.goal // NX)
        return (dx + dy) + (1.41421356 - 2) * min(dx, dy)

    def step(self, budget_s):
        if self.done or self.open is None:
            return self.done
        w = self.w
        NX, NY = w.NX, w.NY
        walk = w.walk
        g, came, openh, closed = self.g, self.came, self.open, self.closed
        goal = self.goal
        tend = time.perf_counter() + budget_s
        it = 0
        D = 1.41421356
        while openh:
            it += 1
            if (it & 127) == 0 and time.perf_counter() > tend:
                return False
            f, gc, n = heapq.heappop(openh)
            if n in closed:
                continue
            closed.add(n)
            self.expanded += 1
            if n == goal:
                self._finish(goal)
                return True
            hn = f - gc
            if hn < self.best[0]:
                self.best = (hn, n)
            I = n % NX
            J = n // NX
            l = I > 0 and walk[n - 1]
            r = I < NX - 1 and walk[n + 1]
            d = J > 0 and walk[n - NX]
            u = J < NY - 1 and walk[n + NX]
            nb = []
            if l: nb.append((n - 1, 1.0))
            if r: nb.append((n + 1, 1.0))
            if d: nb.append((n - NX, 1.0))
            if u: nb.append((n + NX, 1.0))
            if l and d and walk[n - 1 - NX]: nb.append((n - 1 - NX, D))
            if r and d and walk[n + 1 - NX]: nb.append((n + 1 - NX, D))
            if l and u and walk[n - 1 + NX]: nb.append((n - 1 + NX, D))
            if r and u and walk[n + 1 + NX]: nb.append((n + 1 + NX, D))
            for m, cst in nb:
                ng = gc + cst
                if ng < g.get(m, 1e18):
                    g[m] = ng
                    came[m] = n
                    heapq.heappush(openh, (ng + self.h(m), ng, m))
        # goal unreachable: path to the closest node found
        self._finish(self.best[1])
        return True

    def _finish(self, end):
        path = []
        n = end
        while n != -1:
            path.append(n)
            n = self.came.get(n, -1)
        path.reverse()
        self.path = path
        self.done = True
        self.open = None
