"""Cross-section analysis for meshes.

`import_cad_file` on an STL yields a single `Face` - a shell with no solids
and no topology - so `find_holes`, `find_countersinks` and the rest of the
recogniser family return nothing on a downloaded mesh. That is the common case
when remixing a model from Printables or Thingiverse: you need the existing
mounting features in someone else's STL before you can design a part that
bolts to it.

Slicing works where topology does not. Intersect the tessellated triangles
with a plane, chain the segments into closed loops, and read features off the
loops: a loop enclosed by material is a passage through it.

Both tools tessellate via `Shape.tessellate`, so they work on solids too -
useful as a second opinion on `find_holes`, and as the only way to prove an
internal duct is enclosed rather than an open groove.
"""

import json
import math

_MAX_TRIANGLES = 50_000

_AXES = {"X": 0, "Y": 1, "Z": 2}


def _axis_index(axis: str) -> int:
    try:
        return _AXES[axis.upper()]
    except KeyError:
        raise ValueError(f"axis must be X, Y or Z, got {axis!r}") from None


def _triangles(shape, tolerance: float):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite number")
    verts, tris = shape.tessellate(tolerance)
    if len(tris) > _MAX_TRIANGLES:
        raise ValueError(
            f"mesh has {len(tris)} triangles (limit {_MAX_TRIANGLES}); "
            "raise `tolerance` to tessellate more coarsely"
        )
    pts = [(v.X, v.Y, v.Z) for v in verts]
    return [(pts[a], pts[b], pts[c]) for a, b, c in tris]


def _slice(tris, axis: int, value: float):
    """Segments where the plane axis=value cuts the mesh."""
    segs = []
    for tri in tris:
        d = [p[axis] - value for p in tri]
        hits = []
        for i in range(3):
            j = (i + 1) % 3
            if (d[i] > 0.0) != (d[j] > 0.0):
                f = d[i] / (d[i] - d[j])
                hits.append(tuple(tri[i][k] + f * (tri[j][k] - tri[i][k]) for k in range(3)))
        if len(hits) == 2:
            segs.append((hits[0], hits[1]))
    return segs


def _chain(segs, weld: float):
    """Join segments end-to-end into closed loops."""
    if not math.isfinite(weld) or weld <= 0:
        raise ValueError("weld must be a positive finite number")

    # Nearby endpoints can fall on opposite sides of a rounding boundary.
    # Search neighbouring cells and assign one vertex id to points within weld.
    buckets: dict[tuple[int, int, int], list[int]] = {}
    points: list[tuple[float, float, float]] = []

    def vertex(p):
        cell = tuple(math.floor(c / weld) for c in p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbour = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for index in buckets.get(neighbour, ()):
                        if sum((a - b) ** 2 for a, b in zip(p, points[index])) <= weld**2:
                            return index
        index = len(points)
        points.append(p)
        buckets.setdefault(cell, []).append(index)
        return index

    edges: list[tuple[int, int]] = []
    adj: dict[int, list[int]] = {}
    for a, b in segs:
        start, end = vertex(a), vertex(b)
        if start == end:
            continue
        edge = len(edges)
        edges.append((start, end))
        adj.setdefault(start, []).append(edge)
        adj.setdefault(end, []).append(edge)

    seen_edges: set[int] = set()
    loops = []
    for first_edge, (start, _) in enumerate(edges):
        if first_edge in seen_edges:
            continue
        path = [start]
        path_vertices = {start}
        cur, edge = start, first_edge
        while True:
            seen_edges.add(edge)
            a, b = edges[edge]
            nxt = b if cur == a else a
            if nxt == start:
                if len(path) >= 3:
                    loops.append([points[index] for index in path])
                break
            if nxt in path_vertices:
                break
            path.append(nxt)
            path_vertices.add(nxt)
            remaining = (candidate for candidate in adj[nxt] if candidate not in seen_edges)
            next_edge = next(remaining, None)
            if next_edge is None:
                break
            edge = next_edge
            cur = nxt
    return loops


def _to_2d(loop, axis: int):
    u, v = [k for k in range(3) if k != axis]
    return [(p[u], p[v]) for p in loop]


def _point_in_polygon(pt, poly) -> bool:
    """Ray-cast containment test."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if x < xint:
                inside = not inside
    return inside


def _enclosed_flags(loops, axis: int):
    """Which loops are enclosed BY another loop.

    Loop count alone is not containment. A groove cut clean across a bar
    splits the cross-section into two disjoint outlines; counting
    `loop_count - 1` then calls an open groove an enclosed passage, which is
    exactly the distinction these tools exist to make.
    """
    polys = [_to_2d(lp, axis) for lp in loops]
    flags = []
    for i, poly in enumerate(polys):
        if not poly:
            flags.append(False)
            continue
        probe = poly[0]
        flags.append(
            sum(
                _point_in_polygon(probe, other)
                for j, other in enumerate(polys)
                if j != i and len(other) >= 3
            )
            % 2
            == 1
        )
    return flags


def _loop_record(loop, axis: int) -> dict:
    u, v = [k for k in range(3) if k != axis]
    us = [p[u] for p in loop]
    vs = [p[v] for p in loop]
    return {
        "points": len(loop),
        "center": [round((min(us) + max(us)) / 2, 3), round((min(vs) + max(vs)) / 2, 3)],
        "size": [round(max(us) - min(us), 3), round(max(vs) - min(vs), 3)],
        "min": [round(min(us), 3), round(min(vs), 3)],
        "max": [round(max(us), 3), round(max(vs), 3)],
    }


def mesh_section(
    session,
    object_name: str = "",
    axis: str = "Z",
    position: float = 0.0,
    tolerance: float = 0.1,
    weld: float = 0.001,
) -> str:
    """Loops on one cross-section plane, largest first.

    Args:
        object_name: name from show()/import_cad_file (default: current shape)
        axis: "X", "Y" or "Z" - the plane's normal
        position: absolute world coordinate along that axis
        tolerance: tessellation tolerance (larger = coarser = faster)
        weld: point-merge distance when chaining segments into loops

    Returns:
        JSON {axis, position, loop_count, enclosed_passages, loops:[...]} where
        each loop carries points/center/size/min/max in the two axes that are
        not `axis`, plus `enclosed`.

    `enclosed_passages` counts loops at odd nesting depth - passages
    through the material at this height. Containment is tested, not inferred
    from the loop count: a groove cut clean across a bar splits the section
    into two disjoint outlines, and counting `loop_count - 1` would call that
    an enclosed passage. An open groove reads 0 here however deep it looks in
    a render.
    """
    from build123d_mcp.tools.measure import _resolve_shape

    shape = _resolve_shape(session, object_name)
    ax = _axis_index(axis)
    tris = _triangles(shape, tolerance)
    loops = _chain(_slice(tris, ax, position), weld)
    flags = _enclosed_flags(loops, ax)
    records = []
    for lp, enclosed in zip(loops, flags):
        rec = _loop_record(lp, ax)
        rec["enclosed"] = enclosed
        records.append(rec)
    records.sort(key=lambda r: r["size"][0] * r["size"][1], reverse=True)
    return json.dumps(
        {
            "axis": axis.upper(),
            "position": position,
            "loop_count": len(records),
            "enclosed_passages": sum(1 for r in records if r["enclosed"]),
            "loops": records,
        },
        indent=2,
    )


def mesh_holes(
    session,
    object_name: str = "",
    min_diameter: float = 2.0,
    max_diameter: float = 12.0,
    slices: int = 48,
    min_depth: float = 1.0,
    tolerance: float = 0.1,
    weld: float = 0.001,
) -> str:
    """Find fastener holes in a mesh by slicing it on all three axes.

    Args:
        object_name: name from show()/import_cad_file (default: current shape)
        min_diameter: keep loops at least this wide
        max_diameter: keep loops at most this wide. The defaults cover M2-M8
            clearance holes, counterbores and heat-set insert pockets.
        slices: sample planes per axis
        min_depth: drop features shallower than this. Chamfer rings and
            tessellation slivers read as very shallow holes.
        tolerance: tessellation tolerance
        weld: point-merge distance when chaining segments into loops

    Returns:
        JSON {count, holes:[{axis, diameter, location, span, through}]} where
        `axis` is the drilling direction, `location` the hole centre in world
        coordinates, and `span` how far it runs along `axis`.

    Read `span`: a hole running the full extent is a through hole, one
    appearing only near a face is a blind pocket - which is what a heat-set
    insert sits in. A hole shows up as a round loop only on slices normal to
    its own axis, so scanning a single axis finds a fraction of the part.

    This is the mesh counterpart to find_holes(), which needs real topology and
    returns nothing for an imported STL. It reports what the cross-sections
    show and does not classify counterbores, countersinks or thread forms.
    """
    from build123d_mcp.tools.measure import _resolve_shape

    if slices < 1:
        raise ValueError("slices must be at least 1")
    if not math.isfinite(weld) or weld <= 0:
        raise ValueError("weld must be a positive finite number")
    if not 0 < min_diameter <= max_diameter:
        raise ValueError("diameters must be positive and min_diameter <= max_diameter")
    if min_depth < 0:
        raise ValueError("min_depth must be nonnegative")

    shape = _resolve_shape(session, object_name)
    tris = _triangles(shape, tolerance)
    bb = shape.bounding_box()
    lo = (bb.min.X, bb.min.Y, bb.min.Z)
    hi = (bb.max.X, bb.max.Y, bb.max.Z)

    holes = []
    for ax in range(3):
        extent = hi[ax] - lo[ax]
        if extent <= 0:
            continue
        step = extent / slices
        found: dict = {}
        for value in _sample_positions(lo[ax], hi[ax], slices):
            for key in _keys_at(tris, ax, value, weld, min_diameter, max_diameter):
                found.setdefault(key, []).append(value)
        u, v = [k for k in range(3) if k != ax]
        for key, positions in sorted(found.items()):
            cu, cv, dia = key
            # One record per contiguous RUN, not per key. Two blind pockets
            # bored into opposite faces of a bar share a key - same centre in
            # the cross plane, same diameter - and merging them reports one
            # through hole where there are two pockets and solid material
            # between.
            spans = []
            for run in _runs(positions, step):
                # Sampling alone cannot give a span either: a shallow pocket
                # can fall between two sample planes. Walk out from a real hit
                # with a fine step to find where the feature actually stops.
                start, end = _refine_span(
                    tris,
                    ax,
                    key,
                    run,
                    step,
                    lo[ax],
                    hi[ax],
                    weld,
                    min_diameter,
                    max_diameter,
                )
                spans.append((start, end))

            # A missed slice can fragment a bore. Merge short gaps only when
            # the bore centreline has no surface crossing the gap: a pocket
            # floor is evidence of solid material, however thin the land is.
            barriers = _axis_intersections(tris, ax, cu, cv)
            for start, end in _merge_spans(spans, step, barriers, weld):
                if end - start < min_depth:
                    continue  # tessellation sliver, chamfer ring, not a hole
                wall_start, wall_end = _local_wall_span(
                    tris, ax, cu, cv, dia, start, end, (lo[ax], hi[ax])
                )
                edge_error = min(step * 0.6, (wall_end - wall_start) * 0.03)
                centre = [0.0, 0.0, 0.0]
                centre[u], centre[v] = cu, cv
                centre[ax] = round((start + end) / 2, 3)
                holes.append(
                    {
                        "axis": "XYZ"[ax],
                        "diameter": dia,
                        "location": [round(c, 3) for c in centre],
                        "span": [round(start, 3), round(end, 3)],
                        "depth": round(end - start, 3),
                        "through": (
                            start - wall_start <= edge_error and wall_end - end <= edge_error
                        ),
                    }
                )
    return json.dumps({"count": len(holes), "holes": holes}, indent=2)


def _sample_positions(lo: float, hi: float, slices: int):
    """Uniform sample planes, plus a fine sweep just inside each face.

    A uniform grid alone misses shallow blind pockets: a 4 mm insert pocket in
    the end of a 275 mm bar is thinner than one sample step and can fall
    entirely between two planes, so the bar reads as having a pocket at one end
    and nothing at the other. Bores open on faces, so sample near the faces.
    """
    extent = hi - lo
    step = extent / slices
    values = [lo + extent * i / slices for i in range(1, slices)]
    edge = min(step, extent * 0.06)
    for i in range(1, 13):
        values.append(lo + edge * i / 12.0)
        values.append(hi - edge * i / 12.0)
    return sorted(v for v in values if lo < v < hi)


def _merge_spans(spans, step, barriers, weld):
    """Join nearby runs unless a pocket floor crosses the bore centreline."""
    out = []
    for start, end in sorted(spans):
        gap_start = out[-1][1] if out else None
        no_floor = gap_start is not None and not any(
            gap_start + weld < p < start - weld for p in barriers
        )
        if out and start - gap_start <= step * 1.5 and no_floor:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _axis_intersections(tris, axis, u_value, v_value):
    """Coordinates where an axis-parallel line meets mesh triangles."""
    u, v = [k for k in range(3) if k != axis]
    hits = []
    for tri in tris:
        a, b, c = tri
        bu, bv = b[u] - a[u], b[v] - a[v]
        cu, cv = c[u] - a[u], c[v] - a[v]
        du, dv = u_value - a[u], v_value - a[v]
        det = bu * cv - bv * cu
        if abs(det) < 1e-12:
            continue
        wb = (du * cv - dv * cu) / det
        wc = (bu * dv - bv * du) / det
        wa = 1 - wb - wc
        if min(wa, wb, wc) >= -1e-9:
            hits.append(wa * a[axis] + wb * b[axis] + wc * c[axis])
    hits.sort()
    return [p for i, p in enumerate(hits) if i == 0 or p - hits[i - 1] > 1e-6]


def _local_wall_span(tris, axis, cu, cv, diameter, start, end, fallback):
    """Estimate local wall faces from rays just outside the bore perimeter."""
    radius = diameter * 0.65
    diagonal = radius / math.sqrt(2)
    offsets = [
        (radius, 0),
        (-radius, 0),
        (0, radius),
        (0, -radius),
        (diagonal, diagonal),
        (diagonal, -diagonal),
        (-diagonal, diagonal),
        (-diagonal, -diagonal),
    ]
    spans = []
    for du, dv in offsets:
        hits = _axis_intersections(tris, axis, cu + du, cv + dv)
        if len(hits) < 2:
            continue
        wall_start, wall_end = hits[0], hits[-1]
        if wall_start <= start + 0.1 and wall_end >= end - 0.1:
            spans.append((wall_start, wall_end))
    return min(spans, key=lambda span: span[1] - span[0], default=fallback)


def _runs(positions, step):
    """Split sorted sample positions into contiguous runs."""
    out, cur = [], [positions[0]]
    for p in positions[1:]:
        if p - cur[-1] <= step * 1.5:
            cur.append(p)
        else:
            out.append(cur)
            cur = [p]
    out.append(cur)
    return out


def _keys_at(tris, ax: int, value: float, weld: float, min_d: float, max_d: float):
    """Feature keys present on one slice: (center_u, center_v, diameter)."""
    keys = []
    loops = _chain(_slice(tris, ax, value), weld)
    for loop, enclosed in zip(loops, _enclosed_flags(loops, ax)):
        if not enclosed:
            continue  # an outline, or a disjoint piece of one - not a bore
        rec = _loop_record(loop, ax)
        dia = max(rec["size"])
        if min_d <= dia <= max_d:
            keys.append((round(rec["center"][0], 1), round(rec["center"][1], 1), round(dia, 1)))
    return keys


def _refine_span(tris, ax, key, run, step, lo, hi, weld, min_d, max_d):
    """True extent of one contiguous run, by fine stepping outwards."""
    fine = step / 16.0
    start = run[0]
    while start - fine >= lo and key in _keys_at(tris, ax, start - fine, weld, min_d, max_d):
        start -= fine
    end = run[-1]
    while end + fine <= hi and key in _keys_at(tris, ax, end + fine, weld, min_d, max_d):
        end += fine
    return start, end
