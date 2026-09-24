"""export() writes STEP resiliently.

build123d 0.11.0's high-level ``export_step`` (the ``STEPCAFControl_Writer`` path)
raises ``RuntimeError: Failed to write STEP file`` on many imported-STEP-derived
solids that 0.10.0 wrote fine — it hit ~38% of edits of imported parts.
``_write_step`` accepts high-level output only when single-solid files are flat,
then falls back to the basic ``STEPControl_Writer`` (geometry only). These tests
pin the happy path, the forced fallback (high-level writer made to raise, so it
runs on any version), the real reimported-solid regression, and the high-level
success path that can still write a one-component assembly.
"""

import pytest
from build123d import Color, import_step

from build123d_mcp.session import Session
from build123d_mcp.tools.execute import execute_code
from build123d_mcp.tools.export import export_file


def assert_no_assembly_links(step_path):
    assert "NEXT_ASSEMBLY_USAGE_OCCURRENCE" not in step_path.read_text(errors="ignore")


def assert_step_has_label_and_colour(step_path, label):
    text = step_path.read_text(errors="ignore")
    assert label in text
    assert "COLOUR_RGB" in text or "DRAUGHTING_PRE_DEFINED_COLOUR" in text


@pytest.fixture
def session():
    s = Session()
    s.execute("from build123d import *")
    return s


def test_export_step_writes_valid_solid(session, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    execute_code(session, "show(Box(10, 10, 10) - Cylinder(3, 12), 'part')")
    export_file(session, "out", "step", object_name="part")
    out = tmp_path / "out.step"
    assert out.exists()
    assert len(import_step(str(out)).solids()) == 1


def test_export_skipped_mesh_warning_mentions_refined_tessellation(session, tmp_path, monkeypatch):
    """If the isolated mesh gate times out, export's warning must name every mesh
    class that was skipped, including the refined face-tessellation probe."""
    import build123d_mcp.tools.validate as validate_tools

    monkeypatch.chdir(tmp_path)
    execute_code(session, "show(Box(10, 10, 10), 'part')")
    monkeypatch.setattr(validate_tools, "_run_mesh_gate_subprocess", lambda *a, **k: None)

    out = export_file(session, "out", "step", object_name="part")

    assert "too large to mesh-check" in out
    assert "refined face-tessellation" in out


def test_export_step_falls_back_when_high_level_writer_fails(session, tmp_path, monkeypatch):
    """When build123d's export_step raises (the 0.11.0 regression), export still
    produces a valid, reimportable STEP via the raw STEPControl_Writer."""
    monkeypatch.chdir(tmp_path)
    execute_code(session, "show(Box(10, 10, 10) - Cylinder(3, 12), 'part')")
    want_vol = session.objects["part"].volume

    def boom(*a, **k):
        raise RuntimeError("Failed to write STEP file")

    monkeypatch.setattr("build123d.export_step", boom)

    export_file(session, "out", "step", object_name="part")
    out = tmp_path / "out.step"
    assert out.exists()
    reimported = import_step(str(out))
    assert len(reimported.solids()) == 1
    # geometry round-trips: the fallback writer preserved the solid (volume), it
    # only drops CAF labels/colours.
    assert reimported.volume == pytest.approx(want_vol, rel=1e-3)


def test_export_step_fallback_preserves_all_solids(session, tmp_path, monkeypatch):
    """The raw-writer fallback must carry EVERY solid of a multi-solid compound,
    not silently drop bodies — the main silent-degradation risk of the fallback."""
    monkeypatch.chdir(tmp_path)
    execute_code(
        session,
        "show(Compound(children=[Box(10, 10, 10), Pos(20, 0, 0) * Box(10, 10, 10), "
        "Pos(40, 0, 0) * Box(10, 10, 10)]), 'asm')",
    )

    def boom(*a, **k):
        raise RuntimeError("Failed to write STEP file")

    monkeypatch.setattr("build123d.export_step", boom)

    export_file(session, "out", "step", object_name="asm")
    reimported = import_step(str(tmp_path / "out.step"))
    assert len(reimported.solids()) == 3  # all three bodies survived the fallback


def test_export_step_raises_clearly_when_both_writers_fail(session, tmp_path, monkeypatch):
    """If even the raw writer can't write, surface a clear combined error rather
    than a bare OCC failure."""
    monkeypatch.chdir(tmp_path)
    execute_code(session, "show(Box(10, 10, 10), 'part')")

    def boom(*a, **k):
        raise RuntimeError("Failed to write STEP file")

    monkeypatch.setattr("build123d.export_step", boom)

    class _Dead:
        def __init__(self, *a, **k): ...
        def Transfer(self, *a, **k): ...
        def Write(self, *a, **k):
            from OCP.IFSelect import IFSelect_ReturnStatus

            return IFSelect_ReturnStatus.IFSelect_RetFail

    monkeypatch.setattr("OCP.STEPControl.STEPControl_Writer", _Dead)
    with pytest.raises(RuntimeError, match="all failed"):
        export_file(session, "out", "step", object_name="part")


def test_write_step_handles_reimported_solid(tmp_path):
    """gumyr/build123d#1356: on build123d 0.11 ``export_step`` raises on a solid
    that came straight from ``import_step``; ``_write_step`` must still write a
    valid STEP (via the single-solid reconstruct retry) and must NOT mutate the
    caller's shape. Cross-version: on 0.10 the primary path already works. No
    monkeypatch — this exercises the real regression where the installed version
    has it.
    """
    from build123d import Box, export_step, import_step

    from build123d_mcp.tools.export import _write_step

    seed = tmp_path / "seed.step"
    export_step(Box(10, 10, 10), str(seed))
    s = import_step(str(seed))
    parent_before = s.parent

    out = tmp_path / "out.step"
    _write_step(s, str(out))  # must not raise on either build123d version

    assert s.parent is parent_before  # the fallback must not reparent the shape
    back = import_step(str(out))
    assert len(back.solids()) == 1
    assert back.volume == pytest.approx(1000.0, rel=1e-3)


def test_export_single_solid_is_not_written_as_assembly(tmp_path):
    """A single-solid export must be one STEP product, never a one-component
    assembly. On build123d 0.11 ``export_step`` raises on an import-derived solid
    (gumyr/build123d#1356); wrapping it in a ``Compound`` to get past that writes
    ``PRODUCT('COMPOUND')`` -> child + ``NEXT_ASSEMBLY_USAGE_OCCURRENCE``, which a
    CAD kernel opens as an assembly rather than a part. ``_write_step``
    reconstructs the solid instead, so the file stays a single product with the
    body name intact — on 0.10 the primary path already gives that, so this holds
    on either version.
    """
    from build123d import Box, export_step, import_step

    from build123d_mcp.tools.export import _write_step

    seed = tmp_path / "seed.step"
    export_step(Box(10, 10, 10), str(seed))
    s = import_step(str(seed))  # bare Solid, label 'COMPOUND' — the #1356 trigger
    s.label = "widget"
    s.color = Color(1, 0, 0)

    out = tmp_path / "out.step"
    _write_step(s, str(out))

    assert_no_assembly_links(out)  # single product, not an assembly
    assert_step_has_label_and_colour(out, "widget")
    back = import_step(str(out))
    assert len(back.solids()) == 1
    assert back.volume == pytest.approx(1000.0, rel=1e-3)


def test_export_fresh_single_solid_wrapper_is_not_written_as_assembly(
    session, tmp_path, monkeypatch
):
    """build123d's high-level writer can return success while still emitting a
    one-component assembly for fresh one-solid wrappers such as located
    primitives. That is not the #1356 exception path, so export() must inspect the
    written file and rewrite it flat.
    """

    monkeypatch.chdir(tmp_path)
    execute_code(session, "show(Pos(5, 0, 0) * Cylinder(2, 10), 'pin')")
    session.objects["pin"].color = Color(1, 0, 0)

    export_file(session, "pin", "step", object_name="pin")

    out = tmp_path / "pin.step"
    assert_no_assembly_links(out)
    assert_step_has_label_and_colour(out, "pin")
    back = import_step(str(out))
    bbox = back.bounding_box()
    assert len(back.solids()) == 1
    assert back.volume == pytest.approx(session.objects["pin"].volume, rel=1e-3)
    assert bbox.min.X == pytest.approx(3.0)
    assert bbox.max.X == pytest.approx(7.0)


def test_export_fresh_part_wrapper_is_not_written_as_assembly(session, tmp_path, monkeypatch):
    """Part wrappers can also take the high-level success path; keep the same
    flat-file invariant for them so the fix is not limited to primitives.
    """

    monkeypatch.chdir(tmp_path)
    execute_code(session, "show(Pos(5, 0, 0) * (Part() + Box(10, 10, 10)), 'part')")
    session.objects["part"].color = Color(1, 0, 0)

    export_file(session, "part", "step", object_name="part")

    out = tmp_path / "part.step"
    assert_no_assembly_links(out)
    assert_step_has_label_and_colour(out, "part")
    back = import_step(str(out))
    bbox = back.bounding_box()
    assert len(back.solids()) == 1
    assert back.volume == pytest.approx(1000.0, rel=1e-3)
    assert bbox.min.X == pytest.approx(0.0)
    assert bbox.max.X == pytest.approx(10.0)


def test_export_rewrites_successful_single_solid_assembly_export(session, tmp_path, monkeypatch):
    """Pin the invariant independent of build123d's current writer behavior:
    even if the high-level writer reports success with a one-component assembly,
    the final single-solid STEP must be flat.
    """

    from build123d import Compound
    from build123d import export_step as real_export_step

    monkeypatch.chdir(tmp_path)
    execute_code(session, "show(Box(10, 10, 10), 'part')")

    def write_one_component_assembly(shape, file_path, *args, **kwargs):
        return real_export_step(Compound(children=[shape]), file_path, *args, **kwargs)

    monkeypatch.setattr("build123d.export_step", write_one_component_assembly)

    export_file(session, "part", "step", object_name="part")

    out = tmp_path / "part.step"
    assert_no_assembly_links(out)
    back = import_step(str(out))
    assert len(back.solids()) == 1
    assert back.volume == pytest.approx(1000.0, rel=1e-3)


def test_export_multi_solid_compound_keeps_assembly_structure(session, tmp_path, monkeypatch):
    """The single-solid flat-file invariant must not flatten real multi-body
    assemblies.
    """

    monkeypatch.chdir(tmp_path)
    execute_code(
        session,
        "show(Compound(children=[Box(10, 10, 10), Pos(20, 0, 0) * Box(10, 10, 10)]), 'asm')",
    )

    export_file(session, "asm", "step", object_name="asm")

    out = tmp_path / "asm.step"
    assert "NEXT_ASSEMBLY_USAGE_OCCURRENCE" in out.read_text(errors="ignore")
    reimported = import_step(str(out))
    assert len(reimported.solids()) == 2
