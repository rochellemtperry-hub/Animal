from __future__ import annotations

from pathlib import Path

from trailcam_filter.species_reference import load_species_reference


def test_species_reference_bootstrap_and_normalization(tmp_path: Path) -> None:
    ref_dir = tmp_path / "reference"
    ref = load_species_reference(ref_dir)

    assert (ref_dir / "common_trailcam_species.csv").exists()
    assert (ref_dir / "species_aliases.csv").exists()
    assert (ref_dir / "north_burnett_species.csv").exists()

    assert "kangaroo" in ref.suggest()
    assert ref.normalize("roo") == "kangaroo"
    assert ref.normalize("goanna") == "lace monitor"
    assert ref.in_common("kangaroo")
    assert ref.in_regional("lace monitor")
