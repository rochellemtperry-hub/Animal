from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


COMMON_SPECIES_FILE = "common_trailcam_species.csv"
ALIASES_FILE = "species_aliases.csv"
REGIONAL_SPECIES_FILE = "north_burnett_species.csv"


@dataclass(slots=True)
class SpeciesReference:
    common_species: list[str]
    regional_species: set[str]
    alias_map: dict[str, str]

    def suggest(self, limit: int = 12) -> list[str]:
        return self.common_species[:limit]

    def normalize(self, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            return ""
        return self.alias_map.get(normalized, normalized)

    def in_common(self, species: str) -> bool:
        return species in set(self.common_species)

    def in_regional(self, species: str) -> bool:
        return species in self.regional_species

    def known(self, species: str) -> bool:
        return self.in_common(species) or self.in_regional(species)


def _ensure_common_species_csv(path: Path) -> None:
    if path.exists():
        return
    rows = [
        ["species"],
        ["kangaroo"],
        ["wallaby"],
        ["fox"],
        ["cat"],
        ["dog"],
        ["cow"],
        ["horse"],
        ["goanna"],
        ["possum"],
        ["bandicoot"],
        ["echidna"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _ensure_aliases_csv(path: Path) -> None:
    if path.exists():
        return
    rows = [
        ["alias", "normalized_species"],
        ["roo", "kangaroo"],
        ["grey kangaroo", "eastern grey kangaroo"],
        ["goanna", "lace monitor"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _ensure_regional_species_csv(path: Path) -> None:
    if path.exists():
        return
    rows = [
        ["species", "common_name", "notes"],
        ["eastern grey kangaroo", "Eastern Grey Kangaroo", "sample placeholder"],
        ["red-necked wallaby", "Red-necked Wallaby", "sample placeholder"],
        ["lace monitor", "Lace Monitor", "sample placeholder"],
        ["common brushtail possum", "Common Brushtail Possum", "sample placeholder"],
        ["short-beaked echidna", "Short-beaked Echidna", "sample placeholder"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def ensure_reference_files(reference_dir: Path) -> None:
    _ensure_common_species_csv(reference_dir / COMMON_SPECIES_FILE)
    _ensure_aliases_csv(reference_dir / ALIASES_FILE)
    _ensure_regional_species_csv(reference_dir / REGIONAL_SPECIES_FILE)


def _read_first_col(path: Path, expected_header: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        key = expected_header if expected_header in reader.fieldnames else reader.fieldnames[0]
        values = []
        for row in reader:
            value = (row.get(key) or "").strip().lower()
            if value:
                values.append(value)
        return values


def _read_aliases(path: Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alias = (row.get("alias") or "").strip().lower()
            normalized = (row.get("normalized_species") or "").strip().lower()
            if alias and normalized:
                aliases[alias] = normalized
    return aliases


def load_species_reference(reference_dir: Path) -> SpeciesReference:
    ensure_reference_files(reference_dir)
    common = _read_first_col(reference_dir / COMMON_SPECIES_FILE, "species")
    regional = set(_read_first_col(reference_dir / REGIONAL_SPECIES_FILE, "species"))
    aliases = _read_aliases(reference_dir / ALIASES_FILE)
    return SpeciesReference(common_species=common, regional_species=regional, alias_map=aliases)
