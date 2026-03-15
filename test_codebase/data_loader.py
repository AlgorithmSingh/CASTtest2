"""Data loading and preprocessing utilities for CSV and JSON files."""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataRecord:
    """Represents a single data record with typed fields."""

    id: int
    name: str
    value: float
    category: str = "default"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "category": self.category,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataRecord":
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            value=float(data["value"]),
            category=data.get("category", "default"),
            tags=data.get("tags", []),
        )


class CSVLoader:
    """Load and parse CSV files into DataRecord objects."""

    def __init__(self, filepath: str, delimiter: str = ","):
        self.filepath = filepath
        self.delimiter = delimiter
        self._records: Optional[List[DataRecord]] = None

    def load(self) -> List[DataRecord]:
        """Read the CSV file and return a list of DataRecord objects.

        Returns:
            List of DataRecord objects parsed from the CSV.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If required columns are missing.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")

        records = []
        with open(self.filepath, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            for row in reader:
                record = DataRecord(
                    id=int(row["id"]),
                    name=row["name"],
                    value=float(row["value"]),
                    category=row.get("category", "default"),
                    tags=row.get("tags", "").split(";") if row.get("tags") else [],
                )
                records.append(record)

        self._records = records
        return records

    def filter_by_category(self, category: str) -> List[DataRecord]:
        """Return records matching the given category."""
        if self._records is None:
            self.load()
        return [r for r in self._records if r.category == category]

    def filter_by_value_range(
        self, min_val: float, max_val: float
    ) -> List[DataRecord]:
        """Return records whose value falls within [min_val, max_val]."""
        if self._records is None:
            self.load()
        return [r for r in self._records if min_val <= r.value <= max_val]


class JSONLoader:
    """Load and parse JSON files into DataRecord objects."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._records: Optional[List[DataRecord]] = None

    def load(self) -> List[DataRecord]:
        """Read the JSON file and return a list of DataRecord objects.

        Expects the JSON to be either a list of objects or an object
        with a 'records' key containing a list.

        Returns:
            List of DataRecord objects.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"JSON file not found: {self.filepath}")

        with open(self.filepath, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            raw_records = data
        elif isinstance(data, dict) and "records" in data:
            raw_records = data["records"]
        else:
            raise ValueError("JSON must be a list or contain a 'records' key")

        self._records = [DataRecord.from_dict(r) for r in raw_records]
        return self._records

    def save(self, records: List[DataRecord], output_path: str) -> None:
        """Save a list of DataRecord objects to a JSON file."""
        data = [r.to_dict() for r in records]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def merge_files(self, other_paths: List[str]) -> List[DataRecord]:
        """Load and merge records from multiple JSON files.

        Args:
            other_paths: List of additional JSON file paths to merge.

        Returns:
            Combined list of all DataRecord objects, deduplicated by id.
        """
        all_records = {}
        for record in self.load():
            all_records[record.id] = record

        for path in other_paths:
            loader = JSONLoader(path)
            for record in loader.load():
                if record.id not in all_records:
                    all_records[record.id] = record

        return list(all_records.values())


def validate_records(records: List[DataRecord]) -> List[str]:
    """Validate a list of DataRecord objects and return error messages.

    Checks:
        - No duplicate IDs
        - All names are non-empty
        - All values are non-negative
        - Categories are from an allowed set

    Returns:
        List of validation error strings (empty if all valid).
    """
    errors = []
    seen_ids = set()
    allowed_categories = {"default", "premium", "basic", "enterprise"}

    for record in records:
        if record.id in seen_ids:
            errors.append(f"Duplicate ID: {record.id}")
        seen_ids.add(record.id)

        if not record.name.strip():
            errors.append(f"Record {record.id}: empty name")

        if record.value < 0:
            errors.append(f"Record {record.id}: negative value {record.value}")

        if record.category not in allowed_categories:
            errors.append(
                f"Record {record.id}: invalid category '{record.category}'"
            )

    return errors
