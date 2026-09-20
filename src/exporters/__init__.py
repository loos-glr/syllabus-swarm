"""syllabus-swarm — exporters package."""

from src.exporters.file_writer import (
    OUTPUT_PATHS,
    FileWriteError,
    OutputPathConfig,
    write_directory_tree,
    write_file,
    write_remotion_manifest,
    write_rubric,
    write_syllabus,
)
from src.exporters.manifest import (
    ArtifactSummary,
    ManifestData,
    update_output_manifest,
)

__all__ = [
    # file_writer
    "FileWriteError",
    "OutputPathConfig",
    "OUTPUT_PATHS",
    "write_directory_tree",
    "write_file",
    "write_remotion_manifest",
    "write_syllabus",
    "write_rubric",
    # manifest
    "ArtifactSummary",
    "ManifestData",
    "update_output_manifest",
]
