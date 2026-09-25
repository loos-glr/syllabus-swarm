"""syllabus-swarm — exporters package."""

from src.exporters.file_writer import (
    OUTPUT_PATHS,
    FileWriteError,
    OutputPathConfig,
    write_directory_tree,
    write_file,
    write_lesson_plan,
    write_presentation,
    write_remotion_manifest,
    write_rubric,
    write_syllabus,
)
from src.exporters.manifest import (
    ArtifactSummary,
    ManifestData,
    update_output_manifest,
)
from src.exporters.glr_marp_theme import (
    generate_marp_css,
    get_marp_frontmatter,
)

__all__ = [
    # file_writer
    "FileWriteError",
    "OutputPathConfig",
    "OUTPUT_PATHS",
    "write_directory_tree",
    "write_file",
    "write_lesson_plan",
    "write_presentation",
    "write_remotion_manifest",
    "write_syllabus",
    "write_rubric",
    # manifest
    "ArtifactSummary",
    "ManifestData",
    "update_output_manifest",
    # glr_marp_theme
    "generate_marp_css",
    "get_marp_frontmatter",
]
