"""Regenerate ``User``'s generated posture/joint-center mixin modules.

Run after editing ``src/opensim_models/models/user/_posture_table.py`` or
``_joint_center_table.py`` (e.g. to add a new posture coordinate or
joint-center property):

    python scripts/generate_user_code.py

``render_posture_module()``/``render_joint_centers_module()`` are pure,
side-effect-free string builders (also imported directly by
``tests/test_user_generated_code_is_fresh.py`` to check the committed
generated files are up to date with the tables) -- only the ``__main__``
block below actually writes anything to disk.

The generated methods are deliberately literal ``def``/``@property`` source
(not built via ``setattr``/metaclass metaprogramming) so VS Code/Pylance
autocomplete and hover-docs for e.g. ``user.left_hip`` or
``user.set_left_hip_flexionextension`` work exactly as if they were
hand-written -- static analysis tools cannot see dynamically-assigned
attributes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from opensim_models.models.user._posture_table import POSTURE_ENTRIES
from opensim_models.models.user._joint_center_table import JOINT_CENTER_ENTRIES

POSTURE_GENERATED_PATH = SRC / "opensim_models" / "models" / "user" / "_posture_generated.py"
JOINT_CENTERS_GENERATED_PATH = (
    SRC / "opensim_models" / "models" / "user" / "_joint_centers_generated.py"
)

_HEADER = """# GENERATED FILE -- DO NOT EDIT BY HAND.
#
# Regenerate with: python scripts/generate_user_code.py
# Edit {table} instead, then re-run that command.

from __future__ import annotations
"""


def _render_docstring(text: str, indent: str) -> str:
    """Wrap ``text`` (a raw docstring value) back into a triple-quoted literal.

    Only the opening ``\"\"\"`` marker gets ``indent`` -- every continuation
    line is emitted exactly as stored, since the table already carries each
    docstring's original embedded indentation (8 spaces, matching a method
    body one level deeper than its class), which is also exactly the depth
    a mixin method sits at here. See ``_bootstrap_extract.py``'s
    ``render_triple_quoted`` (used once, to build the tables) for the same
    reasoning.
    """
    lines = text.split("\n")
    quoted_lines = [f'{indent}"""{lines[0]}']
    quoted_lines.extend(lines[1:])
    quoted_lines[-1] += '"""'
    return "\n".join(quoted_lines)


def render_posture_module() -> str:
    """Render ``_posture_generated.py``'s full source from ``POSTURE_ENTRIES``."""
    parts = [_HEADER.format(table="_posture_table.py"), ""]
    parts.append('class _PostureMixin:')
    parts.append(
        '    """Every posture setter/getter pair, generated from'
        " ``_posture_table.POSTURE_ENTRIES``."
        '"""'
    )
    parts.append("")
    for entry in POSTURE_ENTRIES:
        parts.append(f"    def set_{entry.attr_suffix}(self, degrees: float):")
        parts.append(_render_docstring(entry.setter_docstring, "        "))
        parts.append(f'        self.coordinate({entry.coordinate!r}).set_value_degrees(degrees)')
        parts.append("")
        parts.append("    @property")
        parts.append(f"    def {entry.attr_suffix}(self) -> float:")
        parts.append(_render_docstring(entry.getter_docstring, "        "))
        parts.append(f'        return self.coordinate({entry.coordinate!r}).value_degrees')
        parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


def render_joint_centers_module() -> str:
    """Render ``_joint_centers_generated.py``'s full source from ``JOINT_CENTER_ENTRIES``."""
    parts = [_HEADER.format(table="_joint_center_table.py"), ""]
    parts.append("class _JointCenterMixin:")
    parts.append(
        '    """Every joint-center property, generated from'
        " ``_joint_center_table.JOINT_CENTER_ENTRIES``."
        '"""'
    )
    parts.append("")
    for entry in JOINT_CENTER_ENTRIES:
        parts.append("    @property")
        parts.append(f"    def {entry.attr_name}(self) -> tuple[float, float, float]:")
        parts.append(_render_docstring(entry.docstring, "        "))
        parts.append(f'        return self._joint_center({entry.joint_name!r})')
        parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


if __name__ == "__main__":
    POSTURE_GENERATED_PATH.write_text(render_posture_module())
    JOINT_CENTERS_GENERATED_PATH.write_text(render_joint_centers_module())
    print(f"wrote {POSTURE_GENERATED_PATH}")
    print(f"wrote {JOINT_CENTERS_GENERATED_PATH}")
