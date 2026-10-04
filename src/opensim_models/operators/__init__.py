"""Generic add/remove/move operators for OpenSim model components.

Thin wrappers around OpenSim's own component ``Set`` API (``BodySet``,
``JointSet``, ``ForceSet``, ``MarkerSet``, ``ConstraintSet``,
``ControllerSet``, ``ContactGeometrySet``, ``ProbeSet``), covering every
component category an :class:`~opensim_models.model.OpenSimModel` exposes,
plus two repositioning operators -- :func:`rotate_object` and
:func:`translate_object` -- that work on a component or on a whole model.

None of the add/remove functions rebuild the model's system by default:
adding or removing a component is a structural change that leaves
``model.state`` temporarily invalid (not just its defaults, but the state
object itself -- reading it crashes the process rather than raising a
catchable error). Pass ``reinitialize=True`` for a single, self-contained
call; for a batch (e.g. a body and the joint connecting it, or several
removals), wrap the whole batch in
:meth:`~opensim_models.model.OpenSimModel.structural_change` instead and
leave every call's ``reinitialize`` at its default ``False``:

>>> with model.structural_change():
...     body = operators.add_body(model, "b1", mass=2.0)
...     operators.add_joint(model, model.opensim.FreeJoint("b1_to_ground", model.model.getGround(), body.raw))

:func:`rotate_object`/:func:`translate_object` are a different kind of
operator: they don't add or remove anything, only reposition what's
already there (a ``Marker``, a joint's ``PhysicalOffsetFrame``, or an
entire :class:`~opensim_models.model.OpenSimModel`), and call
:meth:`~opensim_models.model.OpenSimModel.reinitialize` themselves, so they
need no ``reinitialize=``/``structural_change()`` handling from the caller.
:meth:`OpenSimModel.rotate <opensim_models.model.OpenSimModel.rotate>` and
:meth:`OpenSimModel.translate <opensim_models.model.OpenSimModel.translate>`
are convenience methods for rotating/translating a whole model without
importing this module directly.

:func:`euclidean_distance` is a third, even simpler kind of operator: a
pure geometry helper with no ``OpenSimModel`` involved at all, for
measuring the straight-line distance between two plain ``(x, y, z)``
points -- e.g. two values returned by
:attr:`~opensim_models.components.Body.position_global`/
:attr:`~opensim_models.components.Body.position_local` or the matching
properties elsewhere in :mod:`opensim_models.components`.

:func:`from_global_to_local`/:func:`from_local_to_global` are a fourth
kind: unlike :func:`euclidean_distance` they do read an object's current
pose (so they are not pure), but unlike every ``add_*``/``remove_*``
function above they take no explicit ``model=``/``reinitialize=`` either
-- they accept a :mod:`opensim_models.components` wrapper (e.g. a
``Body``, ``Box``, ``Screen``, ``OffsetFrame``) or the matching raw
``opensim`` object directly, and transform an arbitrary ``(x, y, z)``
point between the ground frame and that object's own local frame.

Organized one file per component category (``bodies``, ``joints``,
``attachment``, ``frames`` for ``add_offset_frame``, ``primitives``,
``forces``, ``markers``, ``constraints``,
``auxiliary`` for the generic ``add_contact_geometry``/controllers/probes,
``contact`` for the named contact-geometry builders (``add_contact_sphere``/
``add_contact_half_space``/``add_contact_mesh``) and the real, force-based
``add_sliding_point_contact``) plus ``rotation``/``translation`` and their
shared ``_spatial`` helpers, and ``geometry`` for ``euclidean_distance``/
``from_global_to_local``/``from_local_to_global`` --
this module just re-exports every public name from all of them, so
``opensim_models.operators.<name>`` resolves exactly as it did when this
was a single flat file.
"""

from __future__ import annotations

from .. import components

from ._shared import add_component, remove_component
from .bodies import add_body, remove_body
from .joints import (
    add_joint,
    remove_joint,
    add_free_joint,
    add_pin_joint,
    add_ball_joint,
    add_slider_joint,
    add_weld_joint,
)
from .attachment import attach_component
from .frames import add_offset_frame
from .primitives import add_box_body, add_cylinder_body, add_sphere_body
from .forces import add_force, remove_force, add_muscle, remove_muscle
from .markers import add_marker, remove_marker
from .constraints import (
    add_constraint,
    remove_constraint,
    add_weld_constraint,
    add_point_constraint,
    add_coordinate_coupler_constraint,
    add_point_on_plane_constraint,
)
from .auxiliary import (
    add_controller,
    remove_controller,
    add_contact_geometry,
    remove_contact_geometry,
    add_probe,
    remove_probe,
)
from .contact import (
    add_contact_sphere,
    add_contact_half_space,
    add_contact_mesh,
    add_sliding_point_contact,
)
from .rotation import rotate_object
from .translation import translate_object
from .geometry import euclidean_distance, from_global_to_local, from_local_to_global

__all__ = [
    "add_component",
    "remove_component",
    "add_body",
    "remove_body",
    "add_joint",
    "remove_joint",
    "add_free_joint",
    "add_pin_joint",
    "add_ball_joint",
    "add_slider_joint",
    "add_weld_joint",
    "attach_component",
    "add_offset_frame",
    "add_box_body",
    "add_cylinder_body",
    "add_sphere_body",
    "add_force",
    "remove_force",
    "add_muscle",
    "remove_muscle",
    "add_marker",
    "remove_marker",
    "add_constraint",
    "remove_constraint",
    "add_weld_constraint",
    "add_point_constraint",
    "add_coordinate_coupler_constraint",
    "add_point_on_plane_constraint",
    "add_controller",
    "remove_controller",
    "add_contact_geometry",
    "remove_contact_geometry",
    "add_probe",
    "remove_probe",
    "add_contact_sphere",
    "add_contact_half_space",
    "add_contact_mesh",
    "add_sliding_point_contact",
    "rotate_object",
    "translate_object",
    "euclidean_distance",
    "from_global_to_local",
    "from_local_to_global",
]
