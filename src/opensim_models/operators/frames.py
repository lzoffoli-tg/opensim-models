"""Build a standalone ``opensim.PhysicalOffsetFrame`` on a body, as a named
attachment point reusable anywhere this package accepts a body/frame
(``add_offset_frame``)."""

from __future__ import annotations

from typing import Any

from .. import components
from ._shared import _unwrap

__all__ = ["add_offset_frame"]


def add_offset_frame(
    model: "OpenSimModel",
    name: str,
    body: Any,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    reinitialize: bool = False,
) -> Any:
    """Build an ``opensim.PhysicalOffsetFrame`` on ``body`` and add it as a named attachment point, in one call.

    Before this function existed, the only ``PhysicalOffsetFrame`` built
    anywhere in this package was an internal, unnamed one
    (``OpenSimModel.add_model``'s "ground anchor", used purely as bookkeeping
    for joints that need to stay attached to ground across a merge) -- there
    was no public way to create a *new*, standalone offset frame on an
    arbitrary body of your own choosing. :class:`~opensim_models.components.OffsetFrame`
    already lets you *read/write* one that exists (e.g.
    :attr:`~opensim_models.components.Joint.parent_frame`/:attr:`~opensim_models.components.Joint.child_frame`);
    this is the matching *constructor*.

    **Where the new frame is attached, and why, confirmed empirically.**
    Added as a genuine subcomponent of ``body`` itself
    (``body.addComponent(frame)``, path ``/bodyset/<body>/<name>``) --
    *not* at the model's own root the way the internal ground anchor is
    (``model.addComponent(...)``, path ``/<name>``, deliberately "not
    tracked for removal... orphaned after remove_model", per that code's
    own comment). Confirmed directly that attaching to the body instead
    (rather than the model root) is the one choice that survives
    :meth:`~opensim_models.model.OpenSimModel.add_model`/``+`` merging a
    model containing this frame into another: cloning ``body`` (as every
    one of the 8 ``_MERGE_SETS`` categories does for its own members)
    deep-copies this subcomponent right along with it, confirmed by
    finding it at the expected path in the merged result afterward -- a
    root-level component, outside all 8 categories, would instead be
    silently left behind, exactly like the ground anchor already is.
    :meth:`~opensim_models.model.OpenSimModel.copy` (a full
    ``opensim.Model.clone()``) preserves it either way.

    **Confirmed usable anywhere this package accepts a body/frame,
    including as the *parent* side of a brand-new joint** (it is a genuine
    ``opensim.PhysicalFrame``, like any ``opensim.Body``): as ``body=`` to
    :func:`~opensim_models.operators.add_marker`/
    :func:`~opensim_models.operators.add_contact_sphere`/etc., and as
    ``to=`` to :func:`~opensim_models.operators.attach_component` (that
    function's own default ``parent_point="com"`` now falls back to this
    frame's own origin when the target has no mass centre of its own --
    see :func:`~opensim_models.operators.attachment._resolve_attachment_point`).

    Parameters
    ----------
    model : OpenSimModel
        Model ``body`` belongs to.
    name : str
        Name for the new offset frame.
    body : opensim.PhysicalFrame or components wrapper
        The body (or other physical frame) this offset frame is attached
        to.
    translation : tuple[float, float, float], optional
        Offset from ``body``'s own origin, in metres, in ``body``'s own
        local axes. Defaults to ``(0.0, 0.0, 0.0)``.
    orientation_deg : tuple[float, float, float], optional
        Orientation relative to ``body``, as X-Y-Z body-fixed Euler angles
        in degrees about ``body``'s own axes. Defaults to no tilt.
    reinitialize : bool, optional
        See :func:`~opensim_models.operators._shared.add_component`.

    Returns
    -------
    components.OffsetFrame
        The newly created frame, wrapped in
        :class:`~opensim_models.components.OffsetFrame`.

    Raises
    ------
    ValueError
        If ``translation`` or ``orientation_deg`` contains a non-finite
        value.
    """
    body = _unwrap(body)
    frame = model.opensim.PhysicalOffsetFrame(name, body, model.opensim.Transform())
    # Validate (and set) translation/orientation on the frame BEFORE it is
    # ever attached to body, so a bad value raises cleanly with no
    # structural side effect on the model at all -- reuses OffsetFrame's
    # own validated setters instead of duplicating the finiteness checks.
    wrapped = components.OffsetFrame(model, frame)
    wrapped.set_translation(translation)
    wrapped.set_orientation_deg(orientation_deg)

    def _attach() -> None:
        body.addComponent(frame)
        frame.thisown = False  # ownership now belongs to body; avoids a double-free on GC

    if reinitialize:
        with model.structural_change():
            _attach()
    else:
        _attach()
    return wrapped
