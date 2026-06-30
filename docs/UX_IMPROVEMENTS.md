# Portrait Enhancer — UX Improvements (Staged)

This is a usability review of the **Qt frontend** (`portrait_enhancer/ui_qt/main_window.py` —
the active, forward-path UI per `ROADMAP.md`; the legacy Tkinter app in `portrait_enhancer/ui/app.py`
is out of scope and being migrated away from).

Scope note: this is distinct from `ROADMAP.md`, which tracks *missing features* (crop tools,
heal brush, gradients...). Everything below is about *existing* features being harder to use,
discover, or trust than they should be — found by reading the code, not by a usability study.
File:line citations point at the behavior in question; spot-checked a sample directly to confirm
accuracy, but verify against current code before implementing, since this file will drift as the
app changes.

Staged by a mix of effort and risk, not just impact — Stage 1 is deliberately full of small,
isolated, low-risk changes so it can be picked up incrementally without a design discussion.
Stage 3 items touch shared infrastructure (threading model, undo stack, settings) and are sized
for dedicated focus.

---

## Stage 1 — Low-effort, self-contained fixes

Each of these is a localized change (a handler, a label, a dialog) that doesn't require
touching shared architecture. Good first-PR material, any order.

1. **No persistent "which face/person am I editing" indicator.** The Face Target combo
   (`main_window.py:~3065`, moved next to the Layers tabs earlier this project) selects which
   detected face/person the Face/Skin/Eyes/Lips/Hair/Person layers apply to, but nothing else
   on screen reflects that choice — not the layer tabs, not the canvas. A user who switches
   Face Target then gets distracted by the filmstrip can easily edit the wrong person without
   noticing. Fix: add a small persistent label near the layer tabs ("Editing: Face 2") and draw
   a thin highlight around that face's bounding box on the canvas, both updated on every Face
   Target / face-switch change.

2. **No zoom-level readout.** Pinch/scroll zoom and double-click-to-fit are implemented
   (`main_window.py:919-925`, canvas pan/zoom handlers ~807-1002) but there is no on-screen
   percentage anywhere — confirmed via grep, no zoom label exists. Add a small "120%" readout
   in a canvas corner or status bar, updated on every zoom change.

3. **Filmstrip doesn't auto-scroll to the active thumbnail** (`_update_filmstrip_active`,
   `main_window.py:7217-7222`) — it only restyles the thumbnail, never calls
   `ensureWidgetVisible`. In a collection of 50+ images, clicking near either end of the
   filmstrip can select an image whose thumbnail then sits off-screen. One-line fix.

4. **No double-click-to-reset on individual sliders.** The canvas has double-click-to-reset
   for zoom/pan (`main_window.py:919-925`), but slider widgets have no equivalent — resetting
   one slider means dragging it back by hand or finding a layer-level "Reset" button. Add a
   `mouseDoubleClickEvent` on the slider widgets that resets just that control to its default.

5. **No Escape to back out of a modal editing mode.** `keyPressEvent` (`main_window.py:6289-6309`)
   only handles Space (before/after compare) — confirmed by reading the handler. Crop mode,
   mask-edit mode, and the white-balance eyedropper all have to be exited via their own toggle
   button; there's no universal "get me out of here" key.

6. **No confirmation before removing an image from a collection** (`main_window.py:~6954-6970`).
   A misplaced right-click silently drops an image from the collection with only a status-bar
   message and no undo path, unlike collection *deletion*, which already has a confirmation
   dialog a few hundred lines away (`~4005-4011`) — so the precedent for "ask first" already
   exists in this codebase, just not applied consistently here.

7. **Settings-copied indicator doesn't reflect repeated pastes.** The "copied ✓" clipboard
   indicator (`main_window.py:~4301-4302, 4423-4424`) clears after one paste; pasting the same
   clipboard onto a second or third image gives no per-paste confirmation. Add a brief
   transient confirmation ("Pasted to image_003.jpg") on every paste, not just the first.

8. **Paste across a face-count mismatch is silent.** Pasting settings copied from a one-face
   photo onto a three-face photo applies them only to face index 0, with the others left
   untouched (`main_window.py:~4375-4396`) and no warning either way. A one-line warning
   ("Source had 1 face; only Face 1 will be updated here") avoids a confusing half-applied
   result.

9. **Acceleration mode (GPU/CPU) has no UI control.** `_runtime_settings = {"acceleration_mode":
   "auto"}` (`main_window.py:~2319`) is set once at startup and never exposed; a user
   troubleshooting a slow or crashing GPU path has no way to force CPU-only short of editing
   config. A simple dropdown (Auto / CPU-only / GPU-only) surfaces an option that already
   exists in the backend.

10. **Sliders have no tooltip explaining their range/effect.** Most sliders (e.g. Sharpness
    -100..100) show only a numeric value with no indication of what a "normal" adjustment looks
    like, while a few buttons elsewhere do have explanatory tooltips — so the convention exists,
    it's just inconsistently applied. Worth an audit pass across `config.py`'s slider
    definitions and the widgets that render them.

11. **Mask-edit tools (Grow/Shrink/Invert/Feather) have no inline explanation**
    (`main_window.py:~3100-3175`). These are non-obvious operations (does "Grow" expand into
    background or inward?) with no tooltip text today.

12. **"No faces detected" gives no guidance.** When segmentation finds zero faces, the
    Face-Target-dependent layers go inactive with no explanation surfaced
    (`main_window.py:~7638-7647`) — a status message clarifying that global/Subjects/Background
    edits still work, and suggesting a tighter crop if it's a group/landscape shot, would save a
    confused "why are half the tabs greyed out" moment.

---

## Stage 2 — Workflow & feedback improvements

These touch one feature's worth of UI (a dialog, a panel) rather than a single control, and
mostly add visibility into state/processes that already exist but aren't surfaced.

13. **Batch job errors collapse to "the last one."** `BatchJobsDialog` shows aggregate counts
    (`ok=12 skip=3 err=2`) and only the single most recent error string
    (`main_window.py:~579-606, 5693`) — if five different images fail for five different
    reasons, four of those reasons are invisible without opening the raw JSONL log. Show
    distinct error types/counts, or a per-image expandable list.

14. **No warning before a batch export overwrites existing output.** `CollectionExportDialog`
    estimates how many files will be *skipped* when "skip completed" is on, but gives no
    warning about files that will be *overwritten* when it's off (`main_window.py:~4105-4155`).
    Single-image export already prompts before an overwrite elsewhere in the app — extend that
    pattern here.

15. **Culled images have no recovery UI.** Once `ReviewCullDialog` moves images into
    `collection.culled_images`, there's no in-app view of that list and no "restore" action
    (`main_window.py:~7031-7037`) — the data persists in `collections.json`, it's just
    invisible. A "View Culled (N)" entry point with per-image restore closes this gap.

16. **Per-image collection overrides are invisible in the filmstrip.** There's no badge
    distinguishing "this image has custom edits" from "this image is using collection
    defaults" (`main_window.py:~4020-4027`) — easy to forget which images were hand-tuned
    before re-applying a batch preset across the whole collection.

17. **No preset preview before applying.** The preset browser shows a static thumbnail
    (`main_window.py:~4914-4929`) but clicking Apply commits immediately to the working
    image — no live preview-then-confirm step, so trying a preset means being ready to Undo if
    it's wrong.

18. **No "apply preset to whole collection" — only "apply copied settings to collection."**
    `Apply Settings to Collection` (`main_window.py:~4029-4071`) works from the clipboard, not
    from a preset directly, so applying a saved preset collection-wide is a two-step
    copy-then-apply detour instead of one action from the preset browser.

19. **No "paste to multiple selected images" — paste is one-at-a-time** even though the
    filmstrip already supports multi-select (`main_window.py:~7227-7233`). Add a "Paste to
    Selection" action alongside the existing single-image paste.

20. **Import doesn't recurse into subfolders.** Folder import scans only the top level
    (`main_window.py:~6916`) — a date/RAW-organized folder tree (common for camera dumps)
    yields zero images until manually flattened. Worth an "include subfolders" checkbox.

21. **Missing/moved source files surface as a bare error, not a collection-level warning.**
    If a collection references a file that's since been moved or deleted, clicking its
    thumbnail throws a generic "file not found" with no indication of which other images in the
    collection are similarly broken (`main_window.py:~7379-7383`). A one-time scan on collection
    load that flags all missing files at once is more useful than discovering them one click at
    a time.

22. **`acceleration_mode` aside, there's no settings/preferences surface at all** — default
    export folder, default quality, etc. are all picked fresh from a file dialog every time
    (`main_window.py:~7416-7450`). Doesn't need to be elaborate — a single Preferences dialog
    consolidating the handful of app-wide settings that exist today is enough for this stage;
    Stage 3 below covers anything that needs new settings invented.

---

## Stage 3 — Structural work

These need dedicated focus because they touch shared infrastructure: the threading model, the
undo stack, or add a genuinely new subsystem (a settings dialog with persistence, an
accessibility pass). Tackle one at a time.

23. **Several heavy operations run synchronously on the UI thread.** Confirmed directly:
    `_load_image_path` (`main_window.py:6405-6424`) calls `_read_image_file` — a `rawpy`
    decode — with no `QThreadPool`/worker wrapper, so opening a large RAW file freezes the
    window for however long the decode takes. The same pattern recurs for "Fix Eyes"
    (`~6687-6738`), the per-image loop behind "Auto WB (AI)" across a multi-selection
    (`~6020-6067`), and the import dialog's folder scan on large folders (`~6912-6920`). Deep
    Denoise (`~6762-6831`) and segmentation (`~1718` area) *do* already use a worker
    pool/signals pattern — that's the template to extend to the others, not a pattern to invent
    from scratch.

24. **Undo/redo covers three separate, inconsistent scopes.** Slider/layer edits go through
    one document-history stack; mask paint/erase edits go through a separate stack
    (`main_window.py:~2237-2242, 8004-8049`); crop/framing and collection-membership changes
    aren't covered by either. A user has no way to predict, in the moment, whether Ctrl+Z will
    undo what they just did. Either unify into one timeline or make the boundary visible (e.g.
    a labeled split-button: "Undo Edit" / "Undo Mask").

25. **Collection state is saved with a bare `except Exception: return`.**
    `_save_collections_state` (`main_window.py:3921-3930`, confirmed by direct read) — if the
    write fails for any reason (full disk, permissions, a moved/unmounted path), the failure is
    completely silent: no log line, no status message, nothing. The user has no way to know
    their session's collection state didn't persist until they restart and find it missing.
    This is the single highest-value fix in this document relative to its size — a few lines
    to at minimum log the exception and show a non-blocking warning.

26. **No actionable remapping of backend error messages.** Segmentation failures
    (`~7625-7626`), corrupted/unreadable RAW files (`~7390-7398`), and missing model files all
    surface as raw exception text in a status bar or `QMessageBox`. None of these need a new
    subsystem, just a translation layer: catch the known failure classes (OOM, model-not-found,
    decode error) at the point they're already caught, and remap to one sentence of plain-
    language guidance instead of the exception's `str()`.

27. **First-run / empty states under-inform.** A blank filmstrip shows shortcut text but no
    actionable button (`~6868-6875`); the startup readiness dialog can be dismissed even when a
    required model is missing, with no persistent reminder afterward (`~5299-5309`). Worth a
    pass over every "nothing here yet" and "something's degraded" state at once, since they
    share a design (a short explanation + one clear action), rather than fixing them
    one-by-one.

28. **No app-wide settings/preferences subsystem.** Stage 1 #9 (acceleration mode) and Stage 2
    #22 (defaults) are symptoms of there being no single place persisted settings live or get
    edited. Building one real `QDialog` + a small persisted-settings file is the structural fix
    those smaller items are working around.

29. **Accessibility basics are largely unaddressed.** No `accessibleName`/`accessibleDescription`
    on key widgets (e.g. the Face Target combo's items are positional text only,
    `~7643-7644`); a couple of stylesheets hardcode pixel font sizes that won't respect OS text
    scaling (`~2395-2413`); histogram clipping indicators rely on color alone with no text/
    pattern backup (`~1362-1376`). Worth a dedicated pass rather than fixing in isolation, since
    these all need the same kind of review (screen-reader labels, scalable fonts, redundant
    color+text/pattern encoding) across the whole window.

---

## Suggested order

Stage 1 items are independent — pick any subset for a single PR. Within Stage 2, prioritize
#13/#14 (batch reporting/overwrite safety) and #15/#16 (culled-image and override visibility)
since they affect trust in the export pipeline, which is where mistakes are most expensive.
Within Stage 3, do #25 (silent save failure) first regardless of order chosen for the rest — it's
small, high-value, and unblocks nothing else, so there's no reason to sequence it behind the
bigger items.
