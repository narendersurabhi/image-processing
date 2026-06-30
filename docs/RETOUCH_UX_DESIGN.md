# Portrait Enhancer — Low-Resistance Retouch & Export UX

A design document for upgrading the **retouching and export experience** so that polishing a
single portrait *or a group portrait* feels fast, confident, and low-friction — from the moment
an image opens to the moment it's exported.

This is a **design + trade-off** document, not an implementation plan. It proposes a target
experience, argues the decisions behind it, lays out the alternatives we're rejecting and why,
and sequences the work by leverage. File:line references point at today's behavior so each
proposal is anchored to something real; verify against current code before building, since the
UI drifts.

Scope relationship to the other docs:
- [ROADMAP.md](../ROADMAP.md) tracks *missing feature classes* (heal brush, gradients, denoise).
- [docs/UX_IMPROVEMENTS.md](UX_IMPROVEMENTS.md) is a *bug-list-style* review of existing controls
  being harder to use than they should be (29 staged fixes).
- **This document** is the *forward design*: what the end-to-end retouch funnel should become.
  It reuses several UX_IMPROVEMENTS items as ingredients but organizes them around one goal
  instead of by effort/risk.

> **Revision 2026-06-23 — stronger masking foundation landed.** The subject/background and
> per-person mask backend got materially stronger since first draft: a new
> `BackgroundRemovalSegmenter` adds local **RMBG-2.0** matting (`core/segmentation.py:918`), the
> backend order is now **RMBG-2.0 → MODNet → MediaPipe → heuristic**, and when Mask DINO is
> available RMBG foreground is softly **gated by person instances** to suppress table/object
> clutter (with per-image Mask DINO memoization so gating and Person masks don't run DINO twice;
> cache bumped to `CACHE_VERSION = 14`). This doesn't change the *UX* proposals below — it
> *strengthens their foundation*. The per-face targeting (§6.2) and fan-out (§6.3) features lean
> on accurate, clean-edged per-person masks; those are now much closer to reliable. The combined
> mask strategy this unlocks is written up as **§6.0** (new), and the relevant trade-offs in §5
> and §6.2 are updated. RMBG-2.0 ships **opt-in / non-commercial** (gated download via
> `scripts/download_rmbg_model.py`), so none of this can be assumed *present* — see the
> availability note in §6.0.

---

## 1. Goal & non-goals

**Goal.** Minimize *resistance* — the number of decisions, clicks, mode-switches, and moments of
doubt — between "image is open" and "edited file is exported," for both single and group
portraits. Group portraits are the harder, more valuable case and the explicit design driver.

"Resistance" has four components, and the design attacks each:

| Source of resistance | What it feels like | Where it bites hardest |
|---|---|---|
| **Finding** the right control | "Which of 8 tabs and 6 sections has skin smoothing?" | First-time and occasional users |
| **Targeting** the right person/region | "Am I editing Face 2 or did the filmstrip steal focus?" | **Group portraits (×N people)** |
| **Trusting** the result | "Will this preset wreck what I have? Will Ctrl+Z undo the right thing?" | Everyone, every commit |
| **Exporting** | "Five dialogs to get one JPEG out." | Repeat/volume work |

**Non-goals.**
- Not redesigning the rendering pipeline, color management, or the semantic-mask backend — those
  are mature ([ROADMAP.md](../ROADMAP.md) intro) and out of scope here.
- Not adding *new heavy feature classes* for their own sake. Local healing (§6.6) is included
  only because "retouch" is incomplete without it, and it's flagged as the one structural bet.
- Not a visual restyle. This is about flow and information architecture, not theming.

---

## 2. Design principles

These are the tie-breakers for every decision below.

1. **A great starting point beats a blank slate.** The fastest edit is the one already 80%
   done on open. Default to a sensible auto-enhancement the user *adjusts*, not a flat image they
   must *build up*. (Always non-destructive and one-toggle-off.)
2. **Target by pointing, not by naming.** Selecting "who/what to edit" should be a click on the
   thing, on the canvas — not a dropdown of positional labels.
3. **Group = single, repeated for free.** Any per-face action must have a one-gesture "do this
   for everyone" form. The user should never pay an N× tax for N faces unless they *want*
   per-person control.
4. **Preview before commit; one predictable undo after.** Trying something should never require
   courage. Hovering/selecting previews; committing is one Ctrl+Z away on a single timeline.
5. **Progressive disclosure.** The common 20% of controls are always visible and shallow; the
   full depth is one deliberate step away, never gone.
6. **The default path has no dialogs.** Dialogs are for the long-tail. Export, in particular,
   should have a zero-dialog "do what I did last time" path.

---

## 3. Current-state friction audit (grounded)

What the funnel looks like today, and where it leaks. (Single image; multiply targeting steps by
*N* for a group of *N*.)

```
Open ─▶ [blank/neutral] ─▶ pick layer tab ─▶ pick Face Target ─▶ drag sliders ─▶ ... ─▶ Export dialog ─▶ file
        ^no starting point   ^8 tabs           ^combo, no canvas    ^per face,         ^modal, every time
                              + 6 sections       feedback (#1)        repeat ×N
```

- **No starting point on open.** The image opens neutral; the user assembles an edit from zero.
  Guided Recipes exist (`_guided_recipes`, `main_window.py:4533`) but are buried behind a nav
  button and a modal list, and **commit immediately with no live preview**
  ([UX_IMPROVEMENTS #17](UX_IMPROVEMENTS.md)).
- **Targeting is a dropdown, not the canvas.** The Face Target combo (`main_window.py:~3065`)
  drives every per-face layer, but *nothing on screen reflects which face is active*
  ([UX_IMPROVEMENTS #1](UX_IMPROVEMENTS.md)) — and you can't click a face on the canvas to select
  it. In a group, this is the dominant cost: select, scroll, edit, re-select, repeat, with a
  constant risk of editing the wrong person.
- **Control surface is wide and deep.** The right inspector stacks collapsible sections — Global,
  Layers, Masks, Geometry, White Balance, Preset Browser — and the Layers section alone is an
  8-tab `QTabWidget` (Subjects / Person / Face / Skin / Eyes / Lips / Hair / Background,
  `ALL_LAYERS`, `config.py:135`), each with grouped sliders (`LAYER_SLIDER_GROUPS`,
  `config.py:156`). Powerful, but a lot to *find* through for the common "brighten, smooth skin,
  pop the eyes" edit.
- **No "apply to everyone."** Recipes and copy/paste settings apply per-image and resolve to the
  active face; pasting a 1-face edit onto a 3-face photo silently touches only Face 1
  ([UX_IMPROVEMENTS #8](UX_IMPROVEMENTS.md)). There is no "smooth every face the same" action.
- **Trust gaps.** Undo/redo spans three inconsistent scopes (document history vs. mask stack vs.
  uncovered crop/collection changes — [UX_IMPROVEMENTS #24](UX_IMPROVEMENTS.md)), so the user
  can't always predict Ctrl+Z. Recipes/presets have no try-then-confirm.
- **Export is dialog-first, every time.** Single export is a modal with format/quality/resize/
  metadata/dest/filename (`ExportDialog`, `main_window.py:344`); there's no remembered one-click
  "export like last time." Collection export is a separate modal.
- **Some heavy steps block the UI.** RAW decode on open, Fix Eyes, multi-image Auto-WB still run
  on the GUI thread ([UX_IMPROVEMENTS #23](UX_IMPROVEMENTS.md)), so the window freezes mid-flow —
  the opposite of "low resistance." (Segmentation, render, denoise, the new AI-Select/Recalculate
  masks already use the worker-pool pattern; this is about extending it.)

The good news: the hard parts already exist. Semantic masks, per-face detection, recipes,
copy/paste, collections/filmstrip, batch export, a proxy-preview render path, and a worker-pool
threading pattern are all built — and as of the 2026-06-23 revision, the subject/background and
per-person mask backend is now genuinely strong (RMBG-2.0 matting, Mask-DINO person instances,
SAM-assisted repair via AI Select). **The upgrade is mostly flow and surfacing, not new engines:**
the mask *quality* foundation the group-portrait features depend on is largely in place, so the
remaining work is exposing it well — not building the masks.

---

## 4. The target experience (the funnel, redesigned)

```
Open ─▶ Auto-enhanced on arrival ─▶ Click a face on canvas to tweak ─▶ "Apply to all" if group ─▶ Quick Export
        (§6.1, toggleable)            (§6.2 + §6.3, no dropdown)         (§6.4, one gesture)        (§6.5, no dialog)
            │                                                                                            │
            └── one-tap "Enhance" / recipe strip, live-previewed (§6.1) ── unified undo throughout (§6.7) ┘
            └── "Simple / Advanced" inspector mode for control depth (§6.6) ───────────────────────────────┘
```

Concretely, the experience we're aiming for:

1. **Open an image → it already looks good.** A default auto-enhance (per-face) is applied
   non-destructively on arrival, with a single visible toggle to compare against the original or
   turn it off. The user starts by *adjusting*, not building.
2. **Want to change one person?** Click their face on the canvas. A clear badge says "Editing:
   Face 2," that face is outlined, and the inspector retargets. No dropdown.
3. **Group photo, want consistency?** One "Apply to all faces" control fans the current
   adjustment (or recipe) across everyone, with per-face exclusion if someone needs special care.
4. **Try a look risk-free.** A horizontal recipe strip previews on hover/focus and commits on
   click — onto one predictable undo timeline.
5. **Ship it.** A primary **Quick Export** button writes the file with last-used settings and a
   confirming toast; the full dialog is still there behind a smaller "Export As…" affordance. For
   a collection, one action exports the whole set.

---

## 5. The group-portrait deep-dive (the differentiator)

Group portraits are where a generic "good editor" becomes a *low-resistance* one, because every
per-face cost is paid *N* times. Four coordinated moves remove that tax — three about *flow*, and
one (the newest) about the *mask quality* the other three stand on:

**A. Canvas-as-selector (§6.2).** Clicking a face to target it turns an *N-step dropdown hunt*
into *N direct pointing gestures* — and removes the "wrong person" error class entirely, because
the active face is always the one you literally just clicked and it's outlined on screen.

**B. Fan-out actions (§6.3).** "Apply to all faces" / "Apply recipe to everyone" makes the common
group intent ("make all the skin consistent," "brighten everyone equally") a single gesture
instead of *N* repetitions. Crucially, this is **opt-out, not all-or-nothing**: applying to all,
then clicking one face to nudge it, is the natural workflow.

**C. Per-face confidence (§6.2 indicator + a roster).** A small face roster (thumbnail chips of
each detected person, derived from existing detection boxes) gives an at-a-glance "have I touched
everyone? does anyone look off?" — the group-portrait equivalent of the filmstrip, but for faces
*within* the current image.

**D. Mask quality is the foundation (§6.0).** A, B, and C are only as trustworthy as the per-person
masks beneath them: clicking "Person 2" and fanning a recipe across the group both *assume* each
person is cleanly separated from their neighbors, the background, and foreground clutter. The
2026-06-23 backend work (Mask DINO instances, RMBG-2.0 alpha, person-gated foreground) is what
makes that assumption hold in real group shots — see §6.0 for the combined strategy and its
availability caveat.

**Trade-off — automatic per-face vs. uniform global.** Applying a recipe "to everyone" can mean
two different things: (a) the *same slider values* on each face, or (b) *per-face adapted* values
(e.g. smoothing scaled to each face's detected skin texture). (a) is predictable and cheap; (b)
is smarter but can feel inconsistent across a row of faces and is harder to reason about.
**Recommendation: ship (a) first** (uniform values, the mental model users expect from "apply to
all"), and treat (b) as a later, clearly-labeled "Auto-balance faces" action rather than the
default — so "apply to all" never surprises.

---

## 6. Design decisions & trade-offs

Each subsection states the decision, the alternatives considered, the trade-offs, and a
recommendation. Ordered roughly by leverage-per-effort.

### 6.0 Foundation: per-person mask quality (mostly landed)

**Decision.** Treat per-person mask quality as a *combined* result of three specialized backends,
each doing the job it's actually good at, rather than asking any single model to do all of it:

| Stage | Backend | Role |
|---|---|---|
| **Identity** | **Mask DINO** person instances | "*Which* pixels are Person 1 vs Person 2" — the only backend that separates *individuals* |
| **Edge / alpha** | **RMBG-2.0** matting (`core/segmentation.py:918`) | High-quality foreground alpha to recover hair and soft edges *after* identity is known |
| **Repair** | **SAM** via AI Select | User-guided fix when one automatic instance is wrong (click the object, get its mask) |

The target per-person mask = **Mask DINO instance, refined by RMBG alpha, with face/hair
guarantees**, and AI Select as the manual escape hatch. Subject/background (the scene-wide
Subjects/Background layers) uses the RMBG-2.0 → MODNet → MediaPipe → heuristic order, with RMBG
foreground softly **gated by person instances** when Mask DINO is present so tables/props don't
leak into "subject." Most of this shipped on 2026-06-23.

**Alternatives considered (and why rejected).**
- *Use RMBG-2.0 alone to select each person.* **Rejected.** RMBG is a foreground/background matte —
  it emits *one* alpha ("subject(s)" vs "background") and in a group photo merges everyone (and
  can swallow foreground objects). It has no notion of "Person 1 vs Person 2." Excellent for
  *edges*, useless for *identity*.
- *Use Mask DINO alone.* Good identity, but instance-segmentation edges (especially fine hair) are
  coarser than a dedicated matte. RMBG refinement is what makes the cutout *look* clean.
- *One model to rule them all.* No current single model does identity + matte-quality edges +
  guided repair well; the combined stack is strictly better and each piece is independently
  swappable.

**Trade-offs.**
- *Availability is not guaranteed.* RMBG-2.0 is **gated and non-commercial** — it ships opt-in via
  `scripts/download_rmbg_model.py` (explicit, not auto-downloaded), so the app must degrade
  gracefully when it's absent (the backend order already falls through to MODNet/MediaPipe/
  heuristic). **Every UX feature below must assume RMBG *may not be present*** and still work,
  just with slightly softer edges. System Check now surfaces whether RMBG-2.0 matting is ready.
- *Cost.* Running three models is heavier; mitigated by per-image Mask DINO memoization (gating +
  Person masks share one DINO pass) and the existing analysis cache (`CACHE_VERSION = 14`). The
  per-face **Recalculate** control already lets a user force a fresh single-layer recompute when a
  cached mask is wrong.
- *Licensing surfaces in UX.* Because RMBG is non-commercial-unless-licensed, any "this looks
  great" moment it powers carries a licensing footnote. Keep the *download* explicit and the
  *dependency optional* so the default install has no license obligation.

**Recommendation.** This foundation is **largely done** — the design job here is mostly to *expose
its state honestly* (System Check readiness, graceful degradation when RMBG is absent) and to make
sure the per-person features in §6.2/§6.3 read mask quality from this combined result, not from any
single backend. **No new engine work required for the funnel; this is now a dependency the flow
features can rely on.**

### 6.1 One-tap enhance + live-previewed recipe strip

**Decision.** Replace the buried Recipes modal with (1) an always-visible **Enhance** primary
action that applies a sensible default per-face recipe, and (2) a thin horizontal **recipe strip**
(the existing `_guided_recipes` set: Natural Portrait, Studio Clean, Outdoor Warm, Group
Background Pop) that **previews on hover/focus and commits on click**. Optionally, auto-apply the
default Enhance on open behind a preference.

**Alternatives considered.**
- *Keep the modal list.* Lowest effort, but it's a detour and gives no live preview — exactly the
  friction we're removing (UX_IMPROVEMENTS #17).
- *Full "AI auto-everything" with no recipe choice.* Smallest surface, but opaque and
  un-adjustable; users distrust a black box they can't steer.

**Trade-offs.**
- *Auto-enhance-on-open* is the single biggest "starting point" win (Principle 1) but is also the
  most *opinionated* — it changes what the user sees before they ask. Mitigations: it's
  non-destructive, has a prominent before/after toggle (Space already does compare,
  `keyPressEvent`), is one Ctrl+Z from gone, and is **off by default** with a first-run prompt to
  turn it on. Ship the recipe strip first; gate auto-on-open behind a preference once the recipe
  preview path is proven.
- *Live preview cost.* Previewing a recipe means a render. The proxy-preview path already exists
  (low-res, fast); hover-preview should reuse it and debounce. Risk: jank on large images —
  mitigate by previewing on *focus/selection* (keyboard or click-and-hold) if hover proves too
  chatty.

**Recommendation.** Build the recipe strip with live preview (reuses recipes + proxy render +
existing undo). Add **Enhance** as the first strip item / primary button. Defer auto-on-open to a
preference. **High leverage, moderate effort, low risk.**

### 6.2 Click-a-face-on-canvas targeting + persistent "who am I editing"

**Decision.** Make the canvas a face selector: clicking a detected face sets it as the active
Face Target. Add a persistent **"Editing: Face N"** badge near the layer tabs and a thin highlight
around the active face's box (UX_IMPROVEMENTS #1). Add a small **face roster** of chips for
quick selection and an at-a-glance "who's been touched" read.

**Alternatives considered.**
- *Badge + highlight only (no canvas click).* Cheaper, fixes the "which face?" confusion, but
  leaves targeting as a dropdown — half the win.
- *Hover-to-target.* Too twitchy; targeting should be a deliberate click, consistent with the new
  AI-Select tool's click model.

**Trade-offs.**
- *Click conflicts with other canvas tools.* The canvas already multiplexes pan/zoom, crop,
  mask-edit, WB-pick, and the new AI-Select (mutually-exclusive toggle pattern with Escape). Face
  selection must slot into that arbitration cleanly — simplest is "click a face = select it" only
  when no modal tool owns the click (mirroring how double-click-to-reset is gated). Low risk; the
  pattern is established.
- *Detection accuracy.* Clicking relies on detection boxes; a missed face can't be clicked.
  Mitigation: keep the combo/roster as a fallback selector, and the existing "no faces detected"
  guidance (UX_IMPROVEMENTS #12) still applies.
- *Edge quality of the selected person.* Selecting a person is only satisfying if their mask edges
  are clean (hair especially). The §6.0 stack (Mask DINO + RMBG-2.0 refinement) now delivers that
  *when RMBG is installed*; when it isn't, edges are softer but selection still works — so this
  feature must not *hard-depend* on RMBG, only benefit from it. AI Select + Recalculate remain the
  per-person repair path when an automatic instance is wrong.

**Recommendation.** Ship badge + highlight first (small, self-contained, immediate clarity), then
canvas-click targeting, then the roster. **Highest leverage for group portraits.** Builds directly
on the canvas-tool and detection infra already in place.

### 6.3 Fan-out: "Apply to all faces"

**Decision.** Add a one-gesture **Apply to all faces** for the current per-face adjustment/recipe,
applying uniform values to every detected face, with per-face opt-out (click a face afterward to
nudge it). Resolve the silent face-count mismatch on paste (UX_IMPROVEMENTS #8) by routing paste
through the same fan-out with an explicit "applied to N faces" confirmation.

**Alternatives considered.**
- *Per-face adapted ("auto-balance") as the default.* Smarter but unpredictable across a row of
  faces; see §5 trade-off. Defer to an explicit, separately-labeled action.
- *Leave it manual.* This is the core group-portrait tax; not removing it fails the brief.

**Trade-offs.**
- *Destructiveness vs. clarity.* "Apply to all" overwrites per-face edits some users hand-tuned.
  Mitigation: single undo step for the whole fan-out, and a confirming toast ("Applied Natural
  Portrait to 4 faces"). Consider a "skip faces I've already edited" option, defaulting off for
  predictability.
- *Define "all."* Subjects/Background layers are scene-wide already; fan-out applies only to the
  per-face layers (Face/Skin/Eyes/Lips/Hair/Person). Keep that boundary explicit in the label.

**Recommendation.** Ship uniform "Apply to all faces" with a confirming toast and one undo step.
**Together with §6.2, this is the group-portrait headline.** Moderate effort, moderate risk
(touches the per-face apply path).

### 6.4 Quick Export (zero-dialog default) + one-action collection export

**Decision.** Add a primary **Quick Export** that writes the file using last-used settings
(format/quality/dest/naming), shows a confirming toast with the path, and offers Undo-less "Open
folder." Keep the full `ExportDialog` behind a smaller **Export As…**. For collections, surface a
one-action "Export all" using the same remembered settings.

**Alternatives considered.**
- *Always show the dialog.* Status quo; fails Principle 6 for repeat/volume work.
- *Inline export settings in a panel.* More discoverable than a dialog but adds permanent panel
  weight for an action taken once per image; a remembered-defaults button is lighter.

**Trade-offs.**
- *Hidden defaults can surprise* (wrong folder/format). Mitigations: the toast names the exact
  file + path; "Export As…" is always one click away; first Quick Export of a session can confirm
  the destination once. Pairs naturally with a real Preferences home for export defaults
  (UX_IMPROVEMENTS #22/#28).
- *Overwrite safety.* Quick Export must honor the existing single-image overwrite prompt and the
  collection overwrite gap (UX_IMPROVEMENTS #14) — silent overwrite is a trust violation even if
  it's "fast."

**Recommendation.** Ship Quick Export with a path-naming toast and an always-present Export As…
fallback; reuse remembered settings; respect overwrite prompts. **High leverage for volume work,
low risk.**

### 6.5 Simple / Advanced inspector mode

**Decision.** Add a **Simple** inspector mode that surfaces a curated, flat set of the
highest-use controls — Exposure, Skin Smooth, Blemish, Eye Brighten/Whiten, Teeth Whiten,
Background Blur, Warmth — mapped onto the existing layers/sliders under the hood, plus an
**Advanced** toggle that reveals today's full 8-tab layer surface unchanged.

**Alternatives considered.**
- *Reorganize the existing tabs only.* Less disruptive, but the depth (8 tabs × grouped sliders)
  is intrinsic; reshuffling doesn't reduce it for the common edit.
- *Replace tabs entirely with a flat list.* Loses power users and the semantic-layer model that's
  a genuine strength; progressive disclosure keeps both.

**Trade-offs.**
- *Two surfaces to maintain.* Simple mode is a *view* over the same parameter model, not a second
  pipeline, so the cost is mapping + layout, not duplicated logic. Risk: drift between modes —
  mitigate by deriving Simple controls from the same slider definitions (`config.py`).
- *Mode confusion.* A user in Simple mode may not realize Advanced exists. Mitigation: a single,
  always-visible toggle (shipped) and a one-time hint (deferred — the toggle alone reads clearly
  enough in practice; revisit if real usage shows otherwise).

**Recommendation.** Ship Simple mode after §6.1–6.4, since it's the most layout-heavy and benefits
from the targeting/fan-out work landing first. **Moderate effort, low risk, high first-run
impact.** ✅ Shipped 2026-06-23, see Phase 4 in §7.

### 6.6 Unified, predictable undo

**Decision.** Converge the three undo scopes (document history, mask stack, and the
currently-uncovered crop/collection changes — UX_IMPROVEMENTS #24) onto one timeline, or, if full
convergence is too risky, make the boundary explicit and visible.

**Trade-offs.** Full unification touches shared infrastructure and is the riskiest item here. But
*every* preview-then-commit interaction above (recipes, fan-out, auto-enhance) leans on Ctrl+Z as
the safety net — if undo is unpredictable, "try it risk-free" isn't actually risk-free.
**Recommendation:** treat predictable undo as a *prerequisite* for shipping auto-enhance-on-open
and fan-out with confidence. If unification slips, ship a labeled split control ("Undo Edit" /
"Undo Mask") as an interim so the boundary is at least legible.

### 6.7 Local healing / spot retouch (the one structural bet)

**Decision.** Add a spot/heal brush for blemish and stray-object removal — the top
"retouch" gap (ROADMAP P0, "Manual retouching / healing").

**Trade-offs.** This is the only genuinely *new engine* proposed, and it's heavy
(content-aware fill, a new brush tool, its own undo integration). Everything else in this document
is flow/surfacing over existing capability; this is capability. **Recommendation:** scope it as a
separate, later track. "Retouch with minimal resistance" is *mostly* delivered by §6.1–6.6 over
the existing semantic + smoothing tools; healing deepens "retouch" but isn't on the critical path
to low resistance. Sequence it after the funnel is smooth, and design its brush/undo to mirror the
existing mask-edit and AI-Select tooling so it doesn't invent a third interaction model.

### 6.8 Never freeze the window

**Decision.** Extend the existing worker-pool pattern to the remaining synchronous heavy ops —
RAW decode on open, Fix Eyes, multi-image Auto-WB (UX_IMPROVEMENTS #23).

**Trade-off.** Pure plumbing, no design surface, but a *frozen window is maximal resistance* —
it's the difference between "this app is fast" and "this app hangs." Low design risk; medium
implementation care (the established pinned-task + job-id pattern is the template). **Ship
alongside the funnel work**, since auto-enhance-on-open especially must not block the open.

---

## 7. Sequencing

Ordered by leverage-per-effort and dependency. Each phase is independently shippable.

**Phase 0 — Mask-quality foundation (✅ largely landed, 2026-06-23).**
- §6.0 RMBG-2.0 matting + person-gated foreground + Mask DINO memoization + cache v14. *Done.*
- Remaining slice: make the per-person features (Phase 1–2) read mask quality from the combined
  §6.0 result and degrade gracefully when RMBG-2.0 isn't installed (it's opt-in/non-commercial).

**Phase 1 — Targeting & trust (unblocks the group-portrait win). ✅ Done, 2026-06-23.**
- §6.2 badge + active-face highlight — `face_scope_label` + `set_face_highlight` already ship
  (`main_window.py:3881`, `_update_face_scope_indicator` at `~7056`).
- §6.8 async for open/Fix-Eyes/Auto-WB — all three already run via dedicated `QRunnable` tasks
  (`ImageLoadTask`, `FixEyesTask`, `WBAutoAITask`); nothing left synchronous on the GUI thread.
- §6.6 undo — turned out to already be **fully unified**, not just an interim labeled split:
  `_capture_document_state` (`~5389`) covers global/selective params, color, active face, face
  profiles, compare mode, **framing/crop**, mask state, and mask adjustments on one
  `_document_history` timeline; `undo_mask_btn`'s own tooltip confirms ("mask edits share one
  timeline with every other edit"). §6.6 as designed is superseded by this — no further work
  needed there.

**Phase 2 — The group-portrait headline. ✅ Done, 2026-06-23.**
- §6.2 canvas-click face targeting — a plain click on a detected face's box now retargets Face
  Target directly, no dropdown needed. Implemented as the lowest-priority gesture in
  `ImagePreviewLabel.mousePressEvent`/`mouseReleaseEvent` (`main_window.py:~1479-1630`): checked
  only after every modal tool (WB-pick, AI Select, crop, paint) has had first claim on the click,
  and distinguishes a zoomed-in *tap* from a *pan-drag* via a `_pan_moved` flag so panning is
  unaffected. `_update_face_scope_indicator` now also publishes all faces' hit-test rects via
  `set_face_click_targets`, not just the active one's highlight.
- §6.3 "Apply to all faces" fan-out — new `apply_to_all_faces_btn` (visible only with >1 face)
  calls `_apply_active_face_to_all()` (`~7157`): copies the active face's Face/Skin/Eyes/Lips/
  Hair/Person slider values onto every other detected face's stored profile, uniformly (per the
  §5 trade-off — same values, not per-face-adapted), one undo step, each face keeps its own mask.
  Faces not yet visited this session get a placeholder profile exactly like the existing
  paste-settings path already does — safe by construction, since `_restore_face_profile` always
  applies stored `selective_params` before separately checking whether masks need a fresh
  segmentation (verified directly: an unvisited face's slider correctly read the fanned-out value
  once a fresh segmentation completed for it).
- §6.3 silent paste mismatch — turned out **already fixed**: `_paste_settings` (`~5613`) already
  shows a `QMessageBox` confirmation naming the face-count mismatch before pasting. No work needed.
- Face roster (thumbnail chips) deferred — canvas-click targeting covers the primary "who do I
  click" need; the roster's incremental value (at-a-glance "has everyone been touched") didn't
  justify holding up the headline items. Candidate for a later, smaller follow-up.
- Verified on `_MG_3976.CR2` (2 detected faces): click-to-target at default zoom, click-to-target
  while zoomed in (vs. a real pan-drag, confirmed not to misfire), fan-out's stored value
  surviving a fresh per-face segmentation, and Undo restoring the pre-fan-out value. Full suite:
  193 passed, same 2 pre-existing unrelated failures throughout Phase 1 and 2.

**Phase 3 — Starting point & risk-free trying. ✅ Done, 2026-06-23.**
- §6.1 recipe strip + **Enhance** primary action — `_build_recipe_strip()` adds an always-visible
  row above the canvas (`main_window.py:~3598`, `~4593`): "Enhance" (primary-styled, applies the
  first guided recipe) followed by one button per recipe, plus a "Recipes..." button that still
  opens the full modal for browsing descriptions. Every button commits through the **existing**
  `_apply_preset_with_preview` mechanism (already built for the preset browser, previously unused
  by recipes) — live on the canvas immediately, with a Keep/Discard bar, so trying a recipe is
  truly reversible with one click, not just "remember to hit Ctrl+Z."
- Found and fixed a real bug while wiring this up: `open_recipe_dialog` previously called
  `_apply_preset_state` directly, which never pushes a closing `_push_document_history()` —
  applying a recipe left no clean undo checkpoint of its own (a later edit's undo would silently
  bundle the recipe in with it). Routing recipes through `_apply_preset_with_preview` fixes this
  for both the new strip and the old modal.
- Found and fixed a second, unrelated pre-existing bug in the same area: `self.preset_preview_label`
  was assigned to *two different widgets* (the browser's thumbnail swatch, then the Keep/Discard
  bar's text label) — the second silently shadowed the first, so the preset browser's thumbnail
  preview never actually updated on hover/select. Renamed the browser's to
  `preset_browser_thumbnail`; both now work independently.
- §6.1 auto-enhance-on-open — new `auto_enhance_on_open` preference (off by default,
  `PreferencesDialog`, `~765`), applied in `_finish_load_image_path` (`~8233`): only fires when
  the image has **no saved override** (an already-customized image is never touched), and lands
  as its own history entry *after* the neutral baseline so it's one Ctrl+Z away as designed.
- Verified directly: Enhance/recipe buttons apply live with no history change until Keep/Discard,
  Discard restores the exact prior value, Keep pushes exactly one entry and Ctrl+Z cleanly
  reverts it; auto-enhance is off by default, applies and is undoable when on, and correctly
  skips an image that already has saved settings. Full suite: 193 passed throughout, same 2
  pre-existing unrelated failures.
- (§6.6 full undo unification dropped from this phase — already done, see Phase 1.)

**Phase 4 — Friction polish. ✅ Done, 2026-06-23.**
- §6.4 Quick Export — ✅ done. New primary-styled "Quick Export" button (`main_window.py:~3504`,
  `quick_export_image` `~10188`) writes immediately with the last-used format/quality/resize/
  metadata/destination (session-remembered; defaults to the image's own folder + jpeg/92 the
  first time), confirming overwrite the same way the dialog already did. `export_image`
  (renamed "Export As..." in the UI) is unchanged in behavior, just refactored to share
  `_render_and_save_export`/`_remember_export_options` with the new path.
- §6.4 one-action collection export — ✅ done. New "Export All" button next to the existing
  "Export..." (`~3761`, `_quick_export_collection` `~5374`) reuses the same remembered
  `_last_batch_options()` the dialog already persists, skipping straight to dispatching the
  background batch job — including the existing overwrite-warning check
  (`_confirm_collection_overwrite`, factored out of `_export_collection`, unchanged logic). Falls
  back to the full `CollectionExportDialog` if no destination has been remembered yet, since
  guessing a folder for a potentially large batch write is the wrong place to save a click.
- §6.4 "Preferences home for defaults" — turned out **already done**: `PreferencesDialog` with
  acceleration mode + default export folder already existed (this is also where the §6.1
  auto-enhance toggle landed). No further work needed there.
- Verified directly (isolated `/tmp` destinations, never touching the real source folder):
  Quick Export writes a real file using remembered settings, asks before overwriting and honors
  both answers, and the source folder fallback computes correctly when nothing's been exported
  yet. Collection quick-export correctly falls back to the dialog with no remembered destination,
  and correctly skips the dialog and dispatches directly once one exists. Full suite: 193 passed
  throughout, same 2 pre-existing unrelated failures.
- §6.5 Simple / Advanced inspector mode — ✅ done. A "View: Simple / Advanced" toggle
  (`main_window.py:~3960`) shows either a new flat, curated panel (`SIMPLE_CONTROLS`, ~2926;
  `_build_simple_panel` ~4506) or today's unchanged 8-tab surface, mutually exclusive, persisted
  to `_preferences["inspector_mode"]` (default "advanced", so existing muscle memory isn't
  disrupted). True to the design — Simple is a *view*, not a second pipeline: each Simple slider
  is a standalone widget that forwards every change into the real `_sliders[layer][key]` (the
  only thing `_all_params()`/rendering/export ever reads), and a new `_sync_simple_controls()`
  hooked into the render-settled callback (`_on_preview_render_finished`) pulls the Simple panel's
  displayed values back into sync after *any* change — necessary because most existing code
  (presets, recipes, undo/redo, face-switch) sets sliders with `blockSignals`, so a naive
  signal-only mirror would silently miss all of them.
  - Control list adjusted from the original design: dropped "Teeth Whiten" (no teeth-whitening
    feature exists yet — ROADMAP.md P0, "Manual retouching / healing") in favor of global
    Sharpness, an equally high-use control that already exists. Final eight: Exposure, Warmth
    (relabeled global Temperature), Sharpness, Skin Smooth, Blemish Fix, Eye Brighten, Eye
    Whiten, Background Blur.
  - Verified directly: toggle mutual exclusion and persistence (including a *fresh* window
    instance picking up the saved preference from disk); a Simple slider drag correctly drives
    the real slider; a direct real-slider change and a `blockSignals`-applied recipe both
    correctly propagate back into the Simple panel after the render settles; Undo behaves
    identically to dragging the real slider and the Simple panel stays in sync afterward. Full
    suite: 193 passed throughout, same 2 pre-existing unrelated failures.

**Phase 4 status: ✅ all three items done.**

**Phase 5 — Capability deepening (separate track).**
- §6.7 spot/heal brush.

Rationale for the order: targeting (Phase 1–2) is the highest-leverage, lowest-architecture work
and delivers the group-portrait differentiator first. Starting-point and export polish (Phase 3–4)
build on a now-trustworthy preview/undo base. Healing (Phase 5) is decoupled so it never blocks
the flow wins.

---

## 8. How we'll know it worked (success signals)

Lightweight, observable proxies — no formal study required:

- **Clicks-to-first-good-result** drops: opening an image and getting to a satisfactory edit takes
  visibly fewer interactions (auto-enhance + recipe strip).
- **Targeting errors vanish**: "edited the wrong person" becomes structurally impossible (active
  face is always the clicked, outlined one).
- **Group edits are O(1), not O(N)**: making all faces consistent is one gesture, confirmed by a
  toast naming the face count.
- **Export is one click** for the repeat case; the dialog appears only when the user chooses
  "Export As…".
- **No freezes**: opening a large RAW, Fix Eyes, and multi-image Auto-WB never lock the window.
- **Undo is predictable**: Ctrl+Z after any commit (recipe, fan-out, slider, mask) does the
  obvious thing.

---

## 9. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Auto-enhance-on-open surprises users | Med | Off by default; first-run opt-in; before/after toggle; one Ctrl+Z to remove |
| Live recipe preview janks on large images | Med | Reuse proxy render; debounce; fall back to focus/click-hold preview |
| "Apply to all" overwrites hand-tuned faces | Med | Single undo step; confirming toast; optional "skip already-edited" |
| Canvas-click targeting collides with other tools | Low | Gate behind "no modal tool owns the click," like existing double-click-reset |
| Quick Export writes to a surprising place | Med | Path-naming toast; always-present Export As…; honor overwrite prompts |
| Undo unification destabilizes shared state | **High** | Phase it; ship labeled-split-undo interim; treat as prerequisite, not afterthought |
| Simple/Advanced modes drift | Low | Derive Simple controls from the same `config.py` slider model, not a copy |
| Heal brush scope-creeps the whole effort | Med | Hard-separate as Phase 5; mirror existing brush/undo interaction model |
| Group features feel worse where RMBG-2.0 isn't installed | Med | §6.0 backend falls through to MODNet/MediaPipe/heuristic; features benefit from RMBG but never hard-depend on it; System Check surfaces readiness |
| RMBG-2.0 non-commercial license reaches end users | **High** (if shipped commercially) | Keep download explicit + dependency optional; gate behind `.[rmbg]` extra; surface the license in System Check / docs; don't bundle the weights |

---

## 10. Open questions

1. **Auto-enhance default strength.** ✅ Resolved by implementation: uses whichever recipe is
   first in `_guided_recipes()` (currently "Natural Portrait," the gentlest one), same recipe
   "Enhance" applies. If a different/dedicated gentler recipe is wanted later, both share one
   source of truth (`recipes[0]`) so it's a one-line change.
2. **"Apply to all" semantics.** Confirm uniform-values-first (§5); is "skip already-edited faces"
   wanted as an option, and if so, what's its default? Still open — shipped uniform-values-only,
   no skip option.
3. **Simple-mode control set.** ✅ Resolved by implementation, see §6.5 — final eight: Exposure,
   Warmth, Sharpness, Skin Smooth, Blemish Fix, Eye Brighten, Eye Whiten, Background Blur
   ("Teeth Whiten" dropped, no such feature exists yet).
4. **Quick Export destination on first use.** ✅ Resolved by implementation: remembers globally
   per session (`_last_export_dest_dir`), defaulting to the image's own folder until the first
   export (dialog or Quick) sets it — no confirmation prompt.
5. **Undo unification depth.** Full single-timeline, or is the labeled-split-undo interim
   acceptable as the *end state* given the refactor risk?
6. **RMBG-2.0 licensing posture.** Is this app ever distributed commercially? If so, the
   non-commercial RMBG-2.0 weights can't be the *default* edge-quality path — we'd need a
   commercial license or a permissively-licensed matte model as the shipped default, with RMBG as
   an opt-in upgrade. This decision gates how much of the §6.2/§6.3 "clean edges" story we can
   promise out of the box.

---

*This document describes a target experience and the reasoning behind it. It is intentionally
implementation-light; each phase should get its own focused plan before building, validated
against the current code (the UI surface drifts).*
