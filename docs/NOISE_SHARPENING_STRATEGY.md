# Region-Aware Noise & Sharpening — Strategy Guide

How to apply noise reduction and sharpening *differently to different parts of a portrait*, and
how to do it in **this app specifically** (which controls, what order, what values). Grounded in
the actual pipeline — file:line references point at the real behavior; verify against current
code before relying on a detail, since the pipeline drifts.

---

## 1. The core principle

Noise reduction and sharpening are **antagonistic** and **region-dependent**. The single biggest
mistake in detail work is applying either one globally at full strength:

- **Sharpening amplifies noise.** Any unsharp/high-pass boost raises *all* high-frequency
  content — real edge detail *and* sensor noise alike. Sharpen a noisy flat area (skin, sky) and
  you make the noise crunchy and obvious.
- **Noise reduction destroys detail.** Any smoothing that removes noise also softens real
  texture. Denoise eyelashes, iris detail, or hair and they go mushy.

So the whole game is **spatial separation**: denoise the regions that are *supposed* to be smooth
and carry no wanted detail (skin, out-of-focus background, flat shadows, sky), and sharpen only
the regions that carry *wanted* fine detail (eyes, eyebrows/lashes, hair strands, lip edges,
fabric/jewelry texture). Never the same treatment everywhere.

The classic portrait hierarchy, from "denoise hard / never sharpen" to "sharpen hard / never
denoise":

```
  smooth ←──────────────────────────────────────────────→ sharp
  sky / OOF background   skin   neutral   hair   lips   eyes
  (denoise, never        (denoise,        (light  (light (sharpen
   sharpen)               soften)          sharpen) sharpen) hardest)
```

---

## 2. How this app applies detail (the mechanics that make region-aware work)

Three properties of the pipeline are what let you target detail by region. Understanding them is
what turns the sliders from guesswork into a repeatable method.

**a) Global runs first, denoise-before-sharpen, then per-region layers composite on top.**
`process_all_layers` runs `process_global` first, then composites the masked layers in this order
(`MASK_ORDER`, `config.py`): `background → subjects → person → hair → skin → face → eyes → lips`.
Inside `process_global` (`core/processing.py:~1249`), **noise reduction is applied before
sharpening** — the correct order, so you're not sharpening noise you're about to remove. This
means: set your *global* noise floor first, then add *local* sharpening per region on top.

**b) Global sharpening is already edge-aware (it won't sharpen flat noise).**
`_apply_unsharp_mask` (`core/processing.py:127`) builds an **edge mask** from the local gradient
and only boosts detail where an edge actually exists; flat areas (clean skin, sky) are left
alone. The **Sharpen Masking** slider *is* that threshold — higher Masking = sharpening retreats
further to only the strongest edges. This is why global Sharpness is safer here than a naive
unsharp mask, but it's still not a substitute for *not denoising the eyes*.

**c) Masks confine each layer's edit to its region, and skin gets extra protection.**
Each layer (Skin, Eyes, Hair, Background…) only affects its masked pixels, so a Skin-layer smooth
never touches the eyes. The Skin layer additionally builds a **protection mask**
(`_build_skin_protection_mask`, `~840`) so its smoothing pulls back around the eyes/brows/lip
edges even within the skin region. Net effect: you can be aggressive on Skin smoothing without
melting the features embedded in it.

---

## 3. The region-by-region playbook

Which control to reach for, per region, with conservative starting values. (Controls that exist
in the app today are named exactly; see §5 for what's global-only.)

| Region | Goal | Control(s) in this app | Starting point |
|---|---|---|---|
| **Whole image** | Set the noise floor *first* | Global **Luminance NR** (`noise_red`), **Color NR** (`color_noise_red`) | Lum 15–30, Color 0–15 (raise Color for blotchy chroma) |
| **Skin** | Smooth tone, kill remaining grain, *no* sharpening | Skin **Noise Reduc.**, **Smooth**, **Blemish Fix**; keep Skin **Clarity** ≤ 0 | NR 0–25, Smooth 20–35, Blemish 15–30, Clarity −10…0 |
| **Eyes** | Sharpest region in the frame | Eyes **Sharpen**, **Iris Pop**, small **Clarity** | Sharpen 25–45, Iris Pop 10–25, Clarity 5–15 |
| **Hair** | Define strands without crunch | Hair **Clarity**, **Shine**; light global Sharpness | Clarity 8–20, Shine 5–15 |
| **Lips** | Clean edges, smooth fill | Lips **Smooth** (fill), rely on global edge-aware Sharpness for the edge | Smooth 10–20 |
| **Face (overall)** | Gentle structure on bones/jaw, not pores | Face **Sharpness**, Face **Clarity** | Sharpness 5–15, Clarity 5–10 |
| **Background** | Suppress noise, optionally separate subject | Background **Noise Reduc.**, **Blur**, **Clarity** (often negative), **Dehaze** | NR 0–40, Blur 0–30, Clarity −10…+5 |
| **Subjects / Person** | Coarse mid-detail lift on people vs. scene | Subjects/Person **Clarity**; Person **Noise Reduc.** | Clarity 5–12, NR 0–20 |

Reading the table as a method: **denoise globally, then add Clarity/Sharpen where you want
detail and Smooth/Blur where you don't.** The two ends — Eyes (sharpen hardest) and
Skin/Background (smooth, never sharpen) — are where the spatial separation pays off most.

### Why these specific moves

- **Eyes are the anchor.** In a portrait the viewer goes to the eyes; they should be the
  sharpest thing in the frame even if nothing else is touched. Eyes **Sharpen** + **Iris Pop** on
  the eye mask is safe *because* it's masked — it won't drag the surrounding skin noise up with
  it.
- **Skin is the opposite anchor.** Skin should never receive sharpening. If global Sharpness is
  pushed, keep Skin **Clarity** at 0 or negative so the skin layer counteracts mid-frequency
  crunch. Smoothing here doubles as regional noise reduction (see §5).
- **Hair wants Clarity, not Sharpness.** Hair detail is mid-frequency (strand separation), which
  **Clarity** (local contrast) renders more naturally than a fine-radius unsharp mask, which just
  makes individual noisy pixels pop.
- **Background is a noise sink.** Out-of-focus background carries no wanted detail, so it's the
  safest place to smooth hard (Blur) and the *worst* place to add Clarity/Dehaze blindly — Dehaze
  and positive Clarity both re-amplify background noise.

---

## 4. Recommended workflow order

Order matters because each step changes the noise/detail balance the next step sees.

1. **Global noise floor — already seeded for you.** Luminance NR, Color NR, and Sharpness are
   all auto-suggested the moment an image opens, from that image's own measured noise (§5.1) —
   you're adjusting a real starting point, not starting from 0. Don't over-denoise here — you'll
   recover crispness per-region, but you can't recover detail you smoothed away globally.
2. **Tune global Sharpness Radius/Masking** for the *kind* of detail in the shot. Lower Radius =
   fine detail (lashes, fabric weave); higher Radius = broad structure (jawline, hair masses).
   Raise **Masking** to keep sharpening off flat skin/sky.
3. **Sharpen the eyes** (and only then judge whether the face needs any global sharpening at all —
   often the eyes alone are enough).
4. **Protect/soften skin** — Skin Noise Reduc. is *also* already auto-seeded from skin's own
   mask (§5.2); nudge it (or hit its **Auto** button after switching faces) before Smooth,
   Blemish, Clarity ≤ 0.
5. **Add hair/lip detail** (Clarity, Shine, Smooth).
6. **Treat the background last** — Background Noise Reduc. is auto-seeded too; add Blur on top
   if you also want to suppress structure, not just grain.
7. **Output sharpening at export** (§6) — the final, resolution-aware pass.

---

## 5. Per-region noise reduction (Skin / Background / Person)

**Implemented 2026-06-23.** Skin, Background, and Person each have their own **Noise Reduc.**
slider (`noise_red` in `SKIN_SLIDERS`/`BACKGROUND_SLIDERS`/`PERSON_SLIDERS`, `config.py`), wired
through the same `_apply_luma_chroma_denoise` the Global layer uses
(`process_skin_layer`/`process_background_layer`/`process_person_layer`,
`core/processing.py`). Each runs **before** that layer's other detail work (Smooth/Blemish on
Skin, Blur/Dehaze on Background, Clarity on Person) — denoise first, then build texture/contrast
on the cleaner result, the same ordering principle as the global layer. Defaults to 0, so it's
purely additive: existing presets/projects without the key are unaffected.

This means you now have a real choice between **global** and **regional** denoising, and they
compose:

- **Global Luminance/Color NR** sets the noise floor everywhere — use it for "the whole sensor
  capture is grainy."
- **Skin/Background/Person Noise Reduc.** lets you denoise *harder* in one region without paying
  that cost everywhere else — e.g. clean up shadowed skin without softening already-clean hair,
  or denoise a noisy background without touching the in-focus subject.
- They stack: global runs first (in `process_global`), then each region's own slider denoises
  *again*, additively, within its mask. Keep global modest and let the regional sliders do the
  heavy lifting where it's actually needed — that's usually less total smoothing than cranking
  the global slider until the worst region looks clean.

**Still global-only:** Eyes, Hair, Lips, Face, and Subjects have no `noise_red` of their own —
for those, the de-facto pattern from before still applies (global NR + masked Sharpen/Clarity to
recover detail). Eyes/Hair/Lips are exactly the regions you don't want to denoise anyway (§1), so
this isn't really a gap; Subjects (the multi-person, non-Person-Target scoped layer) is the one
plausible future addition if a use case calls for it.

**Sharpening Radius/Masking are still global-only** (`sharpen_radius`/`sharpen_masking` on
Global) — only the global edge-aware unsharp mask has tunable radius/threshold; the per-region
Sharpen/Clarity controls (Eyes, Face) decide *how much*, not *what kind*.

### 5.1 What gets auto-suggested, and how (important: what it does and doesn't measure)

**On every image open**, `_apply_auto_global_suggestions` seeds:
- Global **Luminance NR**, **Color NR**, and **Sharpness** — from `suggest_global_auto_values`.
- **Skin / Background / Person Noise Reduc.** — each from its *own mask*, via
  `suggest_region_noise_red`, not the whole-image number. A shadowed background or a smoother,
  cleaner-skinned face can get a genuinely different regional reading than the frame average.

All of this is **measured from the decoded pixels**, the same Laplacian-of-Laplacian estimator
(Immerkjær 1996) as before — **not** from EXIF/ISO/camera metadata. It reacts to whatever noise
is actually visible in the image, regardless of what produced it (high ISO, a deep shadow push,
even synthetic noise): it measures the *consequence* of sensor+ISO+lighting on the pixels, not
those factors themselves.

**Color NR specifically** is only suggested when chroma noise is *measurably worse than*
luminance noise (`estimate_chroma_noise_sigma(img) > estimate_noise_sigma(img)`, by a margin) —
the diagnostic signature of color blotches in shadows, as opposed to ordinary grain that affects
luminance and chroma about equally (which `noise_red` alone already handles). A clean image, or
one with uniform grain, correctly gets `color_noise_red: 0`.

**Region estimates require a mask.** Skin/Person need at least one detected face; Background's
scene-wide mask is available even with zero faces. No mask → that region's auto-suggestion is
skipped (stays 0), not replaced by a whole-image guess — denoising a region that wasn't detected
would be guessing, not measuring.

**A saved override always wins.** If you've already edited and saved settings for an image, that
saved value is applied *after* the auto-suggestion and overwrites it (`_finish_load_image_path`)
— so your own choice persists across re-opens instead of being replaced by a fresh auto-guess
every time.

### 5.2 Re-running the estimate on demand (the "Auto" buttons)

Each auto-seeded slider also has its own **Auto** button (`AUTO_SUGGEST_SLIDERS`) for
re-measuring later — useful after switching Face Target, since the per-face Person/Skin masks
differ per person. Critically, **Skin/Background/Person's Auto button measures that layer's own
mask**, not the whole image — `_auto_correct_slider` branches on this, so clicking Skin's Auto
re-runs `suggest_region_noise_red` against the *current* skin mask (e.g. after switching to a
different face), while Global's Auto buttons still use the whole-image estimate. They are
deliberately not the same calculation, even though they sit on a slider with the same name.

---

## 6. Output (export-time) sharpening — the separate final pass

The Sharpness slider is tuned for the **working/preview** resolution. A file you downsize on
export (e.g. 24MP → 2048px for web) has denser detail per pixel and needs a *different* final
sharpening, applied **after** resize. That's the **Output Sharpening** selector
(Off / Low / Standard / High) in the Export and Collection-Export dialogs
(`apply_output_sharpening`, `core/processing.py:~166`), calibrated to the export's actual long
edge.

Rule of thumb:
- **Web / social (downsized a lot):** Standard or High — downsizing softens, so the final pass
  matters most here.
- **Full-resolution export (little/no downsize):** Low or Off — the working-res Sharpness slider
  already did the work; a second full pass risks halos.
- **Print:** Standard — print needs slightly more than screen to survive the ink/paper.

Output sharpening is also edge-aware (same `_apply_unsharp_mask` engine with a tight threshold),
so it won't re-crunch the flat regions you denoised earlier.

---

## 7. Cheat sheet

```
NOISE  → apply where there is NO wanted detail:
         Global Luminance NR (whole-image floor, set FIRST)
         Global Color NR     (only if chroma blotches remain)
         Skin Noise Reduc.   (regional NR for skin -- denoise before Smooth/Blemish)
         Background Noise Reduc. (regional NR independent of how much Blur is used)
         Person Noise Reduc. (regional NR for one person's full body)
         Skin Smooth / Background Blur (heavier regional smoothing, not just noise)

SHARPEN → apply where there IS wanted detail:
         Eyes Sharpen + Iris Pop   (sharpest region — the anchor)
         Hair Clarity              (strand definition, mid-freq)
         Face Sharpness (light)    (bone structure, not pores)
         Global Sharpness          (edge-aware; Masking keeps it off flat skin/sky)

NEVER  → sharpen skin or sky; denoise eyes or hair.
ORDER  → global denoise → global sharpen radius/masking → eyes → skin (NR then Smooth/Blemish) → hair/lips → background (NR then Blur) → export output-sharpening.
```

---

*This guide describes photographic strategy mapped onto the app's current controls. Where it
names a slider or function, that control exists today; §5 flags the one real gap (per-region
noise reduction is global-only). Treat starting values as conservative defaults to taste against,
not fixed recipes.*
