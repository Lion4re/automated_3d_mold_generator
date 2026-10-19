# Undercut-aware rigid molds remain the open gap

This project's core product is now a commodity: an STL in, a two-piece planar-split rigid mold out, with keys, a sprue and vents. By October 2026 at least ten browser apps do this ([Meshcast](https://meshcast.app/mold), [splicestl](https://splicestl.com/mold/), [jocam3d](https://jocam3d.com/en/tools/stl-to-mold), [Akritio](https://akritio.com/)), so do about a dozen Blender add-ons ([MoldForge](https://extensions.blender.org/add-ons/moldforge/)), and so do dozens of tiny GitHub repos. Several of those repos use the same trimesh + manifold3d stack ([juppeee/moldmaker](https://github.com/juppeee/moldmaker), [Mathew3000/moldgen](https://github.com/Mathew3000/moldgen)). No tool, open or closed, ships all of the following under a permissive license and with ongoing maintenance:

- automatic parting-direction search;
- non-planar parting surfaces;
- undercut-driven splitting into more than two rigid pieces;
- vent placement grounded in physics;
- material-aware defaults.

Competitors admit the failure themselves. jocam3d says "figurines, faces, anything with deep detail — lock in every direction" ([jocam3d](https://jocam3d.com/en/tools/stl-to-mold)), and splicestl "may produce unusable geometry on highly organic forms" ([splicestl](https://splicestl.com/mold/)).

The academic state of the art has the algorithms but almost no released code. IST Austria holds active patents on FlexMolds, Metamolds and CoreCavity, running to 2036-2041 ([US11931922B2](https://patents.google.com/patent/US11931922)). That pushes an open tool toward older, unpatented computational geometry: Ahn et al. monotonicity, Bose-van Kreveld-Toussaint gravity filling, and Gupta's accessibility-driven multi-piece molds. All of these map onto trimesh ray queries, numpy and manifold3d Booleans at low to medium effort.

The most valuable next steps for this project's repo, in order:

1. Finish, test and document the `gating.py` vent planner based on local maxima plus persistence (a prominence-based version now exists in the uncommitted working tree). The repo's "pour perpendicular to the pull" convention makes this provably sufficient for undercut-free parts ([AMB21 postprint](http://openportal.isti.cnr.it/data/2021/456702/2021_456702.postprint.pdf)).
2. Harden the direction search.
3. Ship material presets with sourced defaults.
4. Build toward multi-piece molds as the long-term moat.

Evidence labels used below, carried over from the research notes:

| Label | Meaning |
|---|---|
| (TDS) | Manufacturer data sheet. |
| (TDS-snippet) | Manufacturer number seen only in a search extract. |
| (secondary) | Reputable secondary source. |
| (snippet-only) | Seen only in a search summary. |
| (anecdotal) | Forum, blog or maker report. |
| (competitor, AI-assisted) | Mold-generator vendor guides, some admittedly AI-drafted. |
| (unverified) | Claim could not be confirmed. |
| (inference) | A researcher's derivation, not a sourced number. |

## 1. Summary of the landscape: two-piece molds are saturated, undercut-driven multi-piece is not

**Activity is high, but every individual project is small.** About 55 candidate repos were read. Almost all have 0-22 stars, were created in 2025-2026, and many have a single commit. The most-starred standalone generator is [S0mbraD/3DPrint_MoldGen](https://github.com/S0mbraD/3DPrint_MoldGen) with 22 stars, and its feature claims are unverified. With 6 stars, this project's [automated_3d_mold_generator](https://github.com/Lion4re/automated_3d_mold_generator) is already in the top tier.

The tech stack has converged, so the stack alone does not differentiate:

- Python tools use trimesh + manifold3d ([GreenShoeGarage](https://github.com/GreenShoeGarage/InjectionMoldFromSTL), [juppeee](https://github.com/juppeee/moldmaker), [Mathew3000](https://github.com/Mathew3000/moldgen)).
- Browser tools run Manifold in WebAssembly ([splicestl](https://splicestl.com/mold/), [jocam3d](https://jocam3d.com/en/tools/stl-to-mold), [matta174/mold-maker](https://github.com/matta174/mold-maker)).

Commercial web apps now treat this as the baseline feature set: upload, auto-orient, flat-plane split, pins, funnel, vents, wall and clearance settings ([Meshcast](https://meshcast.app/mold), [PrintPal](https://printpal.io/tools/mold-box-generator), [Mold Studio](https://moldstudio.app/)). Several of these apps appeared or were updated in 2026, and several 2026 READMEs show AI-assisted authoring (for example [FreeCAD-Molding-Wizard](https://github.com/ryankembrey/FreeCAD-Molding-Workbench) states it "was written by AI"). Expect a fast-growing long tail of short-lived clones.

**There is a gap, but it is narrower than "nobody does multi-piece."** A few tools already reach into it:

- [HansTydecks/Concrete-Mold-Creator](https://github.com/HansTydecks/Concrete-Mold-Creator) auto-splits with a binary space partition. For every piece it searches a pull direction and counts ray-cast-blocked faces as undercuts. It is one commit old and unlicensed.
- [MoldForge](https://github.com/Plesuro/MoldForge) in Blender picks the X or Y axis with the fewest undercuts. It offers 2-4 radial "pie-slice" pieces and a ray-cast contoured parting option.
- [Akritio](https://akritio.com/pricing) sells 4- and 6-part radial molds.
- [jocam3d](https://jocam3d.com/en/tools/stl-to-mold) tries about 80 directions and switches to a curved parting surface.

What is genuinely missing is a combined, permissively licensed package with the following. The open-source researcher read about 55 repos and found none; the nearest are [HansTydecks](https://github.com/HansTydecks/Concrete-Mold-Creator) (unlicensed), [matta174](https://github.com/matta174/mold-maker/blob/main/docs/competitive-analysis.md) (noncommercial) and [VcMoldCreator](https://github.com/vjvarada/VcMoldCreator) (license unclear):

- automatic parting direction and parting surface;
- undercut-driven splitting into more than two pieces;
- draft handling;
- vent placement that follows the cavity's topology.

The commercial researcher reached the same conclusion: nobody claims undercut-driven multi-piece decomposition with non-planar parting for organic shapes, and several vendors steer organic models to silicone instead ([Meshcast guide](https://meshcast.app/guides/stl-to-mold), [Action BOX](https://mold.actionbox.ca/)). This is an inference from vendor pages that were not tested hands-on.

**Our differentiation should rest on five levers, not on the geometry kernel.**

1. **License.** The most visible open web tool, matta174/mold-maker, is PolyForm Noncommercial and "not an OSI-approved open-source license" ([README](https://github.com/matta174/mold-maker)). MoldForge and Mathew3000 are GPL-3.0. HansTydecks has no license, and gachetiberiu's app is "UNLICENSED" ([silicone-mold-app](https://github.com/gachetiberiu-WIZ/silicone-mold-app)). An MIT library is unusual here.
2. **Distribution.** PyPI lookups for moldgen, moldmaker, mold-maker, stl2mold and similar names returned nothing ([PyPI](https://pypi.org/pypi/pymold/json)). No project is a pip-installable library.
3. **Host independence.** MoldForge and MAX Mold Pro need Blender 5.1+ ([MoldForge](https://extensions.blender.org/add-ons/moldforge/), [MAX Mold Pro](https://www.maxmold.com.br/en/)). Fusion needs a paid extension for mesh-to-solid conversion ([Fusion lure add-in](https://github.com/TheJoeJep/AutodeskFusionLureMoldGenerator)). Meshmixer is "no longer in development" ([Autodesk](https://apps.autodesk.com/FUSION/en/Detail/HelpDoc?appId=4108920185261935100&appLang=en&os=Win64)).
4. **Focus on rigid molds cast directly** for wax, plaster, concrete, soap and chocolate. Most in-host tools are silicone-first ([FormArmor Pro](https://superhivemarket.com/products/formarmorpro), [FreeCAD-Molding-Wizard](https://github.com/ryankembrey/FreeCAD-Molding-Workbench)).
5. **Correctness backed by theory,** plus a validation report. Competitors place vents at bounding-box extremities, and a professional caster criticised exactly that ([matta174 ROADMAP](https://github.com/matta174/mold-maker/blob/main/ROADMAP.md); secondhand, because the original Reddit thread link is a placeholder).

matta174's claim that "nobody else in the two-part-mold space is open-source + offline-capable" is now outdated ([competitive-analysis.md](https://github.com/matta174/mold-maker/blob/main/docs/competitive-analysis.md)).

### Patent risk: three active IST Austria patents cover the flagship academic pipelines

The ISTI-CNR and IST Austria line of computational molding work is patented:

| Method | Patent | Assignee | Status | What the claims cover |
|---|---|---|---|---|
| FlexMolds | [EP3301597B1](https://patents.google.com/patent/EP3301597B1/en) | IST Austria | Granted 2023-08-23; anticipated expiry about 2036-10-03 | A simulation-driven loop for merging patches and laying out cuts |
| Metamolds | [US11966666B2](https://patents.google.com/patent/US11966666B2/en) | IST Austria | Expires 2040-09-11 | Surface partitioning by "heuristic removal-effort estimation or volume-based modeling" |
| CoreCavity | [US11931922B2](https://patents.google.com/patent/US11931922) | IST Austria | Expires 2041-03-02 | Splitting, deforming and insetting the object, and optimizing piece orientations and parting lines |

Two further points widen the exposure:

- A European CoreCavity application, [EP3810389A1](https://patents.google.com/patent/EP3810389A1/en), also exists.
- The researcher's interpretation, not a legal finding, is that the 2019 volume-aware composite-mold method plausibly falls within the Metamolds patent's "volume-based modeling" language ([US11966666B2 claims](https://patents.google.com/patent/US11966666B2/en)).

A commercial spin-off, AutoMold, uses ISTA-developed mold-design software ([ISTA news](https://ist.ac.at/en/news/how-to-move-a-eureka-moment-to-the-real-world/)), so these patents are commercially live.

One unverified detail matters directly for vents. The classic-algorithms researcher noted that US 11931922 "places vent origins where air bubbles could be trapped and compares orientations by trapped air", but did not check the claims ([USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11931922)). Bose, van Kreveld and Toussaint published "vents at every local maximum, choose the orientation minimizing them" in 1998 ([Utrecht portal](https://research-portal.uu.nl/en/publications/filling-polyhedral-molds/)), two decades before the patent's 2018 priority date. That is reassuring but not legal advice.

The approaches with the least IP exposure are:

- generic ray-accessibility scoring;
- planar or height-field parting;
- mold halves built from depth maps;
- persistence-based vent placement;
- kit features in the style of Shape Cast.

Have counsel read the claims before any feature resembles a full patented pipeline.

### Repo state recorded in the notes, and what has changed

The research notes record this project's code at `src/moldgen/` as follows ([public repo](https://github.com/Lion4re/automated_3d_mold_generator)):

- `parting.py` defines a planar parting plane `dot(p, direction) == offset` and classifies faces as OK, LOW_DRAFT or UNDERCUT.
- In the notes, `best_offset` and the direction search are `NotImplementedError` stubs.
- `gating.py` routes the sprue inside the parting plane. It stands the mold so that one of ±X/±Y points up, with optional vents.

The local working tree on branch `rewrite/moldgen-package` has moved on since then. This was observed directly at writing time and has no public URL:

- `parting.py` now contains a planar direction search. It samples a Fibonacci hemisphere plus inertia axes, does a normal-only coarse scan with local refinement, and runs a grid-binned occlusion test.
- `gating.plan_gating` was a `NotImplementedError` stub when the report was drafted, but a later check of the (still uncommitted) working tree found it implemented: it scores the four ±X/±Y pour orientations, finds air traps by prominence (union-find over a maximum spanning tree of the vertex graph, with the sprue region as a drain), keeps at most a capped number of the most prominent traps, and snaps vents that sit near the parting plane onto it. This is essentially recommendation 1 below; what remains is validation on test casts and documentation.
- `materials.py` holds "placeholder presets": one resin preset (0.5% shrinkage, 6 mm sprue, 1.5 mm vent, 3 mm minimum wall) and PLA at a 50 °C service limit.
- `config.py` defaults to a 0.25 mm clearance and a 1.0° draft threshold.

The recommendations in Section 6 target exactly these hooks.

## 2. Open-source tools table: about 30 projects, no generator above 22 stars, one with real multi-piece automation

Stars and dates come from the GitHub REST API on 2026-10-05, as recorded in the notes. Features come from READMEs; none of these tools were run. "n/r" means not recorded in the notes.

### 2a. Standalone rigid printed-mold generators (direct competitors)

| Project | Stack | License | Last commit | Stars | Output | Automation (parting / undercut / sprue-vents / keys / draft) | Main weaknesses |
|---|---|---|---|---|---|---|---|
| [matta174/mold-maker](https://github.com/matta174/mold-maker) | TypeScript, three.js, Manifold WASM; web and Electron | PolyForm Noncommercial 1.0.0 (source-available, not OSI) | 2026-05-20 | 15 | Rigid 2-piece. N-piece is claimed in the last commit message (unverified) | Axis plane plus slider. "Auto-Detect" is self-rated "Adequate (vertex-balance + Z-pref)". Oblique 0-30° cuts ([ROADMAP](https://github.com/matta174/mold-maker/blob/main/ROADMAP.md)) / demoldability heatmap on the roadmap / auto sprue plus 2-4 vents "at cavity extremities" / pins with clearance holes / no draft analysis ([competitive analysis](https://github.com/matta174/mold-maker/blob/main/docs/competitive-analysis.md)) | Commercial casting needs a paid license. Clearances are percent-based. Quiet since May 2026 |
| [juppeee/moldmaker](https://github.com/juppeee/moldmaker) | Python, trimesh + manifold3d, pywebview | MIT | 2026-07-21 (1 commit) | 0 | Rigid, 2 halves | Auto axis, "silhouette-based, with undercut scoring" / scoring only / sprue on the parting plane, half in each mold half / dowels, screw holes with hex-nut recesses / none | Axis-aligned planar split only. One commit. German-only docs |
| [Mathew3000/moldgen](https://github.com/Mathew3000/moldgen) | Python, trimesh + manifold3d | GPL-3.0 | 2026-09-07 | 0 | Rigid 2-, 4- or 8-piece via manual seams; core sleeve | Manual seam axes / none / tapered funnel plus one manually placed vent / 2-4 pins with 0.15 mm radial clearance / none | Input "must already be watertight"; no repair; no automatic parting |
| [GreenShoeGarage/InjectionMoldFromSTL](https://github.com/GreenShoeGarage/InjectionMoldFromSTL) | Python, trimesh + manifold3d + shapely | GPL-3.0 | 2026-05-08 | 0 | Two-plate injection mold | Manual axis, bbox-midpoint offset / none / sprue, runner, gate, 0.2 mm vent slots / dowels / none | Built for injection, not gravity casting |
| [BM804/moldmaker2](https://github.com/BM804/moldmaker2) | Python, trimesh | None detected (an MIT claim in a search snippet is unverified) | 2026-09-17 | 0 | Rigid 2-piece plus an A3 drawing | Z mid-height or manual / none / not mentioned / pins / none | Z-only split |
| [mbparks/STL_Mold_Maker](https://github.com/mbparks/STL_Mold_Maker) | Python | GPL-3.0 | 2024-12-03 | 1 | Rigid, 2 halves | Fixed / none / pour spout / keys / none | Abandoned. The author was "not happy with the placement of the keys… nor… the pour spout" ([Substack](https://gearsofresistance.substack.com/p/automate-mold-making-with-python)) |
| [TobiasKoehlerETH/MoldMaker](https://github.com/TobiasKoehlerETH/MoldMaker) | TypeScript, Rust/Tauri, OpenCascade | None | 2026-09-25 | 0 | 2-piece printed mold for injecting RTV silicone | Rotates the thinnest axis onto the split direction / warns on deep undercuts / one syringe gate, up to 2 vents / 4 clamp screws / none; shrinkage-compensated cavity | STEP input only; no license |
| [H-69-stack/mold-auto-generator](https://github.com/H-69-stack/mold-auto-generator) | Python, cadquery/OCCT, Flask | None | 2026-09-23 | 1 | Upper and lower mold plus a core pin (pipe fittings) | Horizontal plane or a **silhouette-following curved surface** / demold interference check / n/r / n/r / n/r | Domain-specific; Chinese |
| [S0mbraD/3DPrint_MoldGen](https://github.com/S0mbraD/3DPrint_MoldGen) | Tauri + React + FastAPI/trimesh, CUDA | MIT | 2026-05-19 | 22 | Claimed multi-piece shells and skin molds | Claimed: adaptive demolding direction, 5 parting styles, gating, Darcy-flow pour simulation, draft analysis. **All unverified** | Windows plus NVIDIA GPU. The README still says `YOUR_USERNAME`, which suggests a template or AI-generated text |
| [HansTydecks/Concrete-Mold-Creator](https://github.com/HansTydecks/Concrete-Mold-Creator) | TypeScript, Manifold, browser | None | 2026-09-26 (1 commit) | 0 | Rigid multi-piece concrete molds, break-away or reusable | Automatic binary-space-partition split, scored by pieces needed and seam length / per-piece pull-direction search with ray-cast undercuts and a heatmap / funnel and vents / dowels and bolted flanges / n/r | The most automated open code found, but unlicensed and one commit old |
| [DR-89/local-mold-studio](https://github.com/DR-89/local-mold-studio) | TypeScript PWA, Manifold WASM | GPL-3.0-only | 2026-09-05 | 0 | Box molds with up to 8 segments; press molds | Auto-orientation, seam bounded by the model / n/r / 1-4 pour openings plus an optional vent / fit clearance, rubber-band grooves, pry pockets / n/r | New; German |
| [f2re/vase-mold-generator](https://github.com/f2re/vase-mold-generator) | Python local web app plus OpenSCAD | MIT | 2026-10-04 | 0 | Silicone formwork, or a rigid 2-half negative | n/r / not solved for hollow or interlocking parts / sprue and vents / pins / n/r | Requires an OpenSCAD install |
| [cauliflowermossgrape/moldgen](https://github.com/cauliflowermossgrape/moldgen) | Python FastAPI | MIT | 2026-07-13 | 1 | Carbon-fibre compression and layup tooling | Auto parting height / lock-and-drag "demolding check" heatmap / resin overflow channels / pins, bolts, jack screws / via the demolding check | Composites, not casting |
| [lisawong/plaster_mould_maker](https://github.com/lisawong/plaster_mould_maker) | Python (built as an LLM "skill") | n/r | 2026-09-26 | 0 | Two-part plaster slip-casting mould plus printable forms | Single planar split / n/r / n/r / tongue-and-groove, clips / n/r | "Release 1" only |
| Baseline: [Lion4re/automated_3d_mold_generator](https://github.com/Lion4re/automated_3d_mold_generator) | Python | MIT | 2026-01-28 (public) | 6 | Rigid mold with keys and pour spout | README: "intelligent wall thickness detection, alignment keys, and pour spouts" | Requester's own project |

### 2b. Silicone mold boxes, mother molds, composite and break-away shells

| Project | Stack | License | Last commit | Stars | Output | Automation | Main weaknesses |
|---|---|---|---|---|---|---|---|
| [vjvarada/VcMoldCreator](https://github.com/vjvarada/VcMoldCreator) | Python desktop, GPU, fTetWild | Ambiguous: MIT badge, but the README says "for academic and research purposes" and there is no LICENSE file | 2026-05-29 (100+ commits) | 0 | Printed hard shell plus a 2-piece silicone composite mold; reimplements Alderighi et al. 2019 | GPU visibility sampling of parting directions, tetrahedral "escape paths", membranes, pouring direction by persistent homology | Heavy dependencies; license unclear |
| [seena18/mothermold](https://github.com/seena18/mothermold) | Python CLI plus a browser "Studio" | GPL-3.0 | 2026-09-30 | 0 | Jacket halves, lid, displacement core, master with lugs | Repair, Booleans, interference checks; reports non-manifold output | The CLI build needs Blender 5.1; alpha |
| [gachetiberiu-WIZ/silicone-mold-app](https://github.com/gachetiberiu-WIZ/silicone-mold-app) | Electron + manifold-3d | "UNLICENSED" (proprietary although public) | 2026-04-23 | 0 | Conforming silicone offset plus a print shell; radial 2/3/4 pieces | 45° step interlock. Sprue, vents and keys "were removed" ([CLAUDE.md](https://github.com/gachetiberiu-WIZ/silicone-mold-app/blob/main/CLAUDE.md)) | Not reusable legally |
| [iphonetikpr/Moldmaker](https://github.com/iphonetikpr/Moldmaker) | TypeScript SPA plus Docker | None | 2026-09-25 | 0 | Adapted box, tray, 2-part silicone with a Z cut | `draftDeg` 1.5, pins Ø3.0 in holes Ø3.25, cast-shrink %, pieces split to at most 250 mm | No direct molds, no internal cavities |
| [TheBrindle/CrushMoldBuilder](https://github.com/TheBrindle/CrushMoldBuilder) | TypeScript, three-mesh-bvh, manifold-3d | None | 2026-07-19 | 0 | Single-piece break-away "eggshell" shell built with `levelSet` | Fill port and vents placed by clicking | Deploy "coming soon" |

### 2c. Add-ons, libraries and slicer modes inside existing tools

| Project | Host | License | Last activity | Popularity | Output | Automation | Main weaknesses |
|---|---|---|---|---|---|---|---|
| [MoldForge](https://extensions.blender.org/add-ons/moldforge/) ([GitHub](https://github.com/Plesuro/MoldForge)) | Blender 5.1+ | GPL-3.0-or-later | Extension v0.41.0 on 29 Sep 2026; GitHub source last pushed 19 Jun 2026 | 4,542 downloads; 5.0/5 from 3 reviews; 7 stars | Rigid direct mold ("Hugging" or "Block"), silicone pour box with glove-skin keys, open tray | Auto split axis (X or Y) by fewest undercuts; ray-cast contoured parting; 2-4 radial wedges; ray-based undercut warning; 1-4 funnels; 0-8 vents from cavity high points; cone keys or teeth; **no draft** | Blender lock-in. Voxel remeshing "smoothed" fine detail. GitHub source lags the extension build |
| [Ruakij/blender-mold-generator](https://github.com/Ruakij/blender-mold-generator) | Blender 4.2+ | GPL-3.0 | 2026-10-03 | 9 stars | 2-part shell / mother mold | Finds the longest horizontal cross-section near the top; extrudes upward to remove undercuts | Horizontal cuts only; no keys or sprue |
| [n2048 mold-generator-blender-addon](https://github.com/n2048-creative-technology/mold-generator-blender-addon) | Blender | MIT | 2026-09-02 | 1 star | Two-part rigid ("Phase 1 of 3") | Undercut detection and multi-part support are only planned | "STATUS: NEEDS WORK" |
| [FormCraft](https://github.com/JMPdesign1337/formcraft_addon) | Blender | README says GPL-3.0; no license file | 2026-05-06 | 0 stars | Plaster slip-casting molds | Auto split on an axis, round keys, pour hole, vent channels | Axis splits only |
| [INGENADOR generator](https://github.com/INGENADOR/GENERADOR-DE-MOLDES-EN-BLENDER) | Blender 4.1+ | None | 2026-08-11 | 0 stars | Rigid 2-part | User-chosen axis, corner pins with clearance, funnel, flipped export | No undercut handling |
| Lure Mold Maker ([zanerogers1990/moldmaker](https://github.com/zanerogers1990/moldmaker)) | Blender 5.2 | Unknown (repo returns 404) | n/r | n/r | Two-part lure molds | Claimed: curved parting surface with pull-direction search, pins, sprue, vents (**unverified, deleted**) | Unavailable |
| [FreeCAD-Molding-Wizard](https://github.com/ryankembrey/FreeCAD-Molding-Workbench) | FreeCAD 1.0+ | LGPL-2.1 | 2026-10-01 | 1 star | Silicone casting molds in 2-3 pieces | Plane seeded at the widest section; user-chosen pull; **tapered cone keys** that "absorb the elephant foot"; pour port, vents, overflow gutter; volume report | "Undercut warnings are advisory"; no non-planar parting |
| Jason Webb OpenSCAD generator ([Thingiverse 31581](https://www.thingiverse.com/thing:31581); CC0 re-host [rekliner/mold-generators](https://github.com/rekliner/mold-generators)) | OpenSCAD | CC0 (re-host) | 2012; re-host 2026-08-25 | n/r | Rigid 2-part or open-face | STL minus a cube; sphere/cube keys; tapered pour hole; dimensions entered by hand | "Avoid undercuts"; slow STL import |
| [Cura Mold mode](https://github.com/Ultimaker/Cura/blob/main/resources/definitions/fdmprinter.def.json) | UltiMaker Cura | LGPL (Cura) | Current | n/r | Single-piece open shell | `mold_width` 5 mm, `mold_roof_height` 0.5 mm, `mold_angle` 40° | No split, keys, sprue or vents. PrusaSlicer closed the request as a "very unusual idea" ([#8661](https://github.com/prusa3d/PrusaSlicer/issues/8661)) |

### 2d. Analysis-only tools and academic code

| Project | License | Last commit | Stars | What it does | Usability |
|---|---|---|---|---|---|
| [lucagarau/MoldGenSegmentation](https://github.com/lucagarau/MoldGenSegmentation) | None | 2026-07-13 | n/r | Per-face extractable directions (Fibonacci sampling, Embree/PyTorch3D visibility), then graph-cut α-expansion that minimizes the number of pull directions | Research-grade; pinned trimesh 3.23.5 and a prebuilt `libcgco.so` |
| [FL0WL0W/STL-Minimum-Draft](https://github.com/FL0WL0W/STL-Minimum-Draft) | n/r | 2026-04-05 | 0 | Draft-angle analysis; clips faces to a minimum draft | Exports "may have geometry inconsistencies" |
| [by-carrot/cad-auditor](https://github.com/by-carrot/cad-auditor) | n/r | 2026-06-27 | n/r | DFM review on trimesh: draft, wall thickness, undercuts | Analysis only |
| [alemuntoni/HeightFieldDecomposition](https://github.com/alemuntoni/HeightFieldDecomposition) | MPL-2.0 | 2020-01-16 | 23 | Muntoni et al. 2018 height-field block decomposition | Needs Ubuntu 18.04, Qt, CGAL and the commercial Gurobi solver |
| [tb2-sy/AIMold](https://github.com/tb2-sy/AIMold) | None detected | 2026-08-04 | 12 | ECCV 2026 injection-mold pipeline from CAD | Code release unclear. arXiv says code is available ([arXiv](https://arxiv.org/abs/2608.00800)), but the repo looked like a teaser (inference) |

Three conclusions follow from these tables:

- **The closest feature-for-feature rival is MoldForge, not a standalone repo.** It is free, has about 4.5k downloads, and already offers automatic axis choice, radial multi-piece, an undercut check, multiple funnels and vents. Its limits are Blender lock-in, multi-piece that only means radial wedges around Z, and detail lost to voxel remeshing ([MoldForge README](https://github.com/Plesuro/MoldForge)).
- **The only open code with general undercut-driven multi-piece logic is HansTydecks.** Its lack of a license means nobody can legally build on it.
- **No open or commercial tool was found that places vents by cavity topology** rather than by high points or extremities. Draft is analyzed only in FormArmor Pro, the FreeCAD Curves workbench and a few repos ([FormArmor Pro](https://superhivemarket.com/products/formarmorpro)).

## 3. Commercial tools: professional CAD automates parting on B-rep, maker apps automate everything except undercuts

The market splits in two:

- **Professional injection-mold CAD** automates parting-line search, shut-off and parting surfaces, core/cavity/slider splits, shrinkage and mold bases. It does this only on B-rep solids, for steel tooling, at quote-based prices. The 2022 survey puts simple two-piece tools at about 2k EUR and high-end suites at several 100k EUR ([STAR preprint §7](https://research-explorer.app.ist.ac.at/download/11993/11994/star_molding_preprint.pdf)).
- **Maker web apps** automate the whole STL-to-printable-mold pipeline cheaply, but mostly with flat-plane two-part splits.

### 3a. Professional and desktop CAD

| Tool | Price / licence | What it automates | What it does not do for STL casting | Sources |
|---|---|---|---|---|
| Autodesk Fusion | Free Personal Use (non-commercial, under US$1,000/yr from the work, 10 editable documents, no extensions). About $680/yr paid (aggregator figure) | Plastic Rules draft/thickness advice. The Simulation Extension does injection fill. App Store add-ins: Mold Commands (parting surfaces from edges, core/cavity split, "may fail"), AIG Mold Wizard (mold bases, sprue/runner), NexGenCAM (DME bases) | No built-in automated parting or core/cavity split. Mesh-to-B-rep is reliable only below about 10k triangles (one guide) | [Autodesk Personal](https://www.autodesk.com/products/fusion-360/personal); [Vagon](https://vagon.io/blog/fusion-360-pricing); [Plastic Rules](https://help.autodesk.com/view/fusion360/ENU/?contextId=SLD-SETUP-PLASTIC-RULES); [Mold Commands](https://apps.autodesk.com/FUSION/en/Detail/HelpDoc?appId=44916951188326107&appLang=en&os=Win64); [AIG](https://apps.autodesk.com/FUSION/en/Detail/HelpDoc?appId=5737568183660398764&appLang=en&os=Win64); [JDL mesh guide](https://jdlinnovations.au/blog/convert-3d-scan-to-brep-fusion-360) |
| SOLIDWORKS Mold Tools | Quote-based | Draft, Undercut and Parting Line Analysis; parting lines; shut-off surfaces; parting surfaces; Tooling Split with an optional automatic interlock | The user picks direction, shut-offs and split sketch. Users report "Cannot split mold with a complex parting line". No sprue, vents or keys for gravity casting | [SOLIDWORKS 2026 Help](https://help.solidworks.com/2026/English/SolidWorks/sldworks/c_Mold_Design.htm); [GoEngineer](https://www.goengineer.com/blog/solidworks-mold-tools-shut-off-surfaces-parting-surfaces-tooling-split); [Forum 183111](https://forum.solidworks.com/thread/183111) |
| IMOLD for SOLIDWORKS | Quote-based, free trial | Automatic core/cavity extraction, undercut detection, mold base, feed systems, cooling | Injection tooling scope | [Capterra](https://www.capterra.com/p/93143/IMOLD/) |
| Siemens NX Mold Wizard; PTC Creo Mold; CATIA Core and Cavity; Cimatron; VISI Mould | Quote-based | Automatic parting-line search, hole patching, core/cavity/slider split, mold bases. Creo applies shrinkage by dimension. Cimatron QuickSplit works "no matter what the quality of the part". VISI imports STL | Steel injection assemblies. Not confirmed to split directly on facet bodies | [Siemens NX](https://www.siemens.com/en-us/products/nx-manufacturing/tooling-fixture-design/mold-design/); [PROLIM](https://www.prolim.com/introduction-to-nx-mold-wizard-basics-of-injection-mold-design/); [Creo shrinkage](https://support.ptc.com/help/creo/creo_pma/r11.0/usascii/mold_and_casting/mold/To_Apply_Shrinkage_by_Dimension.html); [Cimatron QuickSplit](https://help.cimatron.com/en/2026/Menu_Parting_QuickSplit_Tools.htm); [VISI](https://www.softwa.co.za/mould.htm) |
| Autodesk Moldflow | $2,615 to $40,890 per year by tier | Injection fill, pack and warp simulation | No gravity-casting simulation was found in any tool | [Novedge Adviser Premium](https://novedge.com/products/buy-moldflow-adviser-premium-subscription); [Novedge Insight Ultimate](https://novedge.com/products/buy-moldflow-insight-ultimate-subscription) |
| Rhino 7/8; RhinoMold; MatrixGold | Rhino licence; RhinoMold status unknown | DraftAngleAnalysis and RibbonOffset parting surfaces. RhinoMold had automatic cavity/core separation but documents Rhino 5 (looks legacy). MatrixGold automates sprues and trees for jewelry investment casting | No mesh mold generator found | [Rhino mold tools](https://www.rhino3d.com/features/mold-making-tools/); [RhinoMold help](http://help.tdmsolutions.com/rhinomold/4.0/en/); [MatrixGold](https://gemvision.com/matrixgold) |
| ZBrush, Nomad Sculpt, Materialise Magics / 3-matic | Various | Booleans and offsets only | Molds are built by hand. Nomad Booleans fail on fully enclosed objects | [ZBrushCentral](https://www.zbrushcentral.com/t/2-piece-box-mold-with-keys-i-need-help/352027); [Nomad forum](https://forum.nomadsculpt.com/t/fully-enclosed-boolean-for-mold-making/23614); [Magics](https://www.materialise.com/en/industrial/software/magics-data-build-preparation) |
| Paid Blender add-ons | FormArmor Pro $49 (10 sales); MouldCast Studio $27 (60+ sales, MIT); MAX Mold Pro R$147 (about US$29) | FormArmor: automatic or manual planes, keys, screw tabs, **draft-angle verification**, basin and vents. MouldCast: adjustable part count, keys, pour hole. MAX: 11 mold types, 2-4 parts | Silicone-first; planar parting; small sales | [FormArmor Pro](https://superhivemarket.com/products/formarmorpro); [MouldCast](https://superhivemarket.com/products/mouldcast); [MAX Mold Pro](https://www.maxmold.com.br/en/) |

### 3b. Maker web and app services (closed source)

| Service | Pricing | Notable automation | Admitted limits | Source |
|---|---|---|---|---|
| Meshcast | Free: 3 downloads/week, personal use, watermark. $9/mo, $20/mo (commercial). $349 founder deal | Auto-orient on XYZ, flat seam at the widest silhouette, tongue-and-groove or pins, funnel over high points (default 8 mm), vent, pry pocket, 5 mm wall | No automatic undercut detection. "Fill the undercuts" or switch to silicone. Homepage says "multi-piece" but the guide says two-part only (conflict) | [Pricing](https://meshcast.app/pricing); [Mold](https://meshcast.app/mold); [Guide](https://meshcast.app/guides/stl-to-mold) |
| jocam3d | Free | Tries "about 80 directions"; curved parting that "follows the model's edge"; undercuts painted red; corner keys; pour at the highest point and vents at other high points; Manifold in the browser | "Figurines, faces… lock in every direction" | [jocam3d](https://jocam3d.com/en/tools/stl-to-mold) |
| Akritio (Navi3D) | Free: 1 export/day. Pro: $9 per 30 days | "Automatic parting line", 2-, 4- or 6-part molds on 2-3 planes, cone keys, funnel, vents; materials include lost-PLA metal; print-and-ship | Radial and planar only | [Akritio](https://akritio.com/); [Pricing](https://akritio.com/pricing) |
| MoldStudio3D | Free tier; $19 or $29 per month | Parting-direction and line analysis, undercut/demold review; Block, Skin, Silicone and Ceramic modes | Sprue and shrinkage not mentioned | [MoldStudio3D](https://moldstudio3d.com/mold-generator/) |
| 3DMold (iOS) | $2.99 per month | "Intelligent Parting Sweep", runner and vent automation, **shrinkage compensation**, pins with draft | Release year unverified | [App Store](https://apps.apple.com/us/app/3dmold-3d-mold-maker-stl/id6758258851) |
| Moldboxer (desktop) | Lite $59 per the buy page; the other researcher saw no prices | "Unlimited customizable divisions", air-trap detection, silicone calculator | Silicone and slip-casting focus | [Moldboxer](https://moldboxer.com/); [buy page](https://moldboxer.com/buy) |
| splicestl, PrintPal, Mold Studio, AutoMold3D, Action BOX | Free to credit packs ($9-99) | Flat X/Y/Z or automatic plane, pegs, spout, vents; Action BOX exports STEP and has a "Demoldability Check" | splicestl: "unusable geometry on highly organic forms". Action BOX: "may have issues with" undercuts | [splicestl](https://splicestl.com/mold/); [PrintPal](https://printpal.io/tools/mold-box-generator); [Mold Studio](https://moldstudio.app/); [AutoMold3D](https://automold3d.com/); [Action BOX](https://mold.actionbox.ca/) |
| MoldMaker AI | Free (claimed) | Claims "perfect split planes, alignment keys, and draft angles" (**unverified**; page returned HTTP 429) | Unknown | [MoldMaker AI](https://moldmakerai.com/) |

**What these services automate is the pipeline, not the hard geometry.**

- Only jocam3d claims a curved parting surface. Only Akritio and Moldboxer sell more than two pieces, and only as radial or user-placed divisions ([Akritio](https://akritio.com/pricing), [Moldboxer](https://moldboxer.com/)).
- Among maker tools, only 3DMold claims shrinkage compensation ([App Store](https://apps.apple.com/us/app/3dmold-3d-mold-maker-stl/id6758258851)).
- No vendor publishes its algorithms. The "AI" branding appears to be marketing for heuristic geometry (inference, [MoldMaker AI](https://moldmakerai.com/)).
- Printer and material vendors publish manual design guides, not generators: [Formlabs](https://formlabs.com/blog/casting-silicone-guide/) and [Prusa](https://blog.prusa3d.com/the-beginners-guide-to-mold-making-and-casting_31561/).
- Protolabs runs free automated DFM checks (draft, undercuts), but only on CAD formats, not meshes ([Protolabs](https://www.protolabs.com/en-gb/automated-design-analysis/)).

For an STL-first user, professional CAD offers nothing. Fusion's personal tier is non-commercial, and its mesh conversion breaks down on dense organic meshes ([JDL Innovations](https://jdlinnovations.au/blog/convert-3d-scan-to-brep-fusion-360)). That leaves the freemium apps, with their download quotas, watermarks and paywalls on commercial use, as the incumbents an open tool must beat.

## 4. Academic techniques (ranked by practical value for us): 1990s geometry beats 2018 graphics for an open tool

The 2022 survey by Alderighi et al. defines the problem this project solves ([STAR preprint](https://research-explorer.app.ist.ac.at/download/11993/11994/star_molding_preprint.pdf)). Every surface patch must belong to one mold piece. Each piece must be a height field along its extraction direction, and the pieces must come apart by straight-line translation. The survey also names gate, sprue, vent and interlock design and a public dataset as open problems.

The ranking below weighs value to a rigid printed-mold generator against effort in trimesh + manifold3d + numpy/scipy, code availability and patent exposure. Effort ratings are the researchers' engineering judgments from the algorithm descriptions; no prototype was built.

| Rank | Technique | Effort | Value | Code | Patent |
|---|---|---|---|---|---|
| 1 | Gravity-filling vents and sprue (Bose-Toussaint 1995; Bose-van Kreveld-Toussaint 1998) plus persistence ranking and the vents-on-parting rule | Low | Very high | None needed | None found. One CoreCavity claim element is unverified |
| 2 | Castability as monotonicity and ray accessibility (Ahn et al. 2002; Ahn-Cheong-van Oostrum 2003) | Low | Very high | CGAL is 2D only | None found |
| 3 | Visibility and accessibility direction search (Chen-Chou-Woo 1993; Khardekar 2006; Gupta; Elber 2005) | Low-medium | High | None | None found |
| 4 | Parting line and surface for a given direction (Li-Martin-Langbein 2009; Khardekar-McMains 2009; terrain lift; RBF) | Medium | High for organic parts | None | RBF surface is from CoreCavity (generic sub-technique) |
| 5 | Height-field / swept-volume undercut clearing (CoreCavity §6.2; Protective Foam 2025) | Low-medium | High as a fallback | None | Generic |
| 6 | Multi-piece by accessibility, set cover and planar cuts (Priyadarshi-Gupta 2004; Huang-Gupta-Stoppel 2003) | High | Very high (the moat) | None | None found |
| 7 | Kit automation from Shape Cast (CHI 2024/2025) | Low | High | Hosted only | None found |
| 8 | Volume decomposition for two-piece rigid casting (Alderighi et al. 2021) | High | Medium | None | None found |
| 9 | Height-field and pyramidal decompositions (Muntoni 2018; Hu 2014; Herholz 2015) | Very high | Medium (borrow the pattern) | Muntoni, MPL-2.0 | None found |
| 10 | Volume-aware composite molds (2019) | High | Low-medium (cheap pieces only) | Third-party reimplementation | Plausibly US11966666B2 (interpretation) |
| 11 | Metamolds (2018) | High | Low | None | US11966666B2 to 2040 |
| 12 | CoreCavity, full method (2018) | High | Low | None (dataset only) | US11931922B2 to 2041 |
| 13 | FlexMolds (2016) | High | Low | None | EP3301597B1 to about 2036 |
| 14 | Shape-deforming castability (Stein et al. 2019) | Medium-high | Low | None | None found |
| 15 | Other 2022-2026 work (AIMold, PackMolds, fabric formwork, Eggshell, joinery, ChewTect) | Varies | Mostly low | AIMold: no license | None found |

### 4.1 Sprue and vent placement from gravity filling

**Low effort, very high value.**

**Bose, van Kreveld and Toussaint proved the vent rule for a mold filled from the top with gravity along −z.** A polyhedral mold fills from one pin gate without trapping air **if and only if the cavity has exactly one local maximum in +z**. The gate therefore goes at the global maximum and vents at every other local maximum. Their algorithms test a fixed orientation in O(n) and find all orientations that allow single-gate filling in O(n²) ([Toussaint PostScript](https://cgm.cs.mcgill.ca/~godfried/publications/casting3D.ps.gz); [Utrecht portal](https://research-portal.uu.nl/en/publications/filling-polyhedral-molds/)). Bose and Toussaint's 2D version finds orientations that minimize the number of vents in Θ(n log n) ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/001044859500018M)).

Two later results make this practical on noisy meshes:

- **Persistence pairs.** The volume-aware composite-mold paper ranks trapped-air sites by maximum-saddle persistence pairs of the height function ([Volume-aware PDF §5.2](https://opus.lib.uts.edu.au/bitstream/10453/137699/4/composite_compressed.pdf)).
- **Vents on the parting surface.** Alderighi et al. 2021 show that when the pour direction is orthogonal to the pull direction, every local maximum of a double-height-field part lies on the boundary between the d and −d regions. Vents can then live entirely on the parting surface ([AMB21 postprint §4.3.2](http://openportal.isti.cnr.it/data/2021/456702/2021_456702.postprint.pdf)).

**Implementation in this stack:**

1. Find vertex local maxima with `mesh.vertex_neighbors`.
2. Run a union-find over vertices sorted by height to get 0-dimensional persistence.
3. Vent only maxima whose persistence exceeds about 0.3-1 mm (inference).
4. Choose among the repo's four ±X/±Y pour orientations the one with the fewest significant maxima.

This takes roughly 100-200 lines of numpy, at O(V log V).

**The repo already pours perpendicular to the pull.** Its mold frame puts the pull along ±Z and stands the mold with ±X/±Y up. That is exactly the AMB21 condition. So for an undercut-free part with a planar parting plane, all vents can be printed as half-channels in the plane, just like the existing sprue (inference from AMB21).

**Caveats:** the model ignores viscosity, so treat the vents it finds as necessary but not sufficient. Also recompute after adding the sprue, which can create a new maximum (inference from [Bose et al. 1998](https://cgm.cs.mcgill.ca/~godfried/publications/casting3D.ps.gz)).

### 4.2 Castability, monotonicity and directional uncertainty

**Low effort, very high value.**

**Ahn et al.: a two-part mold with opposite pulls exists exactly when the part is monotone.** A polyhedron is castable in a two-part mold with opposite pulls ±d if and only if it is **monotone** in d, meaning every line parallel to d meets it in one segment. When it is, a valid parting surface always exists as a terrain obtained by lifting a triangulation of the projection ([UU-CS-2001-48](https://webdoc.sub.gwdg.de/ebook/serien/ah/UU-CS/2001-48.pdf)). The exact all-directions algorithm runs in O(n³ log n) (secondary, from the abstract) ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448501001191)).

**Ahn, Cheong and van Oostrum add draft.** A part is castable under angular uncertainty α if and only if it is α-monotone and has no facet within α of parallel to d. That is effectively a minimum-draft condition. They search a set of O(1/α²) sphere samples whose covering radius is α ([UU-CS-2001-48 §4](https://webdoc.sub.gwdg.de/ebook/serien/ah/UU-CS/2001-48.pdf)). This theoretically justifies the repo's Fibonacci-hemisphere sampling. At α = 2°, about 1,650 hemisphere directions are needed (inference).

**The survey tells us how to test this on a mesh.** Cast rays along d: a front-face hit is an overhang and a back-face hit is an overlap ([STAR §3](https://research-explorer.app.ist.ac.at/download/11993/11994/star_molding_preprint.pdf)). AMB21's **undercut tolerance ε = 0.4 mm** lets casts "snap out of mild undercuts". It was used with PLA/ABS molds and Shore 70D urethane ([AMB21 postprint](http://openportal.isti.cnr.it/data/2021/456702/2021_456702.postprint.pdf)).

**Undercut volume is a better objective than undercut area.** Shoot a grid of lines parallel to d and count hits per line (inference from the monotonicity lemma).

**Existing implementations:** CGAL's casting package is 2D only ([CGAL manual](https://doc.cgal.org/latest/Set_movable_separability_2/index.html)). The 2024 optimal single-part-mold algorithms were implemented in CGAL, but no public repository was found ([CGT journal](https://www.cgt-journal.org/index.php/cgt/article/view/15)).

### 4.3 Visibility maps and accessibility for choosing the parting direction

**Low to medium effort, high value.**

The CAD literature offers three families of methods:

- **Spherical visibility maps.** Chen, Chou and Woo pick an antipodal direction pair that covers the most "pocket" visibility maps (secondary, via the [Bassi et al. review](https://www.cad-journal.net/files/vol_7/CAD_7(5)_2010_621-637.pdf)). Woo formalized visibility on the sphere ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0010448594900035)).
- **Ray and sweep accessibility.** Examples are Hui and Tan's "blocking factor" ([HKU](https://hub.hku.hk/handle/10722/156364)) and Dhaliwal et al.'s global accessibility cones ([ASME](https://asmedigitalcollection.asme.org/computingengineering/article-abstract/3/3/200/460132/Algorithms-for-Computing-Global-Accessibility)).
- **GPU image-space tests.** Khardekar, Burton and McMains show that undercut-freeness is constant within cells of a great-circle arrangement ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448506000248)).

Gupta's "one-click" tool handled 5,000 facets in about 2 seconds ([UMD overview](https://drum.lib.umd.edu/bitstreams/a0209646-8ff2-49f4-84a8-7e4e421e2051/download)). That is a good argument for running the search on a decimated proxy mesh.

Elber, Chen and Cohen prove that a genus-g surface has a valid two-piece partition if and only if some direction yields exactly g+1 silhouette loops ([Elber PDF](https://gershon.cs.technion.ac.il/papers/MoldAccessibility.pdf); the journal venue is secondary). That gives a cheap validator: count loops from `mesh.face_adjacency` sign changes.

**The practical recipe:**

1. Build a candidate set from Fibonacci samples, PCA axes and convex-hull pocket directions (`hull() - part`, then `decompose()` in manifold3d).
2. Score every candidate with a vectorized normal-only score.
3. Run ray tests on only the top candidates.

Exact accessibility cones and aspect graphs are theory only. Bassi et al. warn that ray sampling misses features smaller than the sampling step ([CAD&A 2010](https://www.cad-journal.net/files/vol_7/CAD_7(5)_2010_621-637.pdf)).

### 4.4 Parting line and parting surface

**Medium effort, high value for organic parts.**

Every method built for meshes first builds a **band of near-vertical triangles** (tolerance "typically less than four degrees") and fits the parting line inside it ([Khardekar-McMains 2009](https://mcmains.me.berkeley.edu/pubs/SPM09khardekarMcMains.pdf)).

**Khardekar and McMains** prove that a parting line lying in the minimum number of planes is **NP-complete**. They give a greedy linear-programming method that produces stepped planes along the outermost silhouette loop ([PDF](https://mcmains.me.berkeley.edu/pubs/SPM09khardekarMcMains.pdf)).

**Li, Martin and Langbein** smooth the parting line and remove small undercuts by deforming the mesh. Their method is slow: a 20k-triangle model needed 8 minutes for undercut removal, and a 437k-triangle model about 2 hours. It also alters the geometry ([PDF](https://langbein.org/wp-content/uploads/2009/08/li2009.pdf)).

**Gupta's flatness ratio ρ** equals 1 only for a planar loop, so it is a clean test for when a flat plane suffices ([UMD overview](https://drum.lib.umd.edu/bitstreams/a0209646-8ff2-49f4-84a8-7e4e421e2051/download)).

**Two general non-planar surfaces exist:**

- Ahn et al.'s lifted terrain, which triangulates the mold block minus the projection and lifts it to the parting line ([UU-CS-2001-48](https://webdoc.sub.gwdg.de/ebook/serien/ah/UU-CS/2001-48.pdf)).
- CoreCavity's RBF height-field interpolation through the parting-line vertices ([CoreCavity PDF §6.3](https://research-explorer.ista.ac.at/download/12/5360/IST-2018-1037-v1%2B1_CoreCavity-AuthorVersion.pdf)).

Both map to `scipy.interpolate.RBFInterpolator` or `trimesh.creation.triangulate_polygon`, followed by a watertight cutter and a manifold3d split. Expect sliver triangles where silhouettes are near-vertical.

Recommended order: planar surface; then stepped planes when ρ is low; then a terrain surface.

### 4.5 Undercut clearing with height fields and swept volumes

**Low to medium effort, high value as a guaranteed fallback.**

CoreCavity sweeps each mold face along d and −d and combines the sweeps with Booleans to remove residual undercuts ([CoreCavity PDF §6.2](https://research-explorer.ista.ac.at/download/12/5360/IST-2018-1037-v1%2B1_CoreCavity-AuthorVersion.pdf)). The 2025 Protective Foam paper builds each half from depth textures, so each half is a height field by construction ([arXiv 2503.09019](https://arxiv.org/abs/2503.09019)).

**The easiest robust version:**

1. Ray-cast a grid along ±d to get the upper and lower envelopes of the cast.
2. Make each mold half the solid beyond its envelope.
3. Subtract with manifold3d.

The halves always release, at the cost of filling undercuts into the cast. Meshcast already offers this as "Fill the undercuts" ([Meshcast](https://meshcast.app/mold)).

### 4.6 Multi-piece molds from accessibility analysis

**High effort, very high strategic value.**

**Priyadarshi and Gupta** automated the whole chain: global accessibility, per-region parting directions, parting lines and surfaces, and pieces with guaranteed disassembly. They solved representative industrial parts in under 5 minutes ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448503001076); [UMD overview](https://drum.lib.umd.edu/bitstreams/a0209646-8ff2-49f4-84a8-7e4e421e2051/download)).

**Huang, Gupta and Stoppel** restrict piece boundaries to planes. They choose 1-3 cuts by **set covering** over enumerated candidate planes. This is the most transferable variant, because manifold3d does planar splits natively (via the [UMD overview](https://drum.lib.umd.edu/bitstreams/a0209646-8ff2-49f4-84a8-7e4e421e2051/download)).

**Banerjee and Gupta's side-action work** maps to small removable inserts ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448507001339)). The open [MoldGenSegmentation](https://github.com/lucagarau/MoldGenSegmentation) shows a modern graph-cut variant.

**Python sketch:**

1. Build an accessibility matrix A[face, direction] from 14-26 or sampled directions.
2. Pick the fewest directions that cover all faces by greedy set cover. Report faces covered by no direction as needing a core.
3. Label faces with a binary or α-expansion min-cut. Check solver licenses: gco is research-only per the notes, and this is unverified.
4. Make planar cuts and verify each piece's accessibility.
5. Check removal order with sweep tests.

No open-source implementation of the UMD algorithms exists.

### 4.7 Shape Cast and its kit automation

**Low effort, high value.**

Lyons' Shape Cast (CHI EA 2024, extended at CHI 2025) automates printed plaster slip-casting mold kits. It had a public beta with 501 sign-ups and 626 finalized models ([CHI 2025 DOI](https://doi.org/10.1145/3706598.3713866)). Its pipeline:

- scales the pot for 13% clay shrinkage;
- splits the outer mold into **2 or 4 flanged pieces** with M3 bolts and heat-set inserts;
- adds a gasket provision and a pour hole;
- computes plaster cavity volume and the dry-plaster and water weights ([Shape Cast PDF](http://www.emmielyons.com/pubs/shapecast-chi24.pdf)).

None of this is algorithmically hard. All of it is what users need next to the geometry. The code is not released.

### 4.8 Volume decomposition for two-piece rigid casting (Alderighi et al. 2021)

**High effort, medium value.**

The method splits the object's volume into a few parts that are each a double height field, casts each part in a two-piece mold, and glues the parts together.

**Pipeline:**

1. TetWild tetrahedra.
2. Accessibility for 512 directions × 15 samples per tetrahedron, computed with Embree.
3. The ε tolerance.
4. α-expansion graph cut, run 15 times.
5. Poisson parting surfaces.

**Runtimes and results:** accessibility took 5-320 s, the graph cut 2-1911 s and mold generation 224-2113 s, yielding 1-9 parts ([AMB21 postprint](http://openportal.isti.cnr.it/data/2021/456702/2021_456702.postprint.pdf)). No code and no patent were found ([VCG page](https://cnr-isti-vclab.github.io/publication/2021/AMBCP21/)).

**A simplified version is feasible.** It would compute accessibility on a voxel grid instead of tetrahedra, label greedily, and rebuild solids with `Manifold.level_set` (inference). Gluing the cast parts afterwards is a real usability cost.

### 4.9 Height-field and pyramidal decomposition

**Very high effort. Borrow the pattern only.**

Deciding whether a shape splits into k terrains is **NP-hard** ([Fekete and Mitchell 2001](https://www.worldscientific.com/doi/abs/10.1142/S0218195901000687)). The three main methods:

- **Hu et al.'s pyramidal decomposition** takes 8-16 minutes per shape with voting, clustering and exact cover ([PDF](https://www.cs.sfu.ca/~haoz/pubs/hu_siga14_pym.pdf)).
- **Muntoni et al.'s axis-aligned height-field blocks** target milling. Their code is MPL-2.0 but needs Gurobi and Ubuntu 18.04 ([GitHub](https://github.com/alemuntoni/HeightFieldDecomposition)), and the survey says it over-segments for molding ([STAR §4.2](https://research-explorer.app.ist.ac.at/download/11993/11994/star_molding_preprint.pdf)).
- **Herholz et al.** deform the shape to fit a few height fields ([MIT CDFG](https://cdfg.mit.edu/publications/approximating-free-form-geometry-height-fields-manufacturing)). That their segmentation uses a graph cut is unverified.

Their shared primitive is a per-face height-field test followed by a covering or labeling solve. That is exactly the shape of the Section 4.6 sketch.

A 2023 representation paper finds that typical closed shapes are intersections of three or fewer double height fields ([CN-DHF arXiv](https://arxiv.org/abs/2304.13141)). That suggests trying the principal axes first. This is speculative and untested.

### 4.10 Volume-aware composite molds (Alderighi et al. 2019)

**High effort. Patent-adjacent. Adopt only the cheap pieces.**

Each half is a rigid printed shell plus a thin silicone liner with automatically computed cut membranes. The method:

- picks two directions that minimize non-visible area from sampled GPU visibility;
- uses tetrahedral "escape paths" to place membranes;
- sets the silicone thickness with a **fixed 15 mm offset** for 95-300 mm objects;
- adds **Perlin noise** to the silicone parting surface "to improve registration";
- chooses the resin pour direction by persistence pairing of trapped air.

Measured cast error was RMS 0.18-0.49 mm, and tetrahedralization alone took about 35-47 minutes per model ([Volume-aware PDF](https://opus.lib.uts.edu.au/bitstream/10453/137699/4/composite_compressed.pdf)). There is no author code. [VcMoldCreator](https://github.com/vjvarada/VcMoldCreator) reimplements it under an ambiguous license.

Worth taking: the persistence vent idea, the 15 mm silicone default for a mother-mold mode, and noise registration for silicone halves only. Do not adopt the membrane pipeline.

### 4.11 Metamolds (Alderighi et al. 2018)

**High effort, patented. Skip the core method.**

Silicone is cast into 3D-printed "metamolds" whose geometry defines the cuts that release the cast. The method combines GPU per-direction visibility, a "moldability score" for peeling flexible pieces, and cut membranes for topological handles. ISTA reports casts in resin, chocolate and ice with "the smallest possible number of mold pieces" ([VCG page](https://cnr-isti-vclab.github.io/publication/2018/AMGPBC18/); [ISTA news](https://ista.ac.at/en/news/metamolds-molding-a-mold/)). It fails on knotted loops and on complex shapes such as the dragon model ([Volume-aware PDF §6](https://opus.lib.uts.edu.au/bitstream/10453/137699/4/composite_compressed.pdf)).

No code was found. A replicability.graphics entry reportedly said so, but that page now returns 404 (unverified) ([replicability.graphics](https://replicability.graphics/papers/10.1145-3197517.3201381/index.html)). The method is covered by [US11966666B2](https://patents.google.com/patent/US11966666B2/en) until 2040.

A plain offset-plus-planar-parting silicone box is a different, much older technique. Whether the patent's claims read on it is a legal question the researchers could not answer.

### 4.12 CoreCavity, full method (Nakashima et al. 2018)

**High effort, patented. Mine it for sub-techniques.**

The method decomposes a thin shell of the object into parts, each castable in a two-piece rigid mold, and the parts are then glued. It uses:

- variational shape approximation (VSA) with a moldability criterion;
- active contours;
- graph-cut parting lines;
- as-rigid-as-possible (ARAP) deformation;
- swept volumes.

Its useful defaults are 162 icosphere directions and a 1° draft. Gates go at the lowest and locally highest parting-line points ([CoreCavity PDF](https://research-explorer.ista.ac.at/download/12/5360/IST-2018-1037-v1%2B1_CoreCavity-AuthorVersion.pdf)). The method needs shells and gluing, is patented until 2041 ([US11931922B2](https://patents.google.com/patent/US11931922)), and has no code.

The supplemental ZIP contains high- and low-resolution meshes, decompositions and top/bottom mold STLs for several models. It could serve as a regression benchmark, but its license is not stated ([ISTA supplemental](https://research-explorer.ista.ac.at/download/12/5361/IST-2018-1037-v1%2B2_CoreCavity-Supplemental.zip)).

### 4.13 FlexMolds (Malomo et al. 2016)

**High effort, patented. Not a fit.**

FlexMolds designs a single-piece flexible printed shell with cuts. It greedily closes cuts from a dense quad layout, guided by a dynamic extraction simulation. The cuts must be sealed by hand, and thin shells deform for large objects ([VCG page](https://cnr-isti-vclab.github.io/publication/2016/MPBC16/); [STAR §5](https://research-explorer.app.ist.ac.at/download/11993/11994/star_molding_preprint.pdf)). There is no code, and [EP3301597B1](https://patents.google.com/patent/EP3301597B1/en) runs to about 2036.

### 4.14 Shape-deforming castability (Stein, Jacobson and Grinspun 2019)

**Medium-high effort, low value.**

The method deforms a shape with ARAP until a single planar cut can cast it, and was demonstrated with gummy candy ([project page](https://odedstein.com/projects/casting/)). Changing the user's model contradicts a replica tool. The castability energy could at most serve as a diagnostic score. No code exists.

### 4.15 Other 2022-2026 work

**Mostly low value.** The survey's citation graph shows little new graphics-venue work on automatic rigid molds for meshes since 2022 ([S2 citations](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1111/cgf.14581/citations)). This is absence of evidence, not proof.

| Work | What it is | Use for us |
|---|---|---|
| **AIMold** (ECCV 2026) | Predicts demolding orientations and parting surfaces on **B-rep CAD for injection molds**. Introduces the MoldCAD dataset: 4,934 CAD models and more than 3,850 mold assemblies ([arXiv](https://arxiv.org/abs/2608.00800)). The repo has no license ([GitHub](https://github.com/tb2-sy/AIMold)) | At most a benchmark for pull-direction choice |
| **PackMolds** (2024) | Optimizes thermoforming molds to user-set draft constraints ([Springer](https://link.springer.com/article/10.1007/s00371-024-03462-8)) | Only for a vacuum-forming mode |
| **Strain-field fabric formwork** (2026) | Fabric formwork segmentation ([DOI](https://doi.org/10.1111/cgf.70367)) | Needs cloth simulation |
| **Eggshell** | 1.5 mm printed formwork for concrete ([Liebert](https://www.liebertpub.com/doi/pdf/10.1089/3dp.2019.0197)) | Supports a thin break-away shell mode |
| **Stiver et al.** (2025) | Dovetail-jointed segmented sand-printed molds ([Springer](https://link.springer.com/content/pdf/10.1007/s40962-025-01757-7.pdf)) | Suggests joinery for splitting molds to fit the print bed |
| **ChewTect** (DIS 2026) | Silicone molds whose internal structure controls food texture ([DOI](https://doi.org/10.1145/3800645.3812893)) | Niche |

No "differentiable casting" or neural moldability paper was found in graphics venues.

## 5. Casting and material defaults table with sources: vendor data covers resins and gypsum, inference covers the rest

### 5a. Mold geometry defaults

| Parameter | Suggested default | Basis and evidence grade | Sources |
|---|---|---|---|
| Key/pin clearance, FDM | **0.2 mm per side** (0.15 on tuned Bambu/Prusa-class machines, 0.3 on older Bowden printers) | Lab test results: peg/hole alignment 0.1-0.15 mm per side on a Bambu H2D Pro, 0.2 mm per side on an Ultimaker S7. Older service-bureau tables say 0.5 mm (FDM) and 0.2 mm (SLA) for interlocking joints, probably total clearance. A caster said "0.1mm pin-hole clearance is a universal FFF target" (secondhand). Default choice is an inference | [UAL CCI lab wiki](https://wiki.cci.arts.ac.uk/books/digital-fabrication-lab/page/design-the-clearance-and-tolerance-readme); [Hubs](https://www.hubs.com/knowledge-base/how-design-interlocking-joints-fastening-3d-printed-parts/); [Prusa: movable parts ≥0.3 mm](https://help.prusa3d.com/article/modeling-with-3d-printing-in-mind_164135); [matta174 ROADMAP](https://github.com/matta174/mold-maker/blob/main/ROADMAP.md) |
| Key/pin clearance, SLA | **0.1 mm per side** | Formlabs uses a "~0.1 mm offset gap" and pins of "about 1.25 mm" on SLA silicone molds (vendor) | [Formlabs silicone white paper](https://formlabs.com/white-papers/silicone-part-production-with-3d-printed-tools/) |
| Clearance units | Absolute mm, never a percentage of model size | Caster feedback via matta174 (secondhand) | [matta174 ROADMAP](https://github.com/matta174/mold-maker/blob/main/ROADMAP.md) |
| Key shape | Truncated cone or hemisphere, male on one half, in the flange, at least 3 keys, asymmetric layout | Tapered cones "self centre… absorbs the elephant foot" (FreeCAD Wizard). "A minimum of 3 registration marks" (snippet-only). Asymmetric layout and dovetails-only-for-silicone are inferences | [FreeCAD-Molding-Wizard](https://github.com/ryankembrey/FreeCAD-Molding-Workbench); [Product Design Online](https://productdesignonline.com/day-22-of-learn-fusion-360-in-30-days-for-complete-beginners-2023-edition-create-3d-printable-two-part-molds/) |
| Key size and taper | Diameter about 8-12% of the shorter flange dimension, clamped to 4-20 mm; depth about 0.5× diameter; 10-20° flank taper | **Inference, unverified; needs a test print.** Only anchors: commercial plaster mold keys in 3/8, 5/8 and 1 in sizes. RUUIPON's "3-6 mm depth, 6-12 mm diameter" figures were **not on the fetched page; do not cite** | [Ceramic Shop 3/8"](https://www.theceramicshop.com/product/5012/plastic-notch-mold-key-3-8set/); [RUUIPON](https://www.ruuipon.com/blog/silicone-mold-registration-keys-two-part/) |
| Sprue width | Resins, soap, chocolate: **about 10 mm**. Plaster, concrete, wax: **at least 30 mm**. Channel length at least 30-50 mm | Prusa (vendor): "at least 3 cm" wide for plaster, concrete and wax; "1 cm… sufficient" for others | [Prusa beginner's guide](https://blog.prusa3d.com/the-beginners-guide-to-mold-making-and-casting_31561/) |
| Sprue, low-melt metals | 6-10 mm tapered, with a pour cup about 2× the sprue diameter at the mouth | Weakly supported; the "3-10 mm tapered" figure is unattributed. Cup ratio is an inference | [Formlabs pewter guide](https://formlabs.com/blog/metal-miniatures-3d-printed-pewter-casting-molds/) |
| Vent diameter | **2 mm minimum on FDM, 1-1.5 mm on SLA** for gravity pours | Formlabs gives 0.5-2 mm for syringe-filled silicone and 0.05 mm-deep slots for injection (vendor). Practitioners report vents down to about 0.8 mm (snippet-only). FDM minimum is an inference | [Formlabs silicone white paper](https://formlabs.com/white-papers/silicone-part-production-with-3d-printed-tools/); [Formlabs injection FAQ](https://formlabs.com/blog/3d-printed-injection-molds-faq/); [Hackaday](https://hackaday.com/2016/02/09/learn-resin-casting-techniques-duplicating-plastic-parts/) |
| Vent placement | Every significant local maximum in the pour orientation, run straight to the top surface, plus one vent above the sprue junction | Vendor and practitioner guidance agree. Smooth-On: the mold is full when liquid "becomes visible at the top of each vent hole" | [Smooth-On venting](https://www.smooth-on.com/tutorials/venting-silicone-mold-eliminate-bubbles-finished-casting/); [Prusa](https://blog.prusa3d.com/the-beginners-guide-to-mold-making-and-casting_31561/); [Juxtamorph](https://juxtamorph.com/the-resin-faq-casting-resins-into-molds/) |
| Draft (report threshold) | 2° smooth SLA; 3° sealed FDM; 5°+ raw FDM, plaster, concrete, soap; +1° per 25 mm of draw beyond 50 mm | Formlabs: "at least" 2° for SLA silicone molds, 2-5° and 3-5° for printed injection molds, 10-20° in one large-mold case study. Per-material split and deep-draw rule are inferences (injection rule snippet-only). One generator's "2.0° resin → 4.0° cold-process soap" is from a competitor with AI-assisted guides | [Formlabs silicone white paper](https://formlabs.com/white-papers/silicone-part-production-with-3d-printed-tools/); [Formlabs injection FAQ](https://formlabs.com/blog/3d-printed-injection-molds-faq/); [Formlabs injection white paper](https://formlabs.com/white-papers/low-volume-rapid-injection-molding-with-3d-printed-molds/); [Moldforge (3dplotter, competitor)](https://3dplotter.xyz/moldforge) |
| Mold stock and walls | At least 10 mm stock around the cavity, never under 2 mm anywhere; 3 mm uniform shell for metals | Formlabs: "at least 1 cm" stock; surfaces "less than 1-2 mm may deform with heat"; "uniformly 3 mm" for pewter (vendor). Concrete: "1cm wall" plus edge braces (blog) | [Formlabs silicone white paper](https://formlabs.com/white-papers/silicone-part-production-with-3d-printed-tools/); [Formlabs injection FAQ](https://formlabs.com/blog/3d-printed-injection-molds-faq/); [Formlabs pewter](https://formlabs.com/blog/metal-miniatures-3d-printed-pewter-casting-molds/); [MakeUseOf](https://www.makeuseof.com/3d-printing-concrete-molds-tips-tricks/) |
| Pry slots, flange, clamping | Pry slots 5 mm deep into the mold edge; a 10-15 mm flange carrying keys, bolt holes and band notches | Pry slots are from Formlabs (vendor). Flange width is an inference. Gravity head is only about 1-2.4 kPa per 100 mm, so stiffness, not infill, matters (inference) | [Formlabs silicone white paper](https://formlabs.com/white-papers/silicone-part-production-with-3d-printed-tools/) |
| Print settings to report | At least 4 perimeters, 20-40% infill, solid for metals or pressure pots; print parting face down | "4 perimeter walls made the shell sturdy enough" (Prusa, vendor). Infill and orientation are inferences | [Prusa](https://blog.prusa3d.com/the-beginners-guide-to-mold-making-and-casting_31561/) |
| Silicone mold-box mode | 12.7 mm (1/2 in) minimum model-to-wall gap; raise the model 1/2 in; 1/2 in of silicone over the high point; 2 mm minimum silicone shell; degas above 18,000 cps | Smooth-On and Formlabs (vendor). The 1/2 in over the high point is snippet-only | [Smooth-On mold box](https://www.smooth-on.com/tutorials/venting-silicone-mold-eliminate-bubbles-finished-casting/constructing-mold-box/); [Smooth-On Mold Star 15](https://www.smooth-on.com/tutorials/mold-making-minute-mold-star-15/); [Formlabs silicone white paper](https://formlabs.com/white-papers/silicone-part-production-with-3d-printed-tools/) |

**Two of the repo's current defaults conflict with this table.**

- **Sprue too narrow.** The resin preset's 6 mm sprue is below Prusa's roughly 1 cm. The geometry researcher's inferred "8-12 mm for wax" also contradicts Prusa, who puts wax in the 3 cm poor-flow group ([Prusa](https://blog.prusa3d.com/the-beginners-guide-to-mold-making-and-casting_31561/)). Wax also needs headspace for 12-15% volumetric shrink, which argues for a wide sprue that doubles as a reservoir.
- **Clearance unit unclear.** The repo's 0.25 mm clearance default sits inside the FDM range, but it should be labeled per side or on diameter.

### 5b. Casting materials: shrinkage, heat, minimum mold material, release

Shrinkage values are linear unless marked "vol". For volume-to-linear conversion, s_lin ≈ s_vol/3 (inference; valid only for isotropic, unconstrained shrinkage). The cavity scale is 1/(1 − s_lin). Scale by default only when |s_lin| ≥ 0.3% (inference).

| Casting material | Shrinkage | Pour / peak temperature | Minimum printed mold (HDT − 10 °C rule, inference) | Release (FDM/SLA rigid mold) | Sources |
|---|---|---|---|---|---|
| Smooth-Cast 300 / 325 PU | **1.0%** (TDS; 325 TDS-snippet) | Room temperature. Thin castings mild; large mass "hot to the touch" | PLA for thin casts with a barrier release; PETG or ABS for thick casts (patent PU example peaked at about 150 °C at 1/4 in; not Smooth-Cast) | Seal layer lines, then Ease Release 200 brushed plus a mist. PU "may bond very strongly to PLA" (anecdotal; attribution uncertain) | [SC-300](https://www.smooth-on.com/products/smooth-cast-300/); [SC-325](https://www.smooth-on.com/products/smooth-cast-325/); [US 4,036,820](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/4036820); [Mann ER200](https://www.mann-release.com/products/ease-release/200/); [MIT Fab](https://fab.cba.mit.edu/classes/863.25/people/EghosaOhenhen/week10.html) |
| Smooth-Cast 305 PU | **0.65%** (TDS) | Room temperature | As above | As above | [SC-305](https://www.smooth-on.com/products/smooth-cast-305/) |
| Clear epoxy (EpoxAcast 690) | **0.2%** (TDS-snippet) | Maximum thickness 0.95 cm per pour, 1.9 cm with fan cooling. 100 g of generic epoxy can reach 204 °C; deep pours self-heat 93-204 °C | Thin pours: PETG and up (PLA marginal). Thick or fast pours: PA-CF or thermally post-cured High Temp SLA only | Paste wax plus PVA film (Partall #2 + #10), or a sealed surface plus Ease Release 200. Anecdotal reports of epoxy stuck in PLA | [EpoxAcast 690 TB](https://www.smooth-on.com/tb/files/EPOXACAST_690_TB.pdf); [West System](https://www.westsystem.com/safety/uncontrolled-cure/); [MAS (snippet)](https://masepoxies.com/blogs/mas-blog/why-is-my-epoxy-resin-hot-and-smoking); [Fiberglass Warehouse](https://www.fiberglasswarehouse.com/blogs/news/fiberglass-mold-release); [Penturners (anecdotal)](https://www.penturners.org/threads/stuck-epoxy-blank-in-3d-printed-mold.186159/) |
| Generic epoxy / polyester | Epoxy 2-7% vol (some sources say 1-3%); polyester 5-12% vol (secondary, snippet) | Polyester exotherm and styrene attack: not researched | Polyester: PETG at best, marginal (unverified) | Wax plus PVA | [ResearchGate Q&A](https://www.researchgate.net/post/Epoxy_vs_Polyester_resin_against_volumetric_shrinkage_during_curing_process); [ScienceDirect topic](https://www.sciencedirect.com/topics/engineering/resin-shrinkage) |
| Platinum silicone (Mold Star 30) | **<0.1%** (TDS) | Room temperature; heat cure at 60 °C | PLA works (does not inhibit PDMS; cured at 60 °C). **SLA inhibits** unless fully post-cured and sealed | On PLA usually none. On SLA: at least 6 h UV post-cure, XTC-3D or Krylon acrylic sealer, then Ease Release 200; Inhibit X; or switch to tin silicone | [Mold Star 30](https://www.smooth-on.com/products/mold-star-30/); [PMC10059908](https://pmc.ncbi.nlm.nih.gov/articles/PMC10059908/); [Smooth-On FAQ 210](https://www.smooth-on.com/support/faq/210/) |
| Tin silicone (Mold Max 30) | **0.2%** (TDS) | Room temperature | Any | None, or Ease Release 200 | [Mold Max 30](https://www.smooth-on.com/products/mold-max-30/) |
| Gypsum: Hydrocal White / Ultracal 30 | **Expands 0.42% / 0.08%** (TDS-snippet); scale ≤1 | Room temperature; set exotherm unquantified. Plaster of Paris burn reports (anecdotal) | Any. Expansion can lock an undrafted rigid mold (inference) | Murphy's Oil Soap or petroleum jelly (anecdotal); Ease Release 205 on sealed plaster | [USG Hydrocal White](https://www.usg.com/content/dam/USG/pdpmovedocuments/hydrocal-white-gypsum-cement-data-en-IG1381.pdf); [USG Ultracal 30](https://www.usg.com/content/dam/USG/pdpmovedocuments/usg-ultracal-30-southard-submittal-en-IG1939.pdf); [potters.org](http://www.potters.org/subject84275.htm/); [ER205](https://www.smooth-on.com/product-line/ease-release/) |
| Jesmonite AC100 | **Expands 0.15%** (TDS-snippet) | Room temperature; initial set 15-20 min at 18 °C | Any | As for gypsum | [Jesmonite TDS](https://jesmonite.com/wp-content/uploads/2023/05/AC100-Technical-Data-Sheet-2023.pdf) |
| Concrete | **0.04-0.08%** drying shrinkage (secondary) | Room temperature, **alkaline** | PLA marginal: ester hydrolysis accelerates in alkali, worst at 50 °C. PETG marginal; ABS/ASA preferred (alkali resistance of these unverified) | Form oil or petroleum jelly (**no primary source**) | [BASF](https://insights.basf.com/files/pdf/Shrinkage_Of_Concrete_CTIF.pdf); [NRMCA TIP 17](https://www.nrmca.org/wp-content/uploads/2020/04/TIP17w.pdf); [Applied Sciences 2025](https://doi.org/10.3390/app15042165) |
| Paraffin pillar wax | **12-15% vol** for straight paraffin (patent), about 4-5% linear (inference); shows up as sinkholes | Pour at 82 °C, second pour at 88 °C; melts at 57 °C | ASA or PC (ABS marginal). **Not PLA or PETG** | None: "came out of the molds very easily" | [US20050086853A1](https://patents.google.com/patent/US20050086853A1/en); [CandleScience MP-137](https://www.candlescience.com/lab-notes-paraffin-pillar-wax-mp-137/) |
| Soy pillar wax / beeswax | Shrinks enough to release (supplier, qualitative) | Soy: 60-77 °C, recommended 71 °C. Beeswax: 63-71 °C (anecdotal) | ABS and up (PETG marginal) | None (contraction) | [CandleScience EcoSoya PB](https://www.candlescience.com/learning/lab-notes-ecosoya-pb-pillar-soy-wax/); [makercalculators (anecdotal)](https://makercalculators.com/articles/candle-pouring-temperature-by-wax-type) |
| Melt-and-pour soap | **No shrinkage figure found** | Pour 52-66 °C; detail pours 52-54 °C; do not exceed 71 °C | PETG marginal, ABS and up OK (PLA only for pours ≤ about 52 °C) | None usually (**no source**) | [Bramble Berry](https://www.brambleberry.com/beginner/beginners-guide-to-melt-and-pour.html) |
| Cold-process soap | **No shrinkage figure found** | Gel phase up to 82 °C, plus free NaOH. Soap at 32-38 °C to avoid gel | Avoid PLA and PETG (PETG "unstable" in 30%+ NaOH; alkali hydrolysis). A liner or silicone insert is suggested (**no source**) | Liner | [Soap Queen](https://soapqueen.com/bath-and-body-tutorials/tips-and-tricks/gel-phase/); [ERIKS PETG chart](https://solutions-in-plastics.info/nl-be/datasheets/transparante%20kunststoffen/eriks%20-%20petg%20chemical%20resistance.pdf) |
| Chocolate (tempered) | Dark about 2% vol; milk 0.1-0.5% vol; up to about 2% linear when well seeded (secondary) | Work at 28-32 °C | Any thermally. **Food contact:** layer lines harbor bacteria, so prefer a printed master cast into food-grade silicone | None if tempered ("under-tempered chocolate… will stick", anecdotal); polished or sealed mold | [Callebaut](https://www.callebaut.com/en-US/chocolate-video/technique/tempering/callets); [New Food Magazine](https://www.newfoodmagazine.com/article/2048/chocolate-cooling-and-demoulding/); [Prusa food safety](https://blog.prusa3d.com/how-to-make-food-grade-3d-printed-models_40666/); [Formlabs food safety](https://formlabs.com/blog/guide-to-food-safe-3d-printing/) |
| Isomalt | n/r | Pour at about 125 °C (anecdotal) | Only thermally post-cured High Temp SLA or PA-CF (marginal), plus the food caveat | n/r | [Sugar Arts Institute (anecdotal)](https://www.sugarartsinstitute.com/blog/post/isomalt--the-secrets-of-success-isomalt--how-to-cook-hold-pour-and-store-isomalt) |
| Cerrolow 117 / Field's metal | About 0 (secondary) | Melt at 47 °C / 62 °C (pour hotter) | Cerrolow 117: PETG. Field's: ABS and up | Graphite powder; oil (anecdotal) | [Wikipedia fusible alloy](https://en.wikipedia.org/wiki/Fusible_alloy); [Field's metal](https://en.wikipedia.org/wiki/Field%27s_metal) |
| Wood's metal / Cerrobend | About 0: "lack of contraction" (secondary). **Contains Pb and Cd** | Melts at about 70 °C | ASA and up. ABS marginal: it deformed when overheated. TPU "worked best" (anecdotal) | Commercial release; oil "would probably work" (anecdotal) | [Wood's metal](https://en.wikipedia.org/wiki/Wood%27s_metal); [thrinter (anecdotal)](http://thrinter.com/metal-casting-with-a-plastic-printer/) |
| Rose's metal | About 0 (secondary); contains Pb | Melts at 94-98 °C | PC marginal; PA-CF or High Temp SLA | Graphite | [Rose's metal](https://en.wikipedia.org/wiki/Rose%27s_metal) |
| Pewter | n/r | About 260 °C (Formlabs); retail ranges 241-343 °C (snippet) | Only High Temp SLA after thermal post-cure, with a 3 mm shell (marginal, vendor demo). Otherwise plaster investment | **Graphite powder** (vendor) | [Formlabs pewter](https://formlabs.com/blog/metal-miniatures-3d-printed-pewter-casting-molds/); [RotoMetals](https://www.rotometals.com/pewter-alloys/) |

### 5c. Printable mold materials: heat ceiling

| Mold material | HDT at 0.45 MPa | Suggested max mold-face temperature (HDT − 10 °C, inference) | Notes | Sources |
|---|---|---|---|---|
| PLA | 55-57 °C (TDS-snippet) | **45 °C** | The repo's current 50 °C is slightly optimistic under this rule | [Prusament PLA](https://prusament.com/wp-content/uploads/2022/10/PLA_Prusament_TDS_2021_10_EN.pdf); [Bambu PLA](https://wiki.bambulab.com/filament-acc/abs-asa-pc/bambu_pla_basic_technical_data_sheet.pdf) |
| PETG | 68-71 °C (TDS-snippet) | **58-60 °C** | Weak in strong alkali | [Prusament PETG](https://prusament.com/wp-content/uploads/2023/07/PETG_V0_ENG.pdf); [Bambu PETG](https://store.bblcdn.com/s1/default/cb94589bf7994fdcbfa833badefae9cd/Bambu_PETG_Basic_Technical_Data_Sheet.pdf) |
| ABS | 87 °C (TDS-snippet) | **77 °C** | n/a | [Bambu ABS (Scribd mirror)](https://www.scribd.com/document/801170457/Bambu-ABS-Technical-Data-Sheet-V3) |
| ASA | 93-100 °C (TDS-snippet) | **83-90 °C** | n/a | [Prusament ASA](https://prusament.com/wp-content/uploads/2022/10/ASA_Prusament_TDS_2022_16_EN.pdf); [Bambu ASA](https://www.additive-x.com/shop/media/mageplaza/product_attachments/attachment_file/b/a/bambu_asa_technical_data_sheet_1.pdf) |
| PC / PC blend | 112-113 °C | **100 °C** | Bambu PC's 1.8 MPa value exceeds its 0.45 MPa value, a likely extraction error | [Prusament PC Blend](https://prusament.com/wp-content/uploads/2022/10/PCBlend_Prusament_TDS_2022_16_EN.pdf); [Bambu PC](https://wiki.bambulab.com/filament-acc/abs-asa-pc/a52afdccddfd448583d119587122c8c5.pdf) |
| PA6-CF / PAHT-CF | 186-194 °C (retailer/store snippet) | **175 °C** | n/a | [MatterHackers](https://www.matterhackers.com/store/l/bambu-lab-pa6-cf-filament/sk/MC64KFKY); [Bambu PAHT-CF](https://bambulab-us.myshopify.com/products/paht-cf) |
| Annealed HTPLA | "Up to 155 °C" (manufacturer claim, not ISO 75) | **140 °C** | Annealing shrinks the part 1-3% in X/Y; compound this with cast shrinkage | [Proto-pasta](https://proto-pasta.com/pages/high-temp-pla) |
| Standard SLA (Formlabs Grey V4) | 73 °C (reseller snippet) | **60 °C** | Inhibits platinum silicone unless fully cured | [Dynamism](https://www.dynamism.com/formlabs/formlabs-resin-greyv4.html) |
| Tough SLA (Tough 2000) | 70 °C (TDS) | **60 °C** | n/a | [Formlabs Tough 2000](https://formlabs.com/store/materials/tough-2000-resin/) |
| Formlabs High Temp | 49 °C green; 120 °C after a 60 °C cure; **238 °C after thermal post-cure** (TDS) | **110 °C / 225 °C** | Thermal post-cure is a precondition, not an option. Rigid 10K also shows 238 °C, which needs checking | [Formlabs High Temp](https://formlabs.com/store/materials/high-temp-resin/); [Rigid 10K](https://formlabs.com/store/materials/rigid-10k-resin/) |
| Siraya Tech Sculpt | 180 °C at 0.455 MPa (manufacturer) | **165 °C** | Dry before use above 100 °C | [Siraya Sculpt guide](https://siraya.tech/pages/sculpt-user-guide) |
| TPU | Not listed | Treat like PLA/PETG | Worked with Wood's metal (anecdotal) | [Bambu TPU TDS](https://store.bblcdn.eu/s8/default/16df21baf482453999b3dbb61cc110e7/Bambu_TPU_95A_HF_Technical_Data_Sheet.pdf); [thrinter](http://thrinter.com/metal-casting-with-a-plastic-printer/) |

**What the data implies for the tool (inference built on the cited numbers):**

- **Temperature, not chemistry, is the binding constraint** for wax, soap, metals, isomalt and thick epoxy. PLA is only safe for chocolate, gelatin, silicone, gypsum, Jesmonite and thin resin pours.
- **Chemistry decides three cases:**
  - alkali (cold-process soap, concrete) against PLA and PETG;
  - under-cured SLA against platinum silicone;
  - epoxy and PU keying into FDM layer lines unless a barrier is applied ([Smooth-On FAQ 210](https://www.smooth-on.com/support/faq/210/); [Applied Sciences 2025](https://doi.org/10.3390/app15042165)).
- **Use a reservoir, not scaling, for volumetric shrinkers.** Wax, polyester and thick epoxy shrink mostly as sinks and voids, so the tool should add a riser or second-pour headspace rather than rely on cavity scaling ([CandleScience](https://www.candlescience.com/lab-notes-paraffin-pillar-wax-mp-137/)).
- **Make the exotherm check thickness-aware.** Compare the cavity's largest inscribed thickness against limits such as EpoxAcast's 0.95 cm ([EpoxAcast 690 TB](https://www.smooth-on.com/tb/files/EPOXACAST_690_TB.pdf)).

**Gaps remain:**

- published peak exotherm temperatures for Smooth-Cast and EpoxAcast;
- gypsum set temperature;
- soap shrinkage;
- jewelry-wax shrinkage;
- primary sources for concrete and soap release;
- chemical resistance of ABS, ASA, PC and TPU to alkali.

## 6. Top 10 feature recommendations ranked by user value vs. implementation effort

The ranking puts features with a high value-to-effort ratio first. Items 1-7 make the two-piece core best-in-class and defensible within weeks. Items 8-10 are the larger bets that create the moat.

| Rank | Feature | User value | Effort | Why (evidence) | Concrete hook in the repo |
|---|---|---|---|---|---|
| 1 | **Vent and sprue planner based on cavity topology.** Local maxima in the pour frame, filtered by 0-dimensional persistence. Choose among ±X/±Y the orientation with the fewest significant maxima. Vents as half-channels in the parting plane. Recompute after adding the sprue | Very high | Low | Theorem plus O(n) test ([Bose et al. 1998](https://cgm.cs.mcgill.ca/~godfried/publications/casting3D.ps.gz)); vents on the parting surface when pour is perpendicular to pull ([AMB21](http://openportal.isti.cnr.it/data/2021/456702/2021_456702.postprint.pdf)); persistence ranking ([Volume-aware](https://opus.lib.uts.edu.au/bitstream/10453/137699/4/composite_compressed.pdf)). Casters reject bounding-box vents ([matta174 ROADMAP](https://github.com/matta174/mold-maker/blob/main/ROADMAP.md), secondhand). No open tool places vents by cavity topology | Now drafted in the uncommitted working tree (`gating._air_traps`, `_plan_vents`): prominence-filtered traps, four pour orientations scored, vents snapped to the parting plane. Remaining: recompute after sprue insertion, regression tests, a test print. The in-plane sprue and ±X/±Y pour convention already satisfy the AMB21 condition |
| 2 | **Material and printer presets with sourced absolute-mm defaults, plus a compatibility gate.** Check HDT − 10 °C against pour or exotherm temperature, alkali, Pt-silicone-on-SLA and the exotherm thickness limit. Add shrinkage scaling and release-agent advice to the report | High | Low | Tables 5a-5c. Among maker tools only 3DMold claims shrinkage ([App Store](https://apps.apple.com/us/app/3dmold-3d-mold-maker-stl/id6758258851)). Casters want absolute mm ([ROADMAP](https://github.com/matta174/mold-maker/blob/main/ROADMAP.md)) | `materials.py` "placeholder presets" (one resin preset, PLA at 50 °C); `config.shrinkage` |
| 3 | **Harden the direction search and add an undercut/draft report.** Add ε tolerance (0.4 mm), an undercut-volume objective by line stabbing, separate plane-induced from intrinsic undercuts, export a per-face draft heatmap as PLY, and add the g+1 silhouette-loop validator | High | Low | [Ahn Lemma 1](https://webdoc.sub.gwdg.de/ebook/serien/ah/UU-CS/2001-48.pdf); [AMB21 ε](http://openportal.isti.cnr.it/data/2021/456702/2021_456702.postprint.pdf); [Elber](https://gershon.cs.technion.ac.il/papers/MoldAccessibility.pdf). Competitors market heatmaps ([Action BOX](https://mold.actionbox.ca/)) | `parting._search_directions`, `best_offset`, `classify_faces` (planar search now present in the working tree) |
| 4 | **"Always demoldable" fallback.** Build halves from depth maps (height fields) that fill residual undercuts, and report the detail lost | High | Low-medium | [CoreCavity §6.2](https://research-explorer.ista.ac.at/download/12/5360/IST-2018-1037-v1%2B1_CoreCavity-AuthorVersion.pdf); [Protective Foam](https://arxiv.org/abs/2503.09019); Meshcast's "Fill the undercuts" ([Meshcast](https://meshcast.app/mold)) | Between parting analysis and `booleans.py` |
| 5 | **Pip-installable library and CLI** with batch mode and a JSON/HTML validation report | Medium-high | Low | No mold-generator package exists on PyPI ([PyPI](https://pypi.org/pypi/pymold/json)); add-ons are locked to their host ([MoldForge](https://extensions.blender.org/add-ons/moldforge/)) | `pyproject.toml`, `__main__.py`, `report.py` |
| 6 | **Mesh repair and output validation.** Ensure a watertight single solid and never write a broken STL. Use a decimated proxy for analysis and full resolution for Booleans | High | Medium | Repair is "the #1 support burden we can predict" ([competitive analysis](https://github.com/matta174/mold-maker/blob/main/docs/competitive-analysis.md)); MoldForge's practice ([README](https://github.com/Plesuro/MoldForge)); Mathew3000 has no repair ([README](https://github.com/Mathew3000/moldgen)) | `repair.py`, `meshio.py` |
| 7 | **Practitioner kit features.** Asymmetric tapered cone keys in a flange (0.2 mm per side FDM, 0.1 mm SLA); 5 mm pry slots; bolt holes or heat-set inserts; band notches; optional seal bead or tongue-and-groove; parting-face-down export; split into sectors to fit the print bed; cavity volume and material mass; coating allowance | Medium-high | Low-medium | [FreeCAD-Molding-Wizard](https://github.com/ryankembrey/FreeCAD-Molding-Workbench); [Shape Cast](http://www.emmielyons.com/pubs/shapecast-chi24.pdf); [Formlabs pry slots](https://formlabs.com/white-papers/silicone-part-production-with-3d-printed-tools/) | `keys.py`, `pipeline.py` |
| 8 | **Non-planar parting surfaces.** Use Gupta's ρ to choose planar, stepped (interval stabbing) or terrain/RBF surfaces through the outermost silhouette loop | High for organic parts | Medium | Only jocam3d ships curved parting ([jocam3d](https://jocam3d.com/en/tools/stl-to-mold)); matta174 estimates 15-25 days ([competitive analysis](https://github.com/matta174/mold-maker/blob/main/docs/competitive-analysis.md)); [Khardekar-McMains](https://mcmains.me.berkeley.edu/pubs/SPM09khardekarMcMains.pdf) | `parting.mold_frame` and the block split |
| 9 | **Undercut-driven multi-piece rigid molds (3+ pieces).** Accessibility matrix, greedy set cover of directions, planar cuts, disassembly-order check, keys on every interface, report of faces needing a core | Very high (the moat) | High | The gap all three researchers identified; [Priyadarshi-Gupta](https://www.sciencedirect.com/science/article/abs/pii/S0010448503001076); the only open precedent, HansTydecks, is unlicensed ([repo](https://github.com/HansTydecks/Concrete-Mold-Creator)) | New module consuming the per-face classes from `parting.py` |
| 10 | **Silicone mother-mold / mold-box mode** using Smooth-On clearances and a 15 mm liner default | High (largest audience per caster feedback) | Medium | "Matrix mold" is the "single most important piece of direction" from a caster (secondhand, [ROADMAP](https://github.com/matta174/mold-maker/blob/main/ROADMAP.md)). But the space is crowded ([seena18](https://github.com/seena18/mothermold), [MoldForge](https://github.com/Plesuro/MoldForge), [Meshcast](https://meshcast.app/silicone-mold-maker)) and close to the Metamolds patent if it grows into automatic segmentation ([US11966666B2](https://patents.google.com/patent/US11966666B2/en)) | Separate pipeline that reuses parting and keys |

**Three strategic conclusions.**

1. **The best ideas for an open mold generator are the oldest.** The 1995-2004 computational-geometry results on castability, gravity filling and accessibility-driven multi-piece molds are proven, cheap to express in numpy and trimesh, absent from every open tool, and outside the patents that fence off the 2016-2019 graphics methods. This project can therefore lead on correctness rather than on features, and say so in the documentation with citations.
2. **The window is measured in months.** Mold-generator repos and add-ons now appear weekly, many built with AI assistance. Ship items 1-5 quickly under MIT on PyPI.
3. **Calibrate the defaults on real prints.** Key geometry, vent diameter for gravity pours, soap shrinkage and concrete release still rest on inference. A printable clearance-and-vent test coupon plus a few documented test casts would turn this report's "inference" labels into the field's first open, measured defaults.

The CoreCavity supplemental STLs and AIMold's MoldCAD are candidate regression benchmarks, once their licenses are clarified ([CoreCavity supplemental](https://research-explorer.ista.ac.at/download/12/5361/IST-2018-1037-v1%2B2_CoreCavity-Supplemental.zip); [AIMold](https://arxiv.org/abs/2608.00800)).
