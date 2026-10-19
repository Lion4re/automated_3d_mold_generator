"""Casting material and print material presets.

The numbers are starting points for hobby and workshop casting, taken from
manufacturer data sheets and casting guides where available (see each
preset's ``sources``). Always check the data sheet of the exact product you use.

Shrinkage is linear (fraction of length; negative means the material expands
while setting). Temperatures are in degrees Celsius.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    description: str
    linear_shrinkage: float
    """Fraction the cast shrinks per unit length while curing or cooling."""
    pour_temp_c: tuple[float, float] | None
    """Typical pour temperature range, or None for room-temperature materials."""
    sprue_diameter_mm: float
    vent_diameter_mm: float
    min_wall_mm: float
    release_agent: str
    notes: str = ""
    alkaline: bool = False
    """Strongly alkaline while setting (attacks polyester-based prints such as PLA and PETG)."""
    food_contact: bool = False
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrintMaterial:
    key: str
    name: str
    max_service_temp_c: float
    """Highest mold-face temperature before the print softens: heat deflection temperature minus 10 °C."""
    density_g_cm3: float = 1.24
    alkali_resistant: bool | None = None
    """None when no reliable data was found."""
    inhibits_platinum_silicone: bool = False
    notes: str = ""
    sources: tuple[str, ...] = ()


_PRUSA_GUIDE = "https://blog.prusa3d.com/the-beginners-guide-to-mold-making-and-casting_31561/"
_FORMLABS_SILICONE = (
    "https://formlabs.com/white-papers/silicone-part-production-with-3d-printed-tools/"
)

MATERIALS: dict[str, Material] = {
    m.key: m
    for m in [
        Material(
            key="resin",
            name="Polyurethane casting resin",
            description="Fast two-part urethane resins (for example Smooth-Cast 300/305).",
            linear_shrinkage=0.01,
            pour_temp_c=None,
            sprue_diameter_mm=10.0,
            vent_diameter_mm=2.0,
            min_wall_mm=4.0,
            release_agent=(
                "seal the layer lines (for example with XTC-3D), then a release such as "
                "Ease Release 200; urethane can bond strongly to bare PLA"
            ),
            notes="Thick casts get hot while curing; use PETG or ABS molds for bulky parts.",
            sources=(
                "https://www.smooth-on.com/products/smooth-cast-300/",
                "https://www.smooth-on.com/products/smooth-cast-305/",
            ),
        ),
        Material(
            key="epoxy",
            name="Epoxy casting resin",
            description="Clear or filled epoxy resins (for example EpoxAcast 690).",
            linear_shrinkage=0.002,
            pour_temp_c=None,
            sprue_diameter_mm=10.0,
            vent_diameter_mm=2.0,
            min_wall_mm=4.0,
            release_agent="paste wax plus PVA film, or a sealed surface plus Ease Release 200",
            notes=(
                "Epoxy self-heats in thick sections; pour layers no thicker than about 1 cm "
                "unless the resin is rated for deep pours."
            ),
            sources=(
                "https://www.smooth-on.com/tb/files/EPOXACAST_690_TB.pdf",
                "https://www.westsystem.com/safety/uncontrolled-cure/",
            ),
        ),
        Material(
            key="silicone",
            name="Silicone rubber",
            description="Platinum or tin cure silicone for flexible casts.",
            linear_shrinkage=0.001,
            pour_temp_c=None,
            sprue_diameter_mm=10.0,
            vent_diameter_mm=2.0,
            min_wall_mm=4.0,
            release_agent="none needed on PLA; Ease Release 200 on sealed resin prints",
            notes="Degas the mixed silicone before pouring if it is thicker than about 18,000 cps.",
            sources=(
                "https://www.smooth-on.com/products/mold-star-30/",
                "https://www.smooth-on.com/support/faq/210/",
            ),
        ),
        Material(
            key="plaster",
            name="Plaster and gypsum cement",
            description="Plaster of Paris, Hydrocal, Ultracal and similar gypsum products.",
            linear_shrinkage=-0.002,
            pour_temp_c=None,
            sprue_diameter_mm=25.0,
            vent_diameter_mm=3.0,
            min_wall_mm=6.0,
            release_agent="petroleum jelly or diluted oil soap",
            notes=(
                "Gypsum expands slightly while setting (0.08-0.42 % depending on the product), "
                "so generous draft helps release."
            ),
            sources=(
                "https://www.usg.com/content/dam/USG/pdpmovedocuments/hydrocal-white-gypsum-cement-data-en-IG1381.pdf",
                _PRUSA_GUIDE,
            ),
        ),
        Material(
            key="concrete",
            name="Concrete and mortar",
            description="Fine concrete, mortar and cement mixes.",
            linear_shrinkage=0.0006,
            pour_temp_c=None,
            sprue_diameter_mm=25.0,
            vent_diameter_mm=3.0,
            min_wall_mm=8.0,
            release_agent="form oil or petroleum jelly",
            notes="Vibrate or tap the mold after pouring to release trapped air.",
            alkaline=True,
            sources=("https://www.nrmca.org/wp-content/uploads/2020/04/TIP17w.pdf", _PRUSA_GUIDE),
        ),
        Material(
            key="wax",
            name="Paraffin wax",
            description="Candle and pillar paraffin waxes.",
            linear_shrinkage=0.01,
            pour_temp_c=(82.0, 88.0),
            sprue_diameter_mm=20.0,
            vent_diameter_mm=2.0,
            min_wall_mm=5.0,
            release_agent="usually none; the wax contracts away from the mold",
            notes=(
                "Paraffin shrinks 12-15 % by volume, mostly as a sinkhole under the sprue; "
                "top up with a second, slightly hotter pour once the first has set."
            ),
            sources=(
                "https://www.candlescience.com/lab-notes-paraffin-pillar-wax-mp-137/",
                "https://patents.google.com/patent/US20050086853A1/en",
            ),
        ),
        Material(
            key="soy-wax",
            name="Soy or beeswax",
            description="Natural pillar waxes.",
            linear_shrinkage=0.01,
            pour_temp_c=(60.0, 77.0),
            sprue_diameter_mm=20.0,
            vent_diameter_mm=2.0,
            min_wall_mm=5.0,
            release_agent="usually none; the wax contracts away from the mold",
            sources=(
                "https://www.candlescience.com/learning/lab-notes-ecosoya-pb-pillar-soy-wax/",
            ),
        ),
        Material(
            key="soap",
            name="Melt-and-pour soap",
            description="Glycerin melt-and-pour soap bases.",
            linear_shrinkage=0.0,
            pour_temp_c=(52.0, 66.0),
            sprue_diameter_mm=10.0,
            vent_diameter_mm=2.0,
            min_wall_mm=4.0,
            release_agent="usually none; a light mist of isopropyl alcohol removes surface bubbles",
            notes="Do not heat the base above 71 °C. Cold-process soap is alkaline and needs a liner.",
            sources=(
                "https://www.brambleberry.com/beginner/beginners-guide-to-melt-and-pour.html",
            ),
        ),
        Material(
            key="chocolate",
            name="Tempered chocolate",
            description="Tempered dark, milk or white chocolate.",
            linear_shrinkage=0.005,
            pour_temp_c=(28.0, 32.0),
            sprue_diameter_mm=10.0,
            vent_diameter_mm=2.0,
            min_wall_mm=3.0,
            release_agent="none; well-tempered chocolate contracts and releases on its own",
            notes="Under-tempered chocolate sticks. Chill the filled mold before demolding.",
            food_contact=True,
            sources=(
                "https://www.callebaut.com/en-US/chocolate-video/technique/tempering/callets",
                "https://formlabs.com/blog/guide-to-food-safe-3d-printing/",
            ),
        ),
        Material(
            key="low-melt-alloy",
            name="Low-melt alloy",
            description="Field's metal and similar fusible alloys that melt below 100 °C.",
            linear_shrinkage=0.0,
            pour_temp_c=(70.0, 80.0),
            sprue_diameter_mm=8.0,
            vent_diameter_mm=2.0,
            min_wall_mm=5.0,
            release_agent="graphite powder",
            notes=(
                "Wood's metal and Rose's metal contain lead (and Wood's cadmium); handle and "
                "melt them with care. Rose's metal melts at 94-98 °C."
            ),
            sources=(
                "https://en.wikipedia.org/wiki/Field%27s_metal",
                "https://en.wikipedia.org/wiki/Fusible_alloy",
            ),
        ),
        Material(
            key="pewter",
            name="Pewter",
            description="Lead-free pewter for miniatures and jewellery.",
            linear_shrinkage=0.0,
            pour_temp_c=(260.0, 300.0),
            sprue_diameter_mm=8.0,
            vent_diameter_mm=1.5,
            min_wall_mm=3.0,
            release_agent="graphite powder",
            notes=(
                "Only thermally post-cured high-temperature resin prints survive pewter; "
                "otherwise print a master and make a plaster or silicone mold from it."
            ),
            sources=(
                "https://formlabs.com/blog/metal-miniatures-3d-printed-pewter-casting-molds/",
            ),
        ),
    ]
}

PRINT_MATERIALS: dict[str, PrintMaterial] = {
    p.key: p
    for p in [
        PrintMaterial(
            key="pla",
            name="PLA",
            max_service_temp_c=45.0,
            density_g_cm3=1.24,
            alkali_resistant=False,
            sources=(
                "https://prusament.com/wp-content/uploads/2022/10/PLA_Prusament_TDS_2021_10_EN.pdf",
            ),
        ),
        PrintMaterial(
            key="petg",
            name="PETG",
            max_service_temp_c=58.0,
            density_g_cm3=1.27,
            alkali_resistant=False,
            sources=("https://prusament.com/wp-content/uploads/2023/07/PETG_V0_ENG.pdf",),
        ),
        PrintMaterial(key="abs", name="ABS", max_service_temp_c=77.0, density_g_cm3=1.04),
        PrintMaterial(
            key="asa",
            name="ASA",
            max_service_temp_c=85.0,
            density_g_cm3=1.07,
            sources=(
                "https://prusament.com/wp-content/uploads/2022/10/ASA_Prusament_TDS_2022_16_EN.pdf",
            ),
        ),
        PrintMaterial(
            key="pc",
            name="Polycarbonate (PC blend)",
            max_service_temp_c=100.0,
            density_g_cm3=1.20,
            sources=(
                "https://prusament.com/wp-content/uploads/2022/10/PCBlend_Prusament_TDS_2022_16_EN.pdf",
            ),
        ),
        PrintMaterial(
            key="pa-cf",
            name="Carbon-fibre nylon (PA-CF)",
            max_service_temp_c=175.0,
            density_g_cm3=1.20,
        ),
        PrintMaterial(
            key="sla-standard",
            name="Standard SLA resin",
            max_service_temp_c=60.0,
            density_g_cm3=1.18,
            inhibits_platinum_silicone=True,
        ),
        PrintMaterial(
            key="sla-tough",
            name="Tough SLA resin",
            max_service_temp_c=60.0,
            density_g_cm3=1.18,
            inhibits_platinum_silicone=True,
            sources=("https://formlabs.com/store/materials/tough-2000-resin/",),
        ),
        PrintMaterial(
            key="sla-hightemp",
            name="High-temperature SLA resin (thermally post-cured)",
            max_service_temp_c=225.0,
            density_g_cm3=1.20,
            inhibits_platinum_silicone=True,
            notes="Reaches this rating only after the manufacturer's thermal post-cure.",
            sources=("https://formlabs.com/store/materials/high-temp-resin/",),
        ),
    ]
}


def get_material(key: str) -> Material:
    try:
        return MATERIALS[key]
    except KeyError:
        raise KeyError(
            f"Unknown material {key!r}; choose one of {', '.join(sorted(MATERIALS))}"
        ) from None


def get_print_material(key: str) -> PrintMaterial:
    try:
        return PRINT_MATERIALS[key]
    except KeyError:
        raise KeyError(
            f"Unknown print material {key!r}; choose one of {', '.join(sorted(PRINT_MATERIALS))}"
        ) from None


def compatibility_warnings(material: Material, print_material: PrintMaterial) -> list[str]:
    """Return human-readable warnings when the printed mold may not survive the cast."""
    warnings = []
    if material.pour_temp_c is not None:
        hottest = material.pour_temp_c[1]
        if hottest > print_material.max_service_temp_c:
            suitable = sorted(
                (p for p in PRINT_MATERIALS.values() if p.max_service_temp_c >= hottest),
                key=lambda p: p.max_service_temp_c,
            )
            hint = f" Print the mold in {suitable[0].name} or better." if suitable else ""
            warnings.append(
                f"{material.name} is poured at up to {hottest:.0f} °C, but {print_material.name} "
                f"softens above about {print_material.max_service_temp_c:.0f} °C, so the mold "
                f"may deform.{hint}"
            )
    if material.alkaline and print_material.alkali_resistant is False:
        warnings.append(
            f"{material.name} is strongly alkaline and slowly attacks {print_material.name}; "
            "seal the mold or print it in ABS or ASA for repeated use."
        )
    if material.key == "silicone" and print_material.inhibits_platinum_silicone:
        warnings.append(
            f"{print_material.name} can stop platinum silicone from curing. Post-cure the "
            "print fully, seal it, or use tin-cure silicone."
        )
    if material.food_contact:
        warnings.append(
            "Printed layer lines are hard to clean and can harbour bacteria. For food, print "
            "this mold as a master and cast a food-grade silicone mold from it, or seal the "
            "cavity with a food-safe coating."
        )
    return warnings
