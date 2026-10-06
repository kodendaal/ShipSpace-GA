# Data provenance

This document records the external geometric and general-arrangement sources used to support generator development and evaluation. The published synthetic corpus is archived on Zenodo with this release.

## Production hull templates

The production hull library used by the paper contains **17 processed templates**:

| Pool | Templates | Used by generator families |
|---|---:|---|
| Transport | 5 | Bulker, Tanker, Container |
| OSV / work vessel | 6 | OSV |
| Yacht | 3 | Yacht |
| Patrol / defence | 3 | Patrol |
| **Total** | **17** | — |

Sixteen processed templates are derived from the CGTrader **Hulls Collection** and one is based on the public **KRISO Container Ship (KCS)** benchmark hull. Before generator use, geometries were converted to STL where required, aligned to the common coordinate convention, symmetrised where necessary, and independently scaled to the sampled `L`, `B` and `D`.

Source pages:

- CGTrader, *Hulls Collection*: https://www.cgtrader.com/3d-models/watercraft/industrial-watercraft/hulls-collection-2010a682-0584-4725-9da9-e8740bdf8883
- SIMMAN 2008 / KRISO Container Ship (KCS): http://www.simman2008.dk/KCS/kcs_link.htm

The original third-party source geometries are **not redistributed in this GitHub repository**. This repository ships the production voxel hull masks under `data/hull_cache/` (17 files, `64 × 32 × 24`). Any additional processed hull assets supplied separately should be treated according to the terms stated with that archive; the presence of code in this repository does not grant rights to third-party source geometry.

### Production filename pools

The production hull library identifies compatible processed templates by filename prefix. The same prefixes are used for the shipped voxel masks in `data/hull_cache/`:

| Generator pool | Filename prefix |
|---|---|
| Transport | `transport_*` |
| OSV / work vessel | `work_*` |
| Patrol | `defense_*` |
| Yacht | `yacht_*` |

## Reconstructed real-reference corpus

The real-reference corpus contains **23 manually reconstructed general arrangements**: six yachts, five OSVs, and three each of bulk carriers, tankers, container vessels, and patrol vessels. The source drawings were mapped to the common functional taxonomy used by the generator.

These should be interpreted as **spatial reconstructions**, not exact CAD reproductions. They were used as plausibility/reference cases and also informed selected family-specific generator ranges and spatial preferences.

The graphs and metadata shipped with this repository are in `data/real_ga/` (`real_ga_shard_000.pt`, `real_ga_meta.json`). Shipped sample names (`yacht_1`, …, `patrol_3`) correspond to the provenance IDs below.

| ID | Shipped name | Family | Vessel / source | Public source |
|---|---|---|---|---|
| Y01 | `yacht_1` | Yacht | Burgess, *Boardwalk* (Feadship) | https://www.burgessyachts.com/en/buy-a-yacht/yachts-for-sale/boardwalk-v-00008893 |
| Y02 | `yacht_2` | Yacht | YachtCharterFleet, *Deep Blue* (Amels) | https://www.yachtcharterfleet.com/luxury-charter-yacht-48099/deep-blue-layout.htm |
| Y03 | `yacht_3` | Yacht | YachtCharterFleet, *O'Ptasia* (Golden Yachts) | https://www.yachtcharterfleet.com/luxury-charter-yacht-49216/o-ptasia-layout.htm |
| Y04 | `yacht_4` | Yacht | YachtCharterFleet, *Quattroelle* (Lürssen) | https://www.yachtcharterfleet.com/luxury-charter-yacht-27048/quattroelle-layout.htm |
| Y05 | `yacht_5` | Yacht | YachtCharterFleet, *Seven Seas* (Oceanco) | https://www.yachtcharterfleet.com/luxury-charter-yacht-53600/seven-seas-layout.htm |
| Y06 | `yacht_6` | Yacht | YachtCharterFleet, *Sophia* (Feadship) | https://www.yachtcharterfleet.com/luxury-charter-yacht-49014/sophia-layout.htm |
| B01 | `bulker_1` | Bulker | T.K. Shipbuilding, *Nordic London* | https://onebulk.net/wp-content/uploads/2019/09/ZZ-Vessel-Presentation-NORDIC-LONDON.pdf |
| B02 | `bulker_2` | Bulker | Nanjing Dongze Shipyard, *Nanjing Confidence* | https://www.orient.nl/wp-content/uploads/2021/11/General-Arrangement-Nanjing-Confidence.pdf |
| B03 | `bulker_3` | Bulker | Oshima Shipbuilding, 37,200/46,700 DWT open bulk carrier; drawing 10040000 | https://oelsm.com/wp-content/uploads/2025/06/P-04-General-Arrangement-LUNAR-STAR-1.pdf |
| T01 | `tanker_1` | Tanker | Besiktas Shipyard, 7,000 DWT product tanker; drawing RST22M110.1 | https://springmarines.com/uploads/media/Document/2022/07/general-arrangement.pdf |
| T02 | `tanker_2` | Tanker | Spring Marine, *Zumrut Ana* | https://springmarines.com/en/fleet-detail/zumrut-ana |
| T03 | `tanker_3` | Tanker | Neda Maritime, *Seriana* / *Stellata* | https://www.nedamaritime.gr/img/media_room/brochures/NedaAframaxNewBuildings.pdf |
| O01 | `osv_1` | OSV | CAMGSA, *Centurión* | https://camgsa.mx/assets/files/BA_Centurion_Brochure.pdf |
| O02 | `osv_2` | OSV | MMA Offshore, *MMA Leeuwin* | https://www.mmaoffshore.com/theme/mmaoffshorecomau/assets/public/File/vessel-specs/MMA_Leeuwin.pdf |
| O03 | `osv_3` | OSV | MMA Offshore, *MMA Pride*; Cybermarine Technologies drawing CT-18-5524-0002 | https://www.mmaoffshore.com/theme/mmaoffshorecomau/assets/public/File/vessel-specs/MMA_PRIDE.pdf |
| O04 | `osv_4` | OSV | MMA Offshore, *MMA Pinnacle* | https://www.neptunems.com/vessel-fleet/mma-pinnacle |
| O05 | `osv_5` | OSV | VARD, *Skandi Responder*; drawing 808-101-001 | https://www.vard.com/shipbuilding/references/skandi-responder |
| C01 | `container_1` | Container | Daewoo Shipbuilding & Marine Engineering, NRS 4,600 TEU containership; drawing 101Z029 | https://www.naviecapitani.it/Navi%20e%20Capitani/gallerie%20navi/Containers/foto/N/Northern%20Power/Piano%20Generale.pdf |
| C02 | `container_2` | Container | F. A. Vinnen & Co., *Leonidio* | https://vinnen.com/fleet/ |
| C03 | `container_3` | Container | Stocznia Gdynia, Type 8234 containership; drawing 8234/1-DZ/0110-001 | https://www.naviecapitani.it/gallerie%20navi/containers/foto/M/MSC/MSC%20Florida/Piano.pdf |
| P01 | `patrol_1` | Patrol | Austal, *Offshore Patrol 60* | https://www.austal.com/sites/default/files/2025-12/Austal%20Offshore%20Patrol%2060%20-%20Maritime%20Security%20variant%20datasheet.pdf |
| P02 | `patrol_2` | Patrol | Austal, *Patrol 40* (Guardian class) | https://www.austal.com/sites/default/files/2025-12/Austal%20Patrol%2040%20-%20Guardian-class%20datasheet.pdf |
| P03 | `patrol_3` | Patrol | Austal, *Patrol 58* (Cape class) | https://www.austal.com/sites/default/files/2025-12/Austal%20Patrol%2058%20Cape-Class%20%28Original%29%20datasheet.pdf |

## Use in the paper

- **Experiment 1:** all 23 reconstructed references.
- **Experiment 2:** one reconstructed reference per vessel family.
- **Experiment 3:** one fixed real-reference exterior envelope per family.
- **Fixed-hull OSV application:** the same OSV reference/hull envelope used in the OSV reconstruction/controllability studies.

The large generated corpora are archived on Zenodo with this release.
