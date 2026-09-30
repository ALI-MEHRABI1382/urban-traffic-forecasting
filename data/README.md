# Data attribution and provenance

These files are the processed CSV extracts retained from Ali Mehrabi’s 2025 BME coursework, renamed for this repository:

| Repository file | Supplied filename | Description |
| --- | --- | --- |
| `traffic.csv` | `Modified_Traffic_Data2 (1).csv` | Hourly traffic extract; the pipeline selects site 755, detector 1 |
| `noise.csv` | `Noise_Polution_May_December_2022.csv` | May–December 2022 environmental noise extract |

The original report attributes the traffic observations to Dublin City Council’s SCATS system and the noise observations to the Sonitus monitoring network, near Ballymun Road. The provided CSVs are not asserted to be untouched downloads from the catalog. Earlier filtering, aggregation and export steps are not fully documented.

Publisher/source credit: **Dublin City Council**, made available through **Smart Dublin / Dublinked**; environmental noise monitoring by **Sonitus Systems** as identified in the coursework report.

Relevant official catalogs:

- [SCATS traffic volumes, January–June 2022](https://data.smartdublin.ie/dataset/dcc-scats-detector-volume-jan-jun-2022)
- [SCATS traffic volumes, July–December 2022](https://data.smartdublin.ie/dataset/dcc-scats-detector-volume-jul-dec-2022)
- [Ambient Sound Monitoring Network DCC](https://data.smartdublin.ie/dataset/ambient-sound-monitoring-network)
- [Noise and Air Quality Monitoring API DCC](https://data.smartdublin.ie/dataset/sonitus)

The referenced catalog datasets list Creative Commons Attribution licensing. Consult their linked license terms for the authoritative terms. These catalog links identify the source families; the exact historical resource downloads used to produce the supplied extracts have not been independently reconstructed.

Changes in this repository: filenames were normalized; the input CSV contents were retained. During analysis, timestamps are parsed, the detector is selected, conflicting traffic timestamps are excluded, data is reindexed hourly, and short feature gaps are filled from the past. Targets are never filled or smoothed. The input files contain no explicit timezone.
