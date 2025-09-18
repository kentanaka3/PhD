## Comparison (ogscompare.py)

Compare two catalogs of seismic information:

Inputs:
- Base catalog (your “reference,” e.g., OGS data exported as .dat/.hpl/.txt/.pun).
- Target catalog (your pipeline’s outputs, e.g., SeisBench or Gamma producer outputs in Parquet).

It aligns them by day and:
- For picks: compares per-station P/S classifications.
- For events: compares detected events and their attributes (time, location, depth, magnitude). It computes confusion matrices, recall metrics, and produces diagnostic plots (maps, histograms, scatter plots) saved under catalogs/buildCatalog/img.

### High-level architecture
CLI parsing: collect paths and date range.
Base and Target catalogs: load heterogeneous inputs into a common structure.
Comparison engine: apply model equivalence map to pair Base vs Target products, align by day, compute confusion matrices and summary plots.
Key components
Catalog base class: just a harness for args and a RESULTS_PATH list of (module_name, DataSet instance) pairs.
Base(Catalog): expects filesystem trees like .dat, .hpl, .txt, .pun containing CSVs under events/*/*/*.csv and assignments/*/*/*.csv.
Target(Catalog): expects Parquet structures (e.g., assignments/*, events/*) and specialized product folders: SeisBenchPicker, GammaAssociator, PyOctoAssociator, NonLinLoc, OGSMagnitude, or plain CSV files events.csv, assignments.csv.
Comparison: orchestrates loading, station inventory visualization, pairing Base/Target modules, day-wise alignment, confusion matrix computation, and figure exports.
The CLI and inputs
Required:
    -B/--base: base directory for the OGS-like catalog.
    -T/--target: target directory for the machine-produced catalog.
Optional:
    -D/--dates YYMMDD YYMMDD: start and end date (defaults to 240320–240620), and are sorted automatically.
    -S/--station: path to station XMLs (used by inventory() to plot station locations).
    -W/--waveform: waveform directory (not used in comparisons here, but may be for future integrations).
    -v/--verbose: adds verbosity.
Dates are parsed via is_date, which uses the format from OGS_C.YYMMDD_FMT.

Data model: “groups” and daily alignment
All loaders normalize their data into a common shape in memory:

For Base catalog:

base.groups["events"] is a dict keyed by date -> DataFrame.
base.groups["assignments"] is a dict keyed by date -> DataFrame.
For Target catalog:

Target loaders generally produce a single DataFrame per logical group:
target.groups["events"] is a DataFrame with a groups (date) column.
target.groups["assignments"] is a DataFrame with a groups (date) column.
The generic Target.DataSet also writes normalized CSVs back to disk under catalogs/buildCatalog/<ClassName>/<dep>/<YYYY>/<M>/<D>.csv, where dep is assignments or events.
The daily alignment logic:

Target: groupby groups (date column derived from timestamp).
Base: lookup that same date key in base.groups[key].
This daily bucketing ensures comparisons are temporal and fair.

Filesystem expectations with examples
Here’s what the Base and Target folders may look like.

Base folder (OGS-like)
Root contains subfolders named after extensions:
        .dat/assignments/2024/03/20.csv
        .dat/events/2024/03/20.csv
        .hpl/..., .txt/..., .pun/...
The loader walks up and down to detect these extension-labeled directories.
Example: -B /data/OGS

        /data/OGS/.dat/assignments/2024/06/20.csv
        /data/OGS/.dat/events/2024/06/20.csv
        /data/OGS/.hpl/assignments/2024/06/20.csv
        /data/OGS/.hpl/events/2024/06/20.csv
Notes:

For Base events, if ML is missing, it’s added and filled with NaN.
A regional-inclusion check is computed via matplotlib.path.Path.contains_point, but in Base it is not applied as a filter (see “Caveats and tips”).
Target folder (pipeline outputs)
Generic Parquet containers:
assignments/*.parquet (with columns renamed to match constants)
events/*.parquet
Specialized products:
SeisBenchPicker/ uses picks/*.parquet for assignments.
GammaAssociator/, PyOctoAssociator/, NonLinLoc/, OGSMagnitude/ typically have assignments/*.parquet and/or events/*.parquet.
Example: -T /data/Target

/data/Target/SeisBenchPicker/picks/part-000.parquet
/data/Target/GammaAssociator/events/part-001.parquet
/data/Target/NonLinLoc/assignments/part-000.parquet
Module detection and the mapping that drives comparisons
The MDL_EQUIV mapping indicates whether a given module contains:

Index 0: pick-level data (assignments).
Index 1: event-level data (events).
A “2” after adding Base and Target’s vectors means “compare this level”.

Examples:

        Base .dat → [1, 0] (picks only)
        Target SeisBenchPicker → [1, 0] (picks only)
        Sum: [2, 0] → compare picks
        Base .hpl → [1, 1], Target GammaAssociator → [1, 1]
        Sum: [2, 2] → compare both picks and events

The comparison loop: how it works
For each compatible Base/Target module pair:

Decide whether to compare “assignments” (picks) or “events.”

For Target, group by groups (the daily date).

For each day:

Ensure Base has data for that date.



## Bipartite Graph Matching Algorithm
The matching brain behind catalog comparisons, the distance 
functions, the bipartite graph model, the greedy matching algorithm, and how 
confusion matrices and TP/FN/FP outputs are built—plus diagrams and small 
worked examples you can adapt.

### 1) Purpose and mental model
This module solves “what matches what?” between two time-indexed collections:

TRUE nodes (reference) vs PRED nodes (predictions).
Two domains:
Picks/classification: align phase picks per station and time.
Events/source: align event hypotheses by time and spatial proximity.
It builds a bipartite graph with edges connecting temporally (and spatially) 
feasible pairs. Edges carry a similarity score W(T,P) in [0,1], then a greedy 
maximum-weight matching picks the best partner for each prediction, and the 
induced mapping is used to compute confusion matrices and TP/FN/FP sets.

### 2) Distance functions and configuration
At the top, a family of distances converts time, phase, and space differences 
into a normalized [0,1] similarity.

dist_phase(T,P): 1 if phases match, else 0
diff_time(T,P): absolute time difference as timedelta
dist_time(T,P, offset): 1 − diff_time/offset
offset varies by task:
  - picks: OGS_C.PICK_OFFSET
  - events: OGS_C.ASSOCIATE_TIME_OFFSET
dist_space(T,P, offset): 1 − (surface distance / offset)
uses gps2dist_azimuth; offset is km
Composite metrics:

dist_balanced = (dist_time + 9*dist_phase)/10
dist_default = (99*dist_balanced + probability)/100
probability is from PRED row (e.g., model confidence)
dist_event = average(dist_time with event offset, dist_space with event km)
Config hooks:

OGS_C.MATCH_CNFG[CLSSFD].DISTANCE = dist_default
OGS_C.MATCH_CNFG[DETECT].DISTANCE = dist_default
OGS_C.MATCH_CNFG[SOURCE].DISTANCE = dist_event
This lets the comparison engine switch behavior by method: CLSSFD/DETECT for 
picks, SOURCE for events.

Key intuition:

Picks: phase agreement is decisive; small time separation; and higher model 
probability helps.
Events: both time and space must be close.
### 3) The bipartite graph: nodes, gating, edges
Class: myBPGraph(T, P, config)

Inputs:

T: DataFrame of TRUE entries.
P: DataFrame of PRED entries.
config: method-specific dictionary (time window, spatial offset, category labels, etc.).
Data structures:

T is stored as a dict of columns keyed by index (self.T[col][i] = value).
P is stored similarly (self.P[...]).
We maintain two adjacency maps:

self.G[0]: adjacency from TRUE indices → viable PRED matches with weights.
self.G[1]: adjacency from PRED indices → viable TRUE matches with weights.
Temporal (and optional spatial) gating in cnxNode:

Select candidates v such that |t_T − t_P| ≤ TIME_DSPLCMT.
If method == SOURCE (events): compute distances and drop pairs beyond ASSOCIATE_DIST_OFFSET (km).
For remaining candidates: edge weight = W(T,P) via config[OGS_C.DISTANCE_STR].
Adjacency matrix view:

adjMtx() shapes an N×M matrix (N TRUE, M PRED) of weights; mostly sparse.
Why gating matters:

Instead of O(N×M) scoring, we prune using domain knowledge (time windows and regional cutoff), which keeps the graph manageable.
### 4) Matching: greedy, prediction-centric maximum choice
#### Algorithm: maxWmatch()

##### For each TRUE index t:
 - Consider its neighbor set {(t, p, weight)}.
 - Pick the highest-weight p.
 - Keep only the best competing TRUE for each PRED (if multiple TRUE pick the same PRED, only keep the one with highest weight).
 - Return list of (t, p, w) unique by p.


#### Then makeMatch():

##### Clears adjacencies and installs a 1–1 mapping:
    self.G[0][t] = {p: w}
    self.G[1][p] = {t: w}
This is a greedy algorithm (not Hungarian). It’s fast and simple but not globally optimal if there are intertwined conflicts; for most catalog comparisons with good gating and strong top-1 matches, it performs well.

#### Complexity:
- Building neighbors: dominated by gating and edge creation, roughly O(k) per node where k is local neighborhood size.
- maxWmatch: O(N * deg) with a constant factor for picking maxima.

#### Limitations:

In dense periods (aftershocks, bursts), a global assignment (Hungarian) could yield better overall pairing. Greedy is a pragmatic tradeoff for speed and simplicity.

### 5) From matches to confusion matrices and TP/FN/FP
##### confMtx():
- Initializes CFN_MTX as a square DataFrame whose index/columns are category labels:

      Picks: [PWAVE, SWAVE, NONE]
      Events: [EVENT, NONE]

- Iterates TRUE side (G[0]):

If TRUE t matched to PRED p:
Picks: increment CFN_MTX[phase_true, phase_pred]; if same phase, add to TP set with rich tuple payload.
Events: increment CFN_MTX[EVENT, EVENT]; TP add tuple with paired attributes (lat/lon/depth/ML/ERH/ERZ/time).
Else:
Picks: increment CFN_MTX[phase_true, NONE], append FN (false negative).
Events: increment CFN_MTX[EVENT, NONE], append FN.
Iterates PRED side (G[1]):

If PRED p has no match:
Picks: increment CFN_MTX[NONE, phase_pred], record FP.
Events: increment CFN_MTX[NONE, EVENT], record FP.
Returns:

CFN_MTX: DataFrame confusion matrix.
TP: set of tuples.
FN: list of rows.
FP: set of tuples.
Wrapper conf_mtx(TRUE, PRED, model_name, dataset_name, method):

Builds myBPGraph with MATCH_CNFG[method].
Calls makeMatch() then confMtx().
Annotates TP/FN/FP with model/dataset names in the first columns.
Returns (CFN_MTX, TP, FN, FP).
Contract with ogscompare.py:

Picks TP header: [BASE, TARGET, UNKNOWN, INDEX, TIMESTAMP, PHASE, STATION, ERT]
TP stores INDEX and TIMESTAMP as tuples (TRUE, PRED).
Events TP header: [BASE, TARGET, UNKNOWN, INDEX, TIMESTAMP, LAT, LON, DEPTH, ML, ERH, ERZ, NOTES]
Each quantitative field stores (TRUE, PRED) in the TP set.
### 6) Diagrams
A. Graph construction pipeline
TRUE DataFrame ----> Gate by time window ----
---> Weighted edge W(T,P) PRED DataFrame ----> Gate by time window ----/ if SOURCE: space gate too

Result: Bipartite adjacency maps G[0]: T -> {P: weight} G[1]: P -> {T: weight}

B. Matching flow
For each TRUE t: take neighbours (t, p, w) pick argmax w Resolve conflicts so each PRED is matched at most once: keep max-w TRUE for each PRED

Install mapping: G[0][t] = {p: w} G[1][p] = {t: w}

C. Confusion matrix aggregation
Traverse TRUE nodes: matched? → CFN_MTX[true_label, pred_label]++ build TP rich tuple else → CFN_MTX[true_label, NONE]++ add FN row

Traverse PRED nodes: unmatched? → CFN_MTX[NONE, pred_label]++ add FP tuple

### 7) Worked mini examples
#### 7.1 Picks (classification) example
Assume constants:

TIME_DSPLCMT_STR = 1.0 s
PICK_OFFSET = 1.0 s
TRUE (T):

t0: time=10.0, station=OX.BOO, phase=P, index=100, ERT=0.3
t1: time=20.0, station=OX.BOO, phase=S, index=101, ERT=0.5
PRED (P):

p0: time=10.2, station=OX.BOO, phase=P, index=900, prob=0.9
p1: time=19.6, station=OX.BOO, phase=P, index=901, prob=0.55
Edges within 1s:

t0 ↔ p0 (Δt=0.2s) → high weight (phase match P/P, high prob)
t1 ↔ p1 (Δt=0.4s) → weight penalized (phase S vs P mismatch)
Greedy matching:

t0→p0 kept.
t1→p1 kept (no conflicts), but phase mismatch.
Confusion:

CFN[P,P] += 1 (t0 vs p0)
CFN[S,P] += 1 (t1 vs p1)
No NONE cells. TP set contains only the phase-correct match (t0/p0). FN empty; FP empty.
#### 7.2 Events example
Assume:

    ASSOCIATE_TIME_OFFSET = 5s
    ASSOCIATE_DIST_OFFSET = 10 km
TRUE:

    e0: t=100.0, lat=46.05, lon=13.15, depth=10, ML=2.1
PRED:

    h0: t=102.0, lat=46.06, lon=13.16, depth=9.2, ML=2.0 (within 2s and ~1.3 km)
    h1: t=200.0, far away
Edges:

    e0–h0 passes both gates, weight ~ high
    e0–h1 rejected by time/space gates

Match: e0→h0 Confusion:

CFN[EVENT,EVENT] += 1
TP holds tuples of paired numeric fields No FN/FP.


### 8) Caveats, validations, improvements
#### Time units:
diff_time returns timedelta; dist_time divides by offset (also timedelta). That’s consistent. Ensure offsets in OGS_C are td for time-based distances.
#### Probability and ERT semantics:
dist_default blends model probability into the score; ensure probability is in [0,1]. If different scale, rescale upstream.
##### Station and phase consistency:
For picks, DATAFRAMES must include PHASE_STR and STATION_STR consistent with constants. If station codes differ (e.g., “OX.BOO” vs “BOO”), normalize before calling conf_mtx.
#### Spatial distance is surface (gps2dist_azimuth). If depth-aware matching is desired, extend dist_event to include depth difference.
#### Matching optimality:
Consider Hungarian algorithm for global max-weight matching if dense overlaps cause greedy suboptimal matches. That would require building a full cost matrix on the gated candidates and handling rectangular cases (N≠M).
Error handling:
rmvEdge uses dict.remove which does not exist; if used, it should be pop. It’s not called in current flow, but worth fixing before future use.
### 9) Quick usage snippet
Assuming you have two DataFrames with the expected columns and constants:

#### Picks (method=CLSSFD_STR): columns must include

    TIMESTAMP_STR (UTCDateTime or convertible), PHASE_STR, STATION_STR, INDEX_STR
    PRED must include PROBABILITY_STR
    Optionally ERT_STR in TRUE rows

#### Events (method=SOURCE_STR): columns include

    TIMESTAMP_STR, LATITUDE_STR, LONGITUDE_STR, DEPTH_STR
    Optional ML, ERH, ERZ, NOTES, and identifiers (INDEX_STR for TRUE, ID_STR for PRED)
Example:
```python
    from ogsbpgma import conf_mtx import ogsconstants as OGS_C

    CFN, TP, FN, FP = conf_mtx(TRUE_df, PRED_df, model_name=".hpl", dataset_name="GammaAssociator", method=OGS_C.SOURCE_STR)

    print(CFN) print("TP:", len(TP), "FN:", len(FN), "FP:", len(FP))
```
This is exactly what ogscompare.py expects; it then aggregates across days and renders plots.