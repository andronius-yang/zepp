"""Frozen tuning constants. These were selected on the A100 / Slingshot testbed and are not
user knobs; each carries the reason for its value."""

# --- planning ---
PLAN_OVERLAP_MAX_MIB = 16      # combine-metadata derive overlaps the dispatch GEMM up to this per-rank budget;
                               # above it the derive runs inline (under the heavy wire of 32+ MiB the side-stream
                               # derive can stall)
ROUTE_GRAPH = False            # the route kernels run eagerly (they run slower as a CUDA graph)

# --- expert swap ---
SWAP_MAX_MOVES = 8             # staging slots per rank for one swap phase
SWAP_PAIR_MOVES = 1            # exchanges per (heaviest, lightest) pair per orbit round
SWAP_STREAMS = 4               # movement streams the composed exchange is spread over

# --- direct strategy ---
DIRECT_COMM_SMS = 8            # SMs given to the all-to-all wire kernels

# --- capacity cushions (rows) ---
CAPACITY_CUSHION_PER_RANK = 8  # added per rank on top of the provable bounds (kernel drift headroom)
RELAY_SLOTS = 2                # per-round relay staging is a two-slot double buffer (equals tuning::kRelaySlots,
                               # src/core/tuning.h; the dispatch op checks it)

# --- serving layer graphs ---
GRAPH_MAX_BUCKET = 1024        # largest per-rank token bucket captured as a CUDA graph; larger ones run eagerly
                               # (capture time and memory; decode needs the small ones)
HOST_AHEAD_LAYERS = 1          # layer graphs the host may run ahead of the GPU (a deep launch queue slows both the
                               # attention kernels and the layer graphs)

# --- symmetric heap (GiB) ---
HEAP_MIN_GIB = 6
HEAP_MAX_GIB = 16
HEAP_ROW_MULTIPLIER = {False: 24, True: 42}   # keyed by `swap`

# --- CUDA device connections (host->device work queues) per strategy ---
CUDA_DEVICE_MAX_CONNECTIONS = {"overlap": 24, "direct": 8}
# overlap: 24. More connections can stall the combine wire at large node counts and budgets; fewer serialize
# the overlapped streams.
