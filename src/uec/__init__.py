"""
Universal Entropy Coding (UEC) library for holonomy estimation and Arrow-of-Time analysis.

Key functions now support enhanced flexibility:

Holonomy Functions (all support coder="kt"|"lz78"):
- klrate_between_sequences: Estimates KL divergence D(P||Q) using universal coding
- klrate_holonomy_general: General loop holonomy with alignment options ('tail'|'head'|'auto')
- klrate_holonomy_time_reversal_markov: Canonical time-reversal holonomy

Markov Functions:
- entropy_production_rate_bits: Now supports strict=True to raise on one-way edges
- All sampling functions support rng parameter for reproducibility

AoT Functions:
- load_wav_mono: Enhanced with target_std parameter and float WAV support
- aot_from_series: Extended with coder, block_seconds, and enhanced metadata

All holonomy functions are estimators of ideal KL-holonomy rates defined via
universal coding consistency (see accompanying theory paper).
"""

from .markov import (
    _row_stochastic,
    stationary_distribution,
    sample_markov,
    entropy_production_rate_bits,
    random_markov_biased,
)
from .coders import KTMarkovMixture, KTFrozenPredictor, LZ78Coder
from .transforms import (
    Transform,
    Permute,
    MergeSymbols,
    TimeReverse,
    TransitionEncode,
    TransitionDecodeTakeSecond,
    TransitionEncodeLag,
    Downsample,
    UpsampleRepeat,
    apply_loop,
)
from .holonomy import (
    klrate_between_sequences,
    klrate_holonomy_general,
    klrate_holonomy_time_reversal_markov,
    klrate_holonomy_leadlag_markov,
)
from .aot import (
    load_wav_mono,
    load_csv_column,
    aot_from_series,
    discretize_series,
)
from .bio import (
    iaaft_surrogate,
    circular_shift_surrogate,
    make_surrogates,
    counts_from_sequence as counts_from_seq_bio,
    klrate_time_reversal_from_counts,
    joint_tokens_from_series,
    boolean_mask_by_quantile_bin,
    check_sufficiency,
    support_asymmetry,
    stationarity_delta_bits,
    holonomy_day_night,
    holonomy_leadlag,
    surrogate_pvalue,
    coder_agreement,
)

__all__ = [
    "_row_stochastic",
    "stationary_distribution",
    "sample_markov",
    "entropy_production_rate_bits",
    "random_markov_biased",
    "KTMarkovMixture",
    "KTFrozenPredictor",
    "LZ78Coder",
    "Transform",
    "Permute",
    "MergeSymbols",
    "TimeReverse",
    "TransitionEncode",
    "TransitionDecodeTakeSecond",
    "TransitionEncodeLag",
    "Downsample",
    "UpsampleRepeat",
    "apply_loop",
    "klrate_between_sequences",
    "klrate_holonomy_general",
    "klrate_holonomy_time_reversal_markov",
    "klrate_holonomy_leadlag_markov",
    "load_wav_mono",
    "load_csv_column",
    "aot_from_series",
    "discretize_series",
    "iaaft_surrogate",
    "circular_shift_surrogate",
    "make_surrogates",
    "counts_from_seq_bio",
    "klrate_time_reversal_from_counts",
    "joint_tokens_from_series",
    "boolean_mask_by_quantile_bin",
    "check_sufficiency",
    "support_asymmetry",
    "stationarity_delta_bits",
    "holonomy_day_night",
    "holonomy_leadlag",
    "surrogate_pvalue",
    "coder_agreement",
]
