# Hint-removal sweep: current results

Snapshot: 2026-10-06 08:41 EDT

**All hints:** success percentage / mean NFE (model calls per sample).

**Removed hints:** one number, **valid + unique %** = distinct valid boards / generated samples × 100. Each valid board counts once per evaluation prompt; repetitions add nothing. Invalid boards never count.

64 puzzles × 32 generations each, with 5,000-step EMA checkpoints. The GPU inference batch size is 64; the diversity group size is 32 generations per prompt.

| Hints retained | Clues retained | Diversity group | Maximum distinct boards counted |
|---|---:|---|---|
| All | 30 | Success rate only | Not a diversity metric |
| 75% | 23 | 32 samples per prompt | Up to 32/32 (100%) per prompt |
| 50% | 15 | 32 samples per prompt | Up to 32/32 (100%) per prompt |
| 25% | 8 | 32 samples per prompt | Up to 32/32 (100%) per prompt |
| None | 0 | All 2,048 samples share the same empty prompt | Up to 2,048/2,048 (100%), pooled |

These are sampling-budget ceilings, not enumerated Sudoku solution counts. With clues remaining, the attainable maximum for a prompt is min(32, the number of valid completions); a prompt with one possible solution can score at most 1/32 = 3.125%. Reduced-hint percentages average equally across the 64 groups. At zero hints, duplicates are removed across all 2,048 samples, so one repeated valid board scores 0.049%.

`hidden` = normal hidden self-conditioning; `token` = new intermediate token feedback; `none` = no self-conditioning. Regular 32/180 denotes transition steps, not exact NFE. Continuous-only EB is experimental hard commitment. Pending cells have no completed result.

## masked-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 76.6% / 5.00 | 3.0% | 3.1% | 3.1% | 0.049% |
| eb_frozen_0.1 | 70.3% / 4.44 | 3.0% | 3.0% | 3.1% | 0.049% |
| regular_180 | 69.5% / 181.00 | 60.8% | 87.6% | 84.0% | 78.5% |
| regular_32 | 36.3% / 33.00 | 38.8% | 25.6% | 15.5% | 8.8% |

## masked-none

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 82.8% / 10.20 | 2.9% | 2.9% | 3.0% | 0.049% |
| eb_frozen_0.1 | 78.1% / 5.52 | 2.9% | 2.9% | 2.9% | 0.049% |
| regular_180 | 14.0% / 181.00 | 9.0% | 6.4% | 5.1% | 3.8% |
| regular_32 | 10.6% / 33.00 | 5.3% | 3.0% | 1.5% | 0.5% |

## masked-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 96.9% / 7.34 | 3.1% | 3.1% | 3.1% | 0.049% |
| eb_frozen_0.1 | 82.8% / 3.73 | 2.7% | 2.8% | 2.9% | 0.049% |
| regular_180 | 37.7% / 181.00 | 31.8% | 26.3% | 22.2% | 14.8% |
| regular_32 | 32.5% / 33.00 | 20.3% | 11.1% | 6.5% | 2.7% |

## uniform-ce-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 96.0% / 4.66 | 59.2% | 88.0% | 90.5% | 2.0% |
| eb_frozen_0.1 | 93.4% / 3.37 | 56.3% | 69.5% | 56.1% | 1.0% |
| eb_revisable_0.01 | 95.7% / 52.70 | 69.0% | 99.6% | 99.9% | 100.0% |
| eb_revisable_0.1 | 96.4% / 3.52 | 68.7% | 99.5% | 99.7% | 99.7% |
| regular_180 | 96.8% / 181.00 | 69.7% | 99.8% | 99.7% | 99.9% |
| regular_32 | 96.9% / 33.00 | 83.0% | 99.2% | 99.7% | 99.8% |

## uniform-ce-none

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 65.2% / 29.72 | 41.4% | 3.3% | 0.098% | 0.0% |
| eb_frozen_0.1 | 61.6% / 19.10 | 36.3% | 2.4% | 0.0% | 0.0% |
| eb_revisable_0.01 | 54.7% / 115.77 | 73.0% | 81.6% | 84.8% | 86.0% |
| eb_revisable_0.1 | 69.6% / 71.98 | 80.6% | 84.8% | 85.7% | 86.9% |
| regular_180 | 50.0% / 181.00 | 68.9% | 80.4% | 83.4% | 84.3% |
| regular_32 | 50.0% / 33.00 | 67.9% | 73.6% | 73.1% | 72.9% |

## uniform-ce-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 86.1% / 12.80 | 63.3% | 13.0% | 1.0% | 0.0% |
| eb_frozen_0.1 | 78.6% / 7.41 | 54.6% | 7.1% | 0.098% | 0.0% |
| eb_revisable_0.01 | 82.6% / 45.20 | 94.1% | 99.3% | 99.6% | 99.6% |
| eb_revisable_0.1 | 88.9% / 26.22 | 94.7% | 98.8% | 98.2% | 98.6% |
| regular_180 | 76.4% / 181.00 | 93.3% | 98.9% | 99.7% | 100.0% |
| regular_32 | 76.6% / 33.00 | 93.2% | 98.4% | 99.4% | 99.3% |

## uniform-plugin-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 85.9% / 42.85 | 74.5% | 87.8% | 11.4% | 0.0% |
| eb_frozen_0.1 | 80.5% / 9.97 | 67.2% | 43.2% | 0.7% | 0.0% |
| eb_revisable_0.01 | 91.3% / 180.00 | 77.9% | 99.2% | 99.2% | 99.6% |
| eb_revisable_0.1 | 90.9% / 106.42 | 78.9% | 99.3% | 99.3% | 99.1% |
| regular_180 | 92.3% / 181.00 | 78.5% | 100.0% | 100.0% | 99.8% |
| regular_32 | 90.9% / 33.00 | 88.1% | 99.9% | 99.9% | 99.5% |

## uniform-plugin-none

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 40.2% / 60.08 | 48.9% | 9.7% | 0.098% | 0.0% |
| eb_frozen_0.1 | 41.9% / 24.99 | 12.3% | 0.098% | 0.0% | 0.0% |
| eb_revisable_0.01 | 41.4% / 180.00 | 64.9% | 77.1% | 80.9% | 84.5% |
| eb_revisable_0.1 | 44.1% / 148.64 | 66.7% | 81.5% | 85.7% | 87.6% |
| regular_180 | 39.5% / 181.00 | 65.0% | 75.7% | 79.3% | 81.8% |
| regular_32 | 40.0% / 33.00 | 61.2% | 70.7% | 72.4% | 68.6% |

## uniform-plugin-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 73.2% / 36.86 | 67.9% | 1.5% | 0.0% | 0.0% |
| eb_frozen_0.1 | 71.1% / 12.96 | 27.1% | 0.4% | 0.0% | 0.0% |
| eb_revisable_0.01 | 74.7% / 151.82 | 91.9% | 98.6% | 99.3% | 99.3% |
| eb_revisable_0.1 | 75.4% / 76.08 | 91.7% | 98.4% | 99.0% | 98.9% |
| regular_180 | 73.9% / 181.00 | 92.1% | 98.2% | 99.2% | 99.0% |
| regular_32 | 75.1% / 33.00 | 91.8% | 97.1% | 98.5% | 95.8% |

## gaussian-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| experimental_eb_frozen_0.01 | 95.6% / 3.86 | 15.7% | 7.9% | 1.7% | 0.0% |
| experimental_eb_frozen_0.1 | 91.4% / 3.32 | 12.5% | 5.3% | 1.0% | 0.0% |
| regular_180 | 0.0% / 181.00 | 7.7% | 4.6% | 1.2% | 0.0% |
| regular_32 | 7.3% / 33.00 | 11.8% | 2.2% | 0.0% | 0.0% |
| regular_4nfe | 0.0% / 4.00 | 0.0% | 0.0% | 0.0% | 0.0% |
| regular_8nfe | 91.4% / 8.00 | 51.2% | 23.8% | 11.8% | 0.4% |

## gaussian-masked-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 90.5% / 4.88 | 51.9% | 97.1% | 99.1% | 99.1% |
| eb_frozen_0.1 | 86.6% / 4.22 | 50.5% | 96.8% | 98.0% | 97.4% |
| regular_180 | 78.2% / 181.00 | 73.3% | 82.6% | 74.0% | 64.5% |
| regular_32 | 44.9% / 33.00 | 40.6% | 24.9% | 14.8% | 7.2% |

## gaussian-masked-none

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 81.6% / 8.60 | 55.7% | 92.3% | 99.0% | 99.1% |
| eb_frozen_0.1 | 71.1% / 4.53 | 56.2% | 89.0% | 94.9% | 94.2% |
| regular_180 | 27.9% / 181.00 | 21.3% | 17.9% | 13.3% | 8.6% |
| regular_32 | 23.6% / 33.00 | 14.6% | 7.0% | 4.3% | 1.3% |

## gaussian-masked-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 93.5% / 8.09 | 42.5% | 81.5% | 96.6% | 99.7% |
| eb_frozen_0.1 | 85.7% / 4.23 | 44.3% | 82.7% | 97.1% | 99.1% |
| regular_180 | 30.8% / 181.00 | 27.0% | 23.9% | 20.9% | 16.5% |
| regular_32 | 24.7% / 33.00 | 15.5% | 9.6% | 5.9% | 3.1% |

## gaussian-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| experimental_eb_frozen_0.01 | 27.0% / 11.93 | 16.5% | 3.7% | 0.049% | 0.0% |
| experimental_eb_frozen_0.1 | 51.3% / 4.54 | 16.6% | 3.6% | 0.0% | 0.0% |
| regular_180 | 61.3% / 181.00 | 42.9% | 9.1% | 0.9% | 0.0% |
| regular_32 | 65.2% / 33.00 | 22.5% | 0.3% | 0.0% | 0.0% |
| regular_4nfe | 40.0% / 4.00 | 0.6% | 0.0% | 0.0% | 0.0% |
| regular_8nfe | 69.3% / 8.00 | 16.9% | 0.2% | 0.0% | 0.0% |

## gaussian-uniform-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 79.3% / 14.76 | 46.9% | 47.7% | 27.8% | 0.0% |
| eb_frozen_0.1 | 80.5% / 7.01 | 40.2% | 36.4% | 18.8% | 0.0% |
| eb_revisable_0.01 | 79.7% / 98.42 | 70.1% | 97.0% | 78.7% | 0.0% |
| eb_revisable_0.1 | 80.8% / 32.10 | 70.2% | 97.4% | 75.6% | 0.0% |
| regular_180 | 93.7% / 181.00 | 82.4% | 100.0% | 100.0% | 89.2% |
| regular_32 | 92.7% / 33.00 | 89.1% | 99.9% | 99.8% | 88.1% |

## gaussian-uniform-none

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 54.5% / 33.00 | 30.3% | 4.2% | 0.098% | 0.0% |
| eb_frozen_0.1 | 52.5% / 20.13 | 28.1% | 3.3% | 0.098% | 0.0% |
| eb_revisable_0.01 | 20.8% / 178.79 | 32.3% | 42.9% | 45.2% | 47.9% |
| eb_revisable_0.1 | 52.4% / 102.23 | 43.8% | 49.2% | 50.9% | 55.5% |
| regular_180 | 35.8% / 181.00 | 50.8% | 61.8% | 59.1% | 49.9% |
| regular_32 | 34.4% / 33.00 | 48.5% | 49.1% | 42.8% | 30.8% |

## gaussian-uniform-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 71.3% / 23.01 | 57.8% | 8.1% | 0.7% | 0.0% |
| eb_frozen_0.1 | 70.1% / 13.19 | 51.6% | 6.7% | 0.098% | 0.0% |
| eb_revisable_0.01 | 58.9% / 103.32 | 81.2% | 91.9% | 95.5% | 97.4% |
| eb_revisable_0.1 | 74.0% / 58.54 | 85.8% | 94.1% | 96.3% | 97.2% |
| regular_180 | 57.6% / 181.00 | 79.1% | 92.4% | 94.8% | 96.9% |
| regular_32 | 57.0% / 33.00 | 76.8% | 88.4% | 90.3% | 92.0% |

## simplex-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| experimental_eb_frozen_0.01 | 94.6% / 4.87 | 36.7% | 60.3% | 79.6% | 0.0% |
| experimental_eb_frozen_0.1 | 91.5% / 4.09 | 34.5% | 56.5% | 75.0% | 0.0% |
| regular_180 | 96.3% / 181.00 | 53.3% | 60.8% | 0.5% | 0.0% |
| regular_32 | 94.4% / 33.00 | 75.0% | 100.0% | 48.0% | 0.0% |
| regular_4nfe | 66.2% / 4.00 | 40.1% | 13.9% | 0.0% | 0.0% |
| regular_8nfe | 83.8% / 8.00 | 80.7% | 87.9% | 23.1% | 0.0% |

## simplex-masked-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 95.3% / 3.58 | 53.9% | 95.3% | 95.2% | 0.0% |
| eb_frozen_0.1 | 85.8% / 3.19 | 55.5% | 91.6% | 88.0% | 0.0% |
| regular_180 | 90.3% / 181.00 | 71.3% | 85.4% | 85.9% | 0.0% |
| regular_32 | 58.3% / 33.00 | 42.6% | 35.3% | 28.8% | 0.0% |

## simplex-masked-none

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 69.3% / 5.92 | 87.7% | 95.2% | 95.8% | 96.6% |
| eb_frozen_0.1 | 60.9% / 3.91 | 81.3% | 87.3% | 89.2% | 88.1% |
| regular_180 | 35.5% / 181.00 | 51.9% | 66.5% | 69.8% | 69.6% |
| regular_32 | 31.5% / 33.00 | 30.5% | 24.8% | 23.7% | 14.8% |

## simplex-masked-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 90.3% / 4.49 | 91.0% | 99.0% | 98.8% | 98.3% |
| eb_frozen_0.1 | 85.0% / 3.05 | 89.5% | 97.7% | 96.2% | 94.7% |
| regular_180 | 59.0% / 181.00 | 69.6% | 74.7% | 71.7% | 66.4% |
| regular_32 | 57.3% / 33.00 | 44.4% | 34.7% | 26.3% | 17.0% |

## simplex-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| experimental_eb_frozen_0.01 | 73.3% / 19.66 | 45.2% | 12.4% | 0.8% | 0.0% |
| experimental_eb_frozen_0.1 | 67.1% / 13.29 | 33.1% | 7.1% | 0.2% | 0.0% |
| regular_180 | 83.3% / 181.00 | 96.6% | 100.0% | 100.0% | 99.8% |
| regular_32 | 70.8% / 33.00 | 89.1% | 94.7% | 94.3% | 93.4% |
| regular_4nfe | 53.9% / 4.00 | 32.9% | 14.0% | 4.2% | 0.9% |
| regular_8nfe | 58.4% / 8.00 | 64.5% | 59.7% | 46.5% | 32.4% |

## simplex-uniform-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 91.3% / 3.90 | 67.2% | 81.1% | 83.5% | 67.7% |
| eb_frozen_0.1 | 84.4% / 3.44 | 69.6% | 80.1% | 83.7% | 64.5% |
| eb_revisable_0.01 | 91.9% / 6.78 | 84.2% | 100.0% | 100.0% | 100.0% |
| eb_revisable_0.1 | 91.6% / 5.50 | 84.7% | 99.9% | 99.8% | 99.7% |
| regular_180 | 93.7% / 181.00 | 81.4% | 100.0% | 100.0% | 100.0% |
| regular_32 | 93.1% / 33.00 | 84.9% | 100.0% | 100.0% | 100.0% |

## simplex-uniform-none

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 60.7% / 19.47 | 58.8% | 32.7% | 11.2% | 0.7% |
| eb_frozen_0.1 | 53.6% / 13.04 | 47.9% | 25.2% | 8.4% | 0.3% |
| eb_revisable_0.01 | 66.1% / 92.72 | 85.6% | 91.9% | 89.1% | 77.6% |
| eb_revisable_0.1 | 69.5% / 59.22 | 82.6% | 86.2% | 80.4% | 70.6% |
| regular_180 | 58.7% / 181.00 | 84.1% | 93.4% | 91.3% | 80.1% |
| regular_32 | 46.9% / 33.00 | 67.1% | 74.4% | 65.5% | 42.5% |

## simplex-uniform-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 79.7% / 11.25 | 74.7% | 57.8% | 31.4% | 2.2% |
| eb_frozen_0.1 | 75.3% / 6.44 | 66.2% | 48.4% | 25.2% | 2.1% |
| eb_revisable_0.01 | 80.8% / 46.39 | 93.2% | 98.8% | 98.9% | 98.3% |
| eb_revisable_0.1 | 84.7% / 28.65 | 90.3% | 95.2% | 94.3% | 92.5% |
| regular_180 | 75.4% / 181.00 | 94.7% | 99.4% | 99.3% | 98.4% |
| regular_32 | 70.3% / 33.00 | 85.5% | 89.5% | 86.6% | 78.8% |

## spherical-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| experimental_eb_frozen_0.01 | 98.6% / 3.43 | 65.9% | 85.8% | 46.7% | 0.0% |
| experimental_eb_frozen_0.1 | 95.9% / 3.00 | 66.2% | 87.4% | 55.3% | 0.0% |
| regular_180 | 98.2% / 181.00 | 50.8% | 58.0% | 59.5% | 85.0% |
| regular_32 | 98.3% / 33.00 | 46.9% | 56.2% | 49.7% | 29.8% |
| regular_4nfe | 96.7% / 4.00 | 18.6% | 0.6% | 0.0% | 0.0% |
| regular_8nfe | 98.2% / 8.00 | 50.1% | 49.0% | 7.7% | 0.0% |

## spherical-masked-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 91.4% / 4.30 | 62.8% | 95.6% | 91.9% | 92.3% |
| eb_frozen_0.1 | 84.5% / 4.02 | 57.8% | 89.2% | 85.5% | 88.2% |
| regular_180 | 82.1% / 181.00 | 74.5% | 87.1% | 81.9% | 75.3% |
| regular_32 | 46.3% / 33.00 | 38.6% | 29.6% | 19.6% | 11.5% |

## spherical-masked-none

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 16.3% / 13.02 | 6.2% | 9.2% | 16.0% | 5.2% |
| eb_frozen_0.1 | 20.9% / 6.30 | 6.4% | 9.6% | 15.0% | 4.5% |
| regular_180 | 24.0% / 181.00 | 23.6% | 16.7% | 14.6% | 10.1% |
| regular_32 | 21.8% / 33.00 | 12.0% | 6.3% | 3.5% | 1.5% |

## spherical-masked-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 87.5% / 6.74 | 43.0% | 77.2% | 97.0% | 99.9% |
| eb_frozen_0.1 | 83.3% / 3.79 | 44.4% | 77.6% | 95.6% | 98.4% |
| regular_180 | 37.5% / 181.00 | 34.0% | 28.7% | 26.0% | 20.6% |
| regular_32 | 32.4% / 33.00 | 18.5% | 10.7% | 7.0% | 4.1% |

## spherical-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| experimental_eb_frozen_0.01 | 13.0% / 15.40 | 1.0% | 0.1% | 0.0% | 0.1% |
| experimental_eb_frozen_0.1 | 29.8% / 6.41 | 1.0% | 0.2% | 0.098% | 0.049% |
| regular_180 | 60.0% / 181.00 | 55.6% | 24.9% | 8.5% | 1.0% |
| regular_32 | 58.9% / 33.00 | 43.2% | 10.0% | 2.0% | 0.049% |
| regular_4nfe | 57.5% / 4.00 | 15.8% | 0.7% | 0.0% | 0.0% |
| regular_8nfe | 57.6% / 8.00 | 26.5% | 2.9% | 0.049% | 0.049% |

## spherical-uniform-hidden

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 92.2% / 7.64 | 78.2% | 76.0% | 67.5% | 74.0% |
| eb_frozen_0.1 | 90.5% / 4.94 | 75.6% | 69.2% | 63.2% | 72.3% |
| eb_revisable_0.01 | 94.2% / 69.09 | 89.9% | 99.7% | 100.0% | 100.0% |
| eb_revisable_0.1 | 93.1% / 7.59 | 88.6% | 99.4% | 99.7% | 99.8% |
| regular_180 | 93.9% / 181.00 | 90.3% | 100.0% | 100.0% | 100.0% |
| regular_32 | 93.2% / 33.00 | 92.6% | 99.8% | 100.0% | 100.0% |

## spherical-uniform-none

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 66.9% / 18.07 | 82.6% | 78.8% | 64.9% | 47.4% |
| eb_frozen_0.1 | 81.5% / 7.44 | 83.6% | 77.1% | 63.9% | 45.4% |
| eb_revisable_0.01 | 49.2% / 127.12 | 72.1% | 84.3% | 87.5% | 89.4% |
| eb_revisable_0.1 | 83.6% / 33.15 | 83.3% | 89.3% | 89.0% | 87.5% |
| regular_180 | 50.9% / 181.00 | 66.9% | 78.7% | 80.7% | 78.4% |
| regular_32 | 48.1% / 33.00 | 64.4% | 69.9% | 66.7% | 51.5% |

## spherical-uniform-token

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 80.5% / 15.21 | 85.9% | 78.2% | 65.6% | 47.8% |
| eb_frozen_0.1 | 81.6% / 6.59 | 85.3% | 74.6% | 60.2% | 45.8% |
| eb_revisable_0.01 | 72.8% / 70.04 | 82.0% | 91.7% | 94.0% | 95.8% |
| eb_revisable_0.1 | 84.9% / 29.66 | 86.8% | 93.8% | 95.5% | 96.0% |
| regular_180 | 64.6% / 181.00 | 84.0% | 93.6% | 96.1% | 94.7% |
| regular_32 | 63.9% / 33.00 | 83.5% | 90.1% | 88.7% | 81.2% |

## masked-token-detached-depth0

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 93.8% / 6.33 | 3.1% | 3.1% | 3.1% | 0.049% |
| eb_frozen_0.1 | 79.7% / 3.78 | 2.7% | 3.0% | 2.8% | 0.049% |
| regular_180 | 35.2% / 181.00 | 30.9% | 27.1% | 22.9% | 17.0% |
| regular_32 | 31.2% / 33.00 | 19.0% | 9.9% | 6.0% | 2.7% |

## masked-token-detached-depth3

| Sampler | All hints: success / NFE | 75%: valid + unique | 50%: valid + unique | 25%: valid + unique | None: valid + unique |
|---|---:|---:|---:|---:|---:|
| eb_frozen_0.01 | 89.1% / 7.17 | 3.0% | 3.1% | 3.1% | 0.049% |
| eb_frozen_0.1 | 73.4% / 3.77 | 2.9% | 3.0% | 3.0% | 0.049% |
| regular_180 | 36.3% / 181.00 | 29.2% | 21.9% | 17.6% | 12.5% |
| regular_32 | 32.2% / 33.00 | 19.3% | 8.9% | 5.1% | 1.9% |
