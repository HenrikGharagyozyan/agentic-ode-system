# Event lab baseline — run 20261016-142636

Offline re-analysis of three saved real recordings (IDS UE-39B0XCP-E-GL, IMX636, serial 4110039066) with
`store.load` + `analysis.spectrum(fmin_hz=2, fmax_hz=2000)`, other parameters at their defaults (bin_us=100,
min_snr=8). Signal: ON − OFF events per 100 µs bin. snr = peak amplitude / median amplitude in 2–2000 Hz.
The camera was not opened for this analysis.

| recording_id | reference_hz | f0_hz | error_hz | resolution_hz | snr | actual_s | early_stop |
|---|---|---|---|---|---|---|---|
| 20261016-142758-e07515 | 50 | 49.216 | -0.784 | 3.076 | 42.1 | 0.325007 | true |
| 20261016-143041-19997d | 50 | 49.61 | -0.39 | 2.3624 | 66.3 | 0.423264 | true |
| 20261016-143449-ed6b74 | 10 | 10.329 | +0.329 | 0.6455 | 34.2 | 1.549017 | true |

Per-recording JSON (schema `lumen.flicker.v1`): one file per recording_id in this folder.

snr of trial 3 is computed by the report script at the f0 bin (same formula), because spectrum() reports snr
only for its 5 strongest peaks and 10.33 Hz is not one of them.

## Result per trial

- Trials 1 and 2 (50 Hz): f0 is correct within half the resolution (−0.78 Hz at 3.08 Hz resolution,
  −0.39 Hz at 2.36 Hz). In trial 1 the strongest peak is 150.7 Hz (3rd harmonic); the code finds
  49.2 Hz before and after the fix.
- Trial 3 (10 Hz): f0 = 10.329 Hz (error +0.33 Hz at 0.65 Hz resolution). **Before fix: 70.36 Hz.** The
  spectrum has strong odd harmonics of 10.33 Hz, all with almost the same height (snr: 10.3 Hz → 34, 30.3 → 41,
  50.4 → 45, 70.4 → 47, 90.4 → 45, 110.4 → 41; even harmonics ≤ 2). The 5 strongest peaks (70, 50, 90, 211,
  231 Hz) did not contain 10 Hz, and the old code fell back to the dominant peak. The fix tests the divisors
  f_max / k (k = 2..15) of the strongest peak on the whole spectrum; 70.36 / 7 is accepted (strong at f and at
  ≥ 2 multiples), so f0 = 10.329 Hz.

## Limitations

- All three recordings hit the 30 M-event cap (light too bright) and stopped early: 0.33 s, 0.42 s and 1.55 s of
  the requested 2.0 s. So the resolution is only 0.65–3.08 Hz, and two of the three recordings are shorter than
  the 1 s minimum for a precise frequency.
- The reference frequency was changed between trial 2 and trial 3 (50 Hz → 10 Hz). So this is not a 3-trial repeat
  at one frequency. Do not compute a mean or a spread over the three trials.
- Duty cycle and distance were not recorded (null). The odd-only harmonics suggest a duty near 50 %.
- `harmonics_hz` lists only harmonics with snr ≥ 8. Even harmonics are at noise level (snr ≤ 4.6).
- The fix only tests k ≤ 15: if the strongest peak of a square wave is above 15f (possible when the harmonics
  stay flat up to fmax), f0 can still be wrong. Here the strongest peaks were 3f (trial 1), f (trial 2) and 7f
  (trial 3).
