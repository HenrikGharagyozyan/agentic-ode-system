# HW3 hardware notes

## Device, SDK and environment
- Camera: IDS UE-39B0XCP-E-GL event camera, Sony IMX636 sensor, 1280x720 pixels, serial 4110039066, USB3.
- No lens: the camera measures the total light change over time, not an image.
- SDK: OpenEB 5.2.0, built in `~/openeb/build`, plus the IDS HAL plugin.
- Python: venv `~/mv-venv` (Python 3.14).
- Run the shell alias `evcam` first. It sets `PYTHONPATH`, `LD_LIBRARY_PATH` and `MV_HAL_PLUGIN_PATH`;
  nothing works without it.
- Only one process can open the camera at a time (close `metavision_viewer` first).
- Settings reset when the device is reopened.
- stderr floods with `[HAL][WARNING] Evt3 protocol violation: InvalidVectBase`; this is harmless.

## Verified Python API
```python
dev = metavision_hal.DeviceDiscovery.open("")

b = dev.get_i_ll_biases()
b.set("bias_diff_off", 20)
b.get_all_biases()

m = dev.get_i_digital_event_mask()
m.get_pixel_masks()[i].set_mask(x, y, True)   # 64 slots

metavision_core.event_io.EventsIterator.from_device(device=dev, delta_t=10000)  # -> numpy arrays x, y, p, t (us)
```
- Bias names: `bias_diff`, `bias_diff_on`, `bias_diff_off`, `bias_fo`, `bias_hpf`, `bias_refr`.
  Values are offsets; 0 = default.
- Recording twice from the same open device: the stream time **pauses** between recordings and continues
  where the last one stopped (verified 2026-10-14: recording 1 ended at t = 1.000 s; recording 2 started 2 s
  later on the wall clock, and its first event had t = 1.010 s). A new `EventsIterator` starts its slice clock
  at 0, so it first yields empty **catch-up slices** very fast (1000 slices in ~37 ms) before the real ones.
  `EventCamera.record()` remembers where the last recording stopped in `_t_end` (reset to 0 in `open()`)
  and skips every slice that ends at or before it.

## Facilities and defaults
Read with `runs/probe/02_read_values.py` on 2026-10-14, right after opening, nothing changed.

- 19 of 24 `dev.get_i_*()` facilities exist. Not available: `event_frame_diff_decoder`,
  `event_frame_histo_decoder`, `event_rate`, `roi_pixel_mask`, `trigger_out`.
- All on-chip filters are **off** at open:
  - `antiflicker_module`: BandStop, band 50..520 Hz (= supported min/max), duty cycle 50 %, start/stop threshold 6/4.
    Must stay disabled for flicker measurements.
  - `erc_module` (event rate controller): 50,000,000 ev/s, count period 200.
  - `event_trail_filter_module`: type TRAIL, threshold 10000 (range 1000..100000);
    types TRAIL, STC_CUT_TRAIL, STC_KEEP_TRAIL.
  - `digital_crop`: region (0, 0, 0, 0).
- All biases are 0 at open. Ranges from `get_bias_info(name)` (recommended = `get_bias_range()`):

  | bias          | recommended | allowed   | category                      |
  |---------------|-------------|-----------|-------------------------------|
  | bias_diff     | -25..23     | -150..200 | Contrast                      |
  | bias_diff_on  | -85..140    | -150..200 | Contrast                      |
  | bias_diff_off | -35..190    | -150..200 | Contrast                      |
  | bias_fo       | -35..55     | -150..200 | Bandwidth (low-pass)          |
  | bias_hpf      | 0..120      | 0..255    | Bandwidth (high-pass)         |
  | bias_refr     | -20..235    | -50..255  | Advanced (refractory period)  |

- Device config options: `format`, `ll_biases_range_check_bypass` (not tested).
- ROI: at most 1 window. Modes `ROI` and `RONI` (not tested yet). `Window(x, y, width, height)`:
  verified `Window(1, 2, 3, 4)` -> x=1 y=2 width=3 height=4, `to_string()` = `'[1,2,3x4]'`.
  `set_window(Window) -> bool`, `set_windows(list)`, `set_mode(Mode) -> bool`, `set_lines(cols, rows)`,
  `enable(bool) -> bool`.
- Monitoring: `get_temperature()` = 35, `get_pixel_dead_time()` = 73 (units not documented).
  `get_illumination()` is not supported: returns -1 and logs `[HAL][ERROR] Failed to get illumination`.
- Data format EVT3 (EVT21 also available). Camera firmware 3.10.255, IDS plugin 5.0.0.
- `camera_synchronization.get_mode()` and `trigger_in.is_enabled()` need an argument; not explored.

## Noise behaviour
Mev/s = million events per second.

- Two hot pixels, (1053, 704) and (926, 151), give ~90% of all events (~9 Mev/s in the dark).
  With both masked: ~0.7 Mev/s.
- Known limitation (seen once, 2026-10-15, MCP smoke test, phone screen on the sensor): the first recording after
  open() put the events of both hot pixels on (926, 151) (8.67 MHz) and none on (1053, 704); cause not found
  (not our analysis code, not stale mask slots, probe 10). Safety net: the MCP `record` tool warns about pixels
  above 20 kHz.
- Dark noise vs `bias_diff_off` (hot pixels masked):

  | bias_diff_off | 0    | 10   | 20   | 40    | 60    | 80     |
  |---------------|------|------|------|-------|-------|--------|
  | Mev/s         | 0.75 | 0.56 | 0.33 | 0.058 | 0.003 | 0.0005 |

- Start-up burst after opening the device or changing a bias: skip the first 0.5 s of every recording.

## Light sources and test target
- Light sources available on 2026-10-14: ceiling lamp (mains flicker, expected 100 Hz), laptop screen backlight (PWM).
- ESP32 test target available from 2026-10-15: the on-board LED sends Manchester frames.
  - Bit coding: 1 = low->high, 0 = high->low, MSB first.
  - Frame: `0xAA 0xAA 0x7E <len> <ASCII text> <XOR of text bytes>`, then the LED stays low for 300 ms.
  - Bit rate: 100..4000 bit/s.
