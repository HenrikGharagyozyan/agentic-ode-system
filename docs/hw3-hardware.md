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
- Recording twice from the same open device works.

## Noise behaviour
Mev/s = million events per second.

- Two hot pixels, (1053, 704) and (926, 151), give ~90% of all events (~9 Mev/s in the dark).
  With both masked: ~0.7 Mev/s.
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
