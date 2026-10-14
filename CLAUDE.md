# Working rules for this project

I am a student. I must understand every line of code in this repository.
My English is not perfect; use simple English.

## How we work
- We work in SMALL steps. Do only the step I ask for, nothing more.
- Before you edit any file, show me a short plan: which files, which functions,
  and why. Then ASK me to approve it. Do not edit until I say yes.
- If anything is unclear or has more than one good option, STOP and ask me.
  Give me 2-3 options with pros and cons, and your recommendation.
- Never change code outside the current step. Never delete tests.
- Never run `git push` or `git commit` unless I ask.

## After every step, give me a report
1. Files changed (with line ranges).
2. The new code, explained in simple English, block by block.
3. Why this design, and what the alternative was.
4. How to test it (exact command), and the test result.
5. 1-2 questions to check that I understood (I will answer them).
6. What the next step will be. Then wait for me.

## HW3 hardware notes
- Device: IDS UE-39B0XCP-E-GL event camera, Sony IMX636, 1280x720, serial 4110039066, USB3. No lens:
  it measures total light change over time, not an image.
- SDK: OpenEB 5.2.0 built in ~/openeb/build + IDS HAL plugin; venv ~/mv-venv (Python 3.14).
  The shell alias `evcam` sets PYTHONPATH, LD_LIBRARY_PATH, MV_HAL_PLUGIN_PATH; nothing works without it.
- Verified Python API:
  ```python
  dev = metavision_hal.DeviceDiscovery.open("")
  b = dev.get_i_ll_biases(); b.set("bias_diff_off", 20); b.get_all_biases()
  # biases: bias_diff, bias_diff_on, bias_diff_off, bias_fo, bias_hpf, bias_refr (0 = default, values are offsets)
  m = dev.get_i_digital_event_mask(); m.get_pixel_masks()[i].set_mask(x, y, True)   # 64 slots
  metavision_core.event_io.EventsIterator.from_device(device=dev, delta_t=10000)  # -> numpy arrays x, y, p, t (us)
  ```
  Recording twice from the same open device works.
- Two hot pixels (1053,704) and (926,151) give ~90% of events (~9 Mev/s in the dark); masking leaves ~0.7 Mev/s.
- Dark noise vs bias_diff_off (hot pixels masked): 0:0.75, 10:0.56, 20:0.33, 40:0.058, 60:0.003, 80:0.0005 Mev/s.
- Start-up burst after opening or bias change: skip the first 0.5 s of every recording.
- stderr floods with "[HAL][WARNING] Evt3 protocol violation: InvalidVectBase"; harmless.
- Settings reset when the device is reopened. Only one process can open the camera (close metavision_viewer).
- Planned test target (hardware available tomorrow): ESP32 on-board LED sending Manchester frames
  (1 = low->high, 0 = high->low, MSB first): 0xAA 0xAA 0x7E <len> <ASCII text> <XOR of text bytes>,
  then LED low 300 ms. Bit rate 100..4000 bit/s.
- Today's real light sources: room lamp (mains flicker, expected 100 Hz) and laptop screen backlight (PWM).