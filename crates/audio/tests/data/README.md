# Audio test data

Ogg Vorbis files for `tests/vorbis_decoding.rs` (FIX-7.4), encoded with `oggenc` from vorbis-tools 1.4.3 at quality 4 from 16-bit WAV files written by a script:

| File | Content | Command |
|---|---|---|
| `tone_440_mono_48k.ogg` | 1 s of a 440 Hz sine at amplitude 0.5, mono, 48 kHz | `oggenc -Q -q 4 --serial 1 -o tone_440_mono_48k.ogg tone_440_mono_48k.wav` |
| `tones_440_660_stereo_44k.ogg` | 1 s, 440 Hz on the left and 660 Hz on the right, amplitude 0.5, 44.1 kHz | `oggenc -Q -q 4 --serial 2 -o tones_440_660_stereo_44k.ogg tones_440_660_stereo_44k.wav` |

The WAV files hold `0.5 * sin(2π f n / rate)` rounded to 16 bits, for `n` over one second of frames.
