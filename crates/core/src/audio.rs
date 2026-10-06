//! FMOD FSB5 sample banks (`.fsb`) — header decode and sub-sound enumeration.
//!
//! State of Decay: YOSE ships 226 `.fsb` banks under `sounds/`. Every one is
//! FSB5 version 1; the container is fully self-describing and is parsed here
//! without any third-party crate.
//!
//! # Verified layout
//!
//! The layout below was checked against all 226 real files
//! (`extracted-game/sounds/**/*.fsb`); for every file
//! `header + sampleHeaders + nameTable + dataSize == fileLen`.
//!
//! ## File header (60 bytes; `version == 0` adds a trailing `u32`)
//!
//! ```text
//! 0x00 char[4]  magic            = "FSB5"
//! 0x04 u32      version          = 1 in all YOSE banks
//! 0x08 u32      numSamples
//! 0x0C u32      sampleHeadersSize total bytes of the sample-header table
//! 0x10 u32      nameTableSize    total bytes of the name table (offsets + strings)
//! 0x14 u32      dataSize         total bytes of compressed sample payload
//! 0x18 u32      mode             codec id shared by every sub-sound (see `format_name`)
//! 0x1C u8[8]    zero
//! 0x24 u8[16]   hash / guid
//! 0x34 u8[8]    dummy
//! 0x3C ...      sample-header table (`sampleHeadersSize` bytes, variable stride)
//! ```
//!
//! ## Sample header (variable, minimum 8 bytes)
//!
//! The first 8 bytes are one little-endian `u64` of packed bit fields
//! (LSB-first bit order):
//!
//! ```text
//! bits  0..0   next_chunk       1 => one or more metadata chunks follow
//! bits  1..4   frequency_index  0..10, see FREQUENCY_TABLE (overridden by a FREQUENCY chunk)
//! bits  5..6   channels         0 => 1, 1 => 2, 2 => 6, 3 => 8 (overridden by a CHANNELS chunk)
//! bits  7..33  data_offset      in 32-byte units, relative to the start of the data area
//! bits 34..63  num_samples      PCM frame count
//!
//! Note: because the offset unit is 32 bytes, bit 6 (the would-be low bit of a
//! 16-byte-unit interpretation) is always zero, so older parsers that read
//! `bits 6..33 * 16` agree with this one on every conforming file.
//! ```
//!
//! While `next_chunk != 0`, a `u32` chunk header follows:
//!
//! ```text
//! bits  0..0   next_chunk
//! bits  1..24  chunk_size       payload byte count
//! bits 25..31  chunk_type       1 CHANNELS, 2 FREQUENCY, 3 LOOP, 6 XMA seek,
//!                               7 DSP coeffs, 10 XWMA, 11 VORBIS data
//! ```
//!
//! Chunk payloads then follow inline; `CHANNELS` is 1 byte, `FREQUENCY` a
//! little-endian `u32`, `LOOP` two `u32`s. Other types are length-skipped.
//!
//! ## Name table (present when `nameTableSize != 0`)
//!
//! `numSamples` `u32` offsets relative to the table start, followed by the
//! NUL-terminated UTF-8 names. The data area begins at
//! `0x3C (+4 if version 0) + sampleHeadersSize + nameTableSize`, and sample
//! `i` occupies `[data_offset_i, data_offset_{i+1})`, the last running to
//! `dataSize`. All offsets are in 16-byte units.
//!
//! # Observed content
//!
//! All 226 YOSE banks use codec id 11 (`MPEG`/MP3): 3,424 sub-sounds total,
//! 44.1 kHz dominant (also 22.05/32/48 kHz), mostly mono with some stereo and
//! 4-channel ambisonic beds. The data payload begins with an MPEG frame sync,
//! which is a convenient independent confirmation of the codec id.

pub const FSB5_MAGIC: &[u8; 4] = b"FSB5";

/// Size of the fixed FSB5 file header (before the optional version-0 `u32`).
const HEADER_SIZE: usize = 0x3C;

/// Largest sane sub-sound count; rejects absurd values before allocation.
const MAX_SUBSOUNDS: usize = 1_000_000;

/// Metadata chunk types used by the per-sample header.
const CHUNK_CHANNELS: u32 = 1;
const CHUNK_FREQUENCY: u32 = 2;

/// Frequency-index lookup for the packed sample-header field.
const FREQUENCY_TABLE: [u32; 11] = [4000, 8000, 11000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000];

/// One sub-sound of an FSB5 bank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsbSample {
    /// PCM frame count (`num_samples` in the packed header).
    pub num_samples: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u16,
    /// Codec id (the bank-wide `mode` field); see [`format_name`].
    pub format: u32,
}

/// A parsed FSB5 sample bank.
#[derive(Debug, Clone)]
pub struct Fsb {
    /// FSB container version (1 for all YOSE banks).
    pub version: u32,
    /// Number of sub-sounds (`numSamples`).
    pub num_subsounds: usize,
    /// Per-sub-sound metadata, in file order.
    pub samples: Vec<FsbSample>,
    /// First sub-sound name, when a name table is present.
    pub name: Option<String>,
}

/// Internal parse result keeping the absolute payload ranges that
/// [`FsbSample`] deliberately does not expose.
struct Parsed {
    fsb: Fsb,
    /// `(start, end)` absolute byte ranges into the source buffer, one per
    /// sub-sound, clamped to the buffer length.
    ranges: Vec<(usize, usize)>,
}

#[inline]
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[inline]
fn u64le(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3], b[o + 4], b[o + 5], b[o + 6], b[o + 7]])
}

/// Extract `len` bits starting at `start` (LSB-first); `len` must be < 64.
#[inline]
fn bits(v: u64, start: u32, len: u32) -> u64 {
    (v >> start) & ((1u64 << len) - 1)
}

/// Map the packed frequency index to a sample rate (0 = unknown).
fn rate_from_index(idx: u64) -> u32 {
    FREQUENCY_TABLE.get(idx as usize).copied().unwrap_or(0)
}

/// Human-readable name for an FSB5 codec id.
pub fn format_name(codec: u32) -> &'static str {
    match codec {
        0 => "None",
        1 => "PCM8",
        2 => "PCM16",
        3 => "PCM24",
        4 => "PCM32",
        5 => "PCMFloat",
        6 => "GCADPCM",
        7 => "IMAADPCM",
        8 => "VAG",
        9 => "HEVAG",
        10 => "XMA",
        11 => "MPEG",
        12 => "CELT",
        13 => "AT9",
        14 => "XWMA",
        15 => "Vorbis",
        16 => "FADPCM",
        17 => "Opus",
        _ => "Unknown",
    }
}

/// Parse an FSB5 bank, returning its header and sub-sound metadata.
///
/// Returns `None` for non-FSB5 data or structurally truncated input; it never
/// panics and never allocates based on attacker-controlled counts beyond what
/// the declared table sizes justify.
pub fn parse_fsb(bytes: &[u8]) -> Option<Fsb> {
    parse_full(bytes).map(|p| p.fsb)
}

/// Return the raw, still-compressed payload of sub-sound `idx` plus its
/// metadata, or `None` if `idx` is out of range or the file is malformed.
pub fn extract_sample(bytes: &[u8], idx: usize) -> Option<(&[u8], FsbSample)> {
    let parsed = parse_full(bytes)?;
    let sample = parsed.fsb.samples.get(idx)?.clone();
    let &(start, end) = parsed.ranges.get(idx)?;
    if start > end || end > bytes.len() {
        return None;
    }
    Some((&bytes[start..end], sample))
}

fn parse_full(bytes: &[u8]) -> Option<Parsed> {
    if bytes.len() < HEADER_SIZE || &bytes[0..4] != FSB5_MAGIC {
        return None;
    }

    let version = u32le(bytes, 0x04);
    let num_samples = u32le(bytes, 0x08) as usize;
    let sample_header_size = u32le(bytes, 0x0C) as usize;
    let name_table_size = u32le(bytes, 0x10) as usize;
    let data_size = u32le(bytes, 0x14) as usize;
    let codec = u32le(bytes, 0x18);

    if num_samples > MAX_SUBSOUNDS {
        return None;
    }
    // Each sample header is at least 8 bytes, so the declared table size caps
    // the count. This also stops a tiny file from claiming millions of samples.
    if num_samples > 0 && sample_header_size / 8 < num_samples {
        return None;
    }

    let mut pos = HEADER_SIZE;
    if version == 0 {
        pos = pos.checked_add(4)?;
        if bytes.len() < pos {
            return None;
        }
    }

    let sample_table_end = pos.checked_add(sample_header_size)?;
    let name_table_end = sample_table_end.checked_add(name_table_size)?;
    let data_start = name_table_end;
    if data_start > bytes.len() {
        return None;
    }

    let mut samples: Vec<FsbSample> = Vec::with_capacity(num_samples);
    let mut offsets: Vec<usize> = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        if pos + 8 > bytes.len() {
            return None;
        }
        let raw = u64le(bytes, pos);
        pos += 8;

        let mut next = bits(raw, 0, 1);
        let freq_index = bits(raw, 1, 4);
        let mut channels = match bits(raw, 5, 2) {
            0 => 1u16,
            1 => 2,
            2 => 6,
            3 => 8,
            _ => unreachable!(),
        };
        let data_offset = (bits(raw, 7, 27) as usize).checked_mul(32)?;
        let num_frames = bits(raw, 34, 30) as u32;
        let mut sample_rate = rate_from_index(freq_index);

        // Walk the optional metadata chunk chain.
        let mut guard = 0u32;
        while next != 0 {
            guard += 1;
            if guard > 64 || pos + 4 > bytes.len() {
                return None;
            }
            let chunk = u32le(bytes, pos) as u64;
            pos += 4;
            next = bits(chunk, 0, 1);
            let chunk_size = bits(chunk, 1, 24) as usize;
            let chunk_type = bits(chunk, 25, 7) as u32;
            if pos + chunk_size > bytes.len() {
                return None;
            }
            let payload = &bytes[pos..pos + chunk_size];
            pos += chunk_size;

            match chunk_type {
                CHUNK_CHANNELS if chunk_size >= 1 => channels = payload[0] as u16,
                CHUNK_FREQUENCY if chunk_size >= 4 => sample_rate = u32le(payload, 0),
                _ => {}
            }
        }

        offsets.push(data_offset);
        samples.push(FsbSample {
            num_samples: num_frames,
            sample_rate,
            channels,
            format: codec,
        });
    }

    // Name table: `num_samples` u32 offsets then NUL-terminated strings.
    let mut names: Vec<Option<String>> = Vec::new();
    if name_table_size > 0 {
        let table = &bytes[sample_table_end..name_table_end];
        if table.len() >= num_samples * 4 {
            for i in 0..num_samples {
                let name_start = u32le(table, i * 4) as usize;
                if name_start >= table.len() {
                    names.push(None);
                    continue;
                }
                let rest = &table[name_start..];
                let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
                names.push(Some(String::from_utf8_lossy(&rest[..end]).into_owned()));
            }
        }
    }

    // Resolve absolute payload ranges. Sample i runs to the next sample's
    // offset, the last to the declared data size; everything is clamped.
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let start = data_start.saturating_add(offsets[i]).min(bytes.len());
        let end = if i + 1 < num_samples {
            data_start.saturating_add(offsets[i + 1]).min(bytes.len())
        } else {
            data_start.saturating_add(data_size).min(bytes.len())
        };
        if end < start {
            ranges.push((start, start));
        } else {
            ranges.push((start, end));
        }
    }

    let name = names.first().cloned().flatten();
    Some(Parsed {
        fsb: Fsb {
            version,
            num_subsounds: num_samples,
            samples,
            name,
        },
        ranges,
    })
}

// ---------------------------------------------------------------------------
// PCM decoding
// ---------------------------------------------------------------------------
//
// Only the uncompressed / ADPCM codecs that can be decoded with pure `std`
// are supported here:
//
//   * 1  PCM8   - unsigned 8-bit, channel-interleaved
//   * 2  PCM16  - little-endian signed 16-bit, channel-interleaved
//   * 7  IMAADPCM - Xbox/FSB IMA ADPCM (36-byte blocks)
//
// MPEG (11), XMA (10), XWMA (14), Vorbis (15), CELT (12), Opus (17), AT9 (13),
// GCADPCM (6), VAG/HEVAG (8/9) and FADPCM (16) require an external codec and
// return `None`. The 226 YOSE banks all use codec 11 (MPEG), so `decode_sample`
// is expected to fail on the shipped game data until an MP3 decoder is linked.

/// Fully decoded, channel-interleaved 16-bit PCM.
///
/// Samples are laid out L R L R ... for stereo, or simply in order for mono.
/// `bits` records the source bit depth (8 or 16); `pcm` is always expanded to
/// signed 16-bit so callers have a single storage type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedAudio {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u16,
    /// Source bit depth of the decoded payload (8 or 16).
    pub bits: u16,
    /// Interleaved signed 16-bit PCM samples.
    pub pcm: Vec<i16>,
}

/// FSB codec ids this module can decode without an external library.
const CODEC_PCM8: u32 = 1;
const CODEC_PCM16: u32 = 2;
const CODEC_IMAADPCM: u32 = 7;

/// Return the FSB container generation from the magic bytes (`FSB1`..`FSB5`),
/// or `None` if the buffer does not start with a known FSB magic.
///
/// This is the *container* generation, not the internal version field (which
/// for FSB5 is exposed as [`Fsb::version`]).
pub fn container_version(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 4 {
        return None;
    }
    match &bytes[0..4] {
        b"FSB5" => Some(5),
        b"FSB4" => Some(4),
        b"FSB3" => Some(3),
        b"FSB2" => Some(2),
        b"FSB1" => Some(1),
        _ => None,
    }
}

/// Decode sub-sound `idx` of an FSB bank to interleaved 16-bit PCM.
///
/// Returns `None` for an out-of-range index, malformed data, or a codec that
/// needs an external decoder (MPEG/XMA/WMA/Vorbis/...). Never panics.
pub fn decode_sample(bytes: &[u8], idx: usize) -> Option<DecodedAudio> {
    let (payload, sample) = extract_sample(bytes, idx)?;
    decode_payload(payload, sample.sample_rate, sample.channels, sample.format, sample.num_samples)
}

/// Decode a raw sub-sound payload given its metadata.
fn decode_payload(
    payload: &[u8],
    sample_rate: u32,
    channels: u16,
    codec: u32,
    num_frames: u32,
) -> Option<DecodedAudio> {
    if payload.is_empty() || channels == 0 {
        return None;
    }
    let ch = channels as usize;
    let pcm = match codec {
        CODEC_PCM16 => decode_pcm16(payload, ch, num_frames),
        CODEC_PCM8 => decode_pcm8(payload, ch, num_frames),
        CODEC_IMAADPCM => decode_ima(payload, ch, num_frames),
        _ => return None, // external codec required
    };
    let bits = if codec == CODEC_PCM8 { 8 } else { 16 };
    Some(DecodedAudio { sample_rate, channels, bits, pcm })
}

/// Trim a decoded buffer to the declared frame count when it is known and sane.
fn trim_frames(pcm: &mut Vec<i16>, channels: usize, num_frames: u32) {
    if num_frames > 0 {
        let want = (num_frames as usize).saturating_mul(channels);
        if pcm.len() > want {
            pcm.truncate(want);
        }
    }
}

/// Little-endian signed 16-bit PCM (interleaved).
fn decode_pcm16(payload: &[u8], channels: usize, num_frames: u32) -> Vec<i16> {
    let mut pcm: Vec<i16> = payload
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    trim_frames(&mut pcm, channels, num_frames);
    pcm
}

/// Unsigned 8-bit PCM (interleaved), centred and expanded to 16-bit.
fn decode_pcm8(payload: &[u8], channels: usize, num_frames: u32) -> Vec<i16> {
    let mut pcm: Vec<i16> = payload.iter().map(|&b| ((b as i16) - 128) << 8).collect();
    trim_frames(&mut pcm, channels, num_frames);
    pcm
}

/// Standard IMA ADPCM step-size table (89 valid entries, index 0..=88).
const IMA_STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408,
    449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630,
    9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767,
];

/// IMA ADPCM index adjustment per 4-bit code.
const IMA_INDEX_TABLE: [i32; 16] =
    [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

/// Expand one IMA nibble (`shift` 0 = low nibble, 4 = high nibble).
#[inline]
fn ima_expand_nibble(byte: u8, shift: u32, hist: &mut i32, index: &mut i32) {
    let code = ((byte >> shift) & 0x0f) as usize;
    let step = IMA_STEP_TABLE[*index as usize];
    let mut delta = step >> 3;
    if code & 1 != 0 {
        delta += step >> 2;
    }
    if code & 2 != 0 {
        delta += step >> 1;
    }
    if code & 4 != 0 {
        delta += step;
    }
    if code & 8 != 0 {
        delta = -delta;
    }
    *hist = (*hist + delta).clamp(-32768, 32767);
    *index = (*index + IMA_INDEX_TABLE[code]).clamp(0, 88);
}

/// Decode FMOD FSB IMA ADPCM.
///
/// FSB stores IMA in fixed 0x24-byte (36-byte) per-channel blocks, using the
/// Xbox layout: each block starts with a signed 16-bit history sample and an
/// 8-bit step index (plus a reserved byte), followed by 32 bytes = 64 nibbles.
/// Mono and stereo use one / two 36-byte sub-blocks per block; three or more
/// channels use the interleaved-header "FSB IMA" variant. The final nibble of
/// every block is padding, so each block yields 64 samples per channel. Layout
/// verified against vgmstream's `decode_xbox_ima` / `decode_fsb_ima`.
fn decode_ima(payload: &[u8], channels: usize, num_frames: u32) -> Vec<i16> {
    let block_size = 0x24 * channels;
    let mut pcm: Vec<i16> = Vec::new();
    let mut pos = 0usize;
    while pos < payload.len() {
        let end = (pos + block_size).min(payload.len());
        let block = &payload[pos..end];
        let per_channel = decode_ima_block(block, channels);
        let n = per_channel.iter().map(|c| c.len()).min().unwrap_or(0);
        for s in 0..n {
            for ch in &per_channel {
                pcm.push(ch[s]);
            }
        }
        pos += block_size;
    }
    trim_frames(&mut pcm, channels, num_frames);
    pcm
}

/// Decode one IMA block into a per-channel sample buffer.
fn decode_ima_block(block: &[u8], channels: usize) -> Vec<Vec<i16>> {
    let mut out: Vec<Vec<i16>> = Vec::with_capacity(channels);
    for c in 0..channels {
        let mut samples: Vec<i16> = Vec::with_capacity(64);
        let (hist_off, step_off) = if channels <= 2 {
            (4 * c, 4 * c + 2)
        } else {
            (2 * c, 2 * channels + 2 * c)
        };
        if hist_off + 2 > block.len() || step_off >= block.len() {
            out.push(samples);
            continue;
        }
        let mut hist = i16::from_le_bytes([block[hist_off], block[hist_off + 1]]) as i32;
        let mut index = (block[step_off] as i32).clamp(0, 88);
        samples.push(hist as i16);
        for n in 0..63usize {
            let byte_i = if channels == 1 {
                4 + n / 2
            } else if channels == 2 {
                8 + 4 * c + (n / 8) * 8 + (n % 8) / 2
            } else {
                4 * channels + 2 * c + (n / 4) * 2 * channels + (n % 4) / 2
            };
            if byte_i >= block.len() {
                break;
            }
            let shift = if n % 2 == 0 { 0 } else { 4 };
            ima_expand_nibble(block[byte_i], shift, &mut hist, &mut index);
            samples.push(hist as i16);
        }
        out.push(samples);
    }
    out
}

/// Serialise decoded audio as a canonical 44-byte-header RIFF/WAVE file.
///
/// 8-bit `bits` produces unsigned 8-bit PCM, anything else produces signed
/// 16-bit little-endian PCM (and reports 16 in the header). Useful for golden
/// tests and for handing audio to an OS player.
pub fn to_wav(a: &DecodedAudio) -> Vec<u8> {
    let channels = a.channels.max(1);
    let use_8bit = a.bits == 8;

    let data: Vec<u8> = if use_8bit {
        a.pcm
            .iter()
            .map(|&s| (((s >> 8) as i32) + 128).clamp(0, 255) as u8)
            .collect()
    } else {
        let mut d = Vec::with_capacity(a.pcm.len() * 2);
        for &s in &a.pcm {
            d.extend_from_slice(&s.to_le_bytes());
        }
        d
    };
    let bits: u16 = if use_8bit { 8 } else { 16 };
    let block_align: u16 = channels * (bits / 8);
    let byte_rate: u32 = a.sample_rate.saturating_mul(block_align as u32);
    let data_len = data.len() as u32;

    let mut out: Vec<u8> = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32.saturating_add(data_len)).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&a.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(&data);
    out
}

// ---------------------------------------------------------------------------
// MPEG / MP3 elementary-stream extraction
// ---------------------------------------------------------------------------
//
// Every shipped `*.fsb` bank uses codec 11 (MPEG), which [`decode_sample`]
// deliberately cannot decode. The compressed payload is nevertheless a normal
// MPEG audio elementary stream: it starts with a frame sync (`0xFF 0xEx/0xFx`)
// and consists of back-to-back MPEG frames. FMOD pads each frame with zero
// bytes so that the next frame begins at a per-sample fixed stride (a 4- or
// 16-byte multiple of the frame size); the padding must be stripped before a
// decoder will accept the stream cleanly.
//
// These functions reconstruct a clean MP3 stream by walking the standard MPEG
// frame headers (bitrate / sample-rate tables) and dropping the padding
// between frames. They never allocate based on untrusted lengths and never
// panic.

/// FSB codec id for MPEG audio (MP3).
pub const CODEC_MPEG: u32 = 11;

/// Maximum zero padding (bytes) observed between two MPEG frames. Real banks
/// top out at 15 (a 16-byte-aligned fixed stride), so this is a generous cap
/// for resynchronising to the next frame.
const MAX_MPEG_PAD: usize = 64;

/// MPEG1 Layer I..III bitrates (kbit/s), indexed by the 4-bit bitrate index.
/// Index 0 (free) and 15 (forbidden) are rejected by the caller.
const MPEG1_BITRATES: [[u32; 15]; 3] = [
    [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
    [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
    [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
];

/// MPEG2 / MPEG2.5 Layer I..III bitrates (kbit/s).
const MPEG2_BITRATES: [[u32; 15]; 3] = [
    [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
];

/// True when `bytes` is an FSB5 bank whose codec id is MPEG (11).
pub fn is_mpeg(bytes: &[u8]) -> bool {
    bytes.len() >= HEADER_SIZE && &bytes[0..4] == FSB5_MAGIC && u32le(bytes, 0x18) == CODEC_MPEG
}

/// Length in bytes of the MPEG frame starting at `off`, or `None` when the
/// bytes there are not a valid frame header. Uses the standard MPEG version /
/// layer / bitrate / sample-rate tables.
fn mpeg_frame_len(buf: &[u8], off: usize) -> Option<usize> {
    if off + 4 > buf.len() {
        return None;
    }
    if buf[off] != 0xFF || (buf[off + 1] & 0xE0) != 0xE0 {
        return None;
    }
    let version = (buf[off + 1] >> 3) & 0x03; // 0=2.5, 2=2, 3=1
    let layer = (buf[off + 1] >> 1) & 0x03; // 3=L1, 2=L2, 1=L3
    if version == 1 || layer == 0 {
        return None;
    }
    let bitrate_index = ((buf[off + 2] >> 4) & 0x0F) as usize;
    let rate_index = ((buf[off + 2] >> 2) & 0x03) as usize;
    let padding = ((buf[off + 2] >> 1) & 0x01) as usize;
    if bitrate_index == 0 || bitrate_index >= 15 || rate_index == 3 {
        return None;
    }

    let tables = if version == 3 { &MPEG1_BITRATES } else { &MPEG2_BITRATES };
    // Index tables by layer code: layer 3 -> slot 0, layer 2 -> slot 1, layer
    // 1 (Layer III) -> slot 2.
    let table_idx = match layer {
        3 => 0,
        2 => 1,
        _ => 2,
    };
    let bitrate = tables[table_idx][bitrate_index] as usize; // kbit/s
    if bitrate == 0 {
        return None;
    }
    let sample_rate = match (version, rate_index) {
        (3, 0) => 44100,
        (3, 1) => 48000,
        (3, 2) => 32000,
        (2, 0) => 22050,
        (2, 1) => 24000,
        (2, 2) => 16000,
        (0, 0) => 11025,
        (0, 1) => 12000,
        (0, 2) => 8000,
        _ => return None,
    };

    let len = match layer {
        3 => (12 * bitrate * 1000 / sample_rate + padding) * 4, // Layer I
        2 => 144 * bitrate * 1000 / sample_rate + padding,      // Layer II
        _ => {
            // Layer III: MPEG1 uses 144, MPEG2/2.5 uses 72.
            let coeff = if version == 3 { 144 } else { 72 };
            coeff * bitrate * 1000 / sample_rate + padding
        }
    };
    if len < 4 {
        return None;
    }
    Some(len)
}

/// First byte offset `>= from` (within `from..from+window`) that starts a valid
/// MPEG frame, or `None`. Requiring a fully valid header avoids false syncs in
/// compressed data.
fn find_mpeg_sync(buf: &[u8], from: usize, window: usize) -> Option<usize> {
    let limit = from.saturating_add(window).min(buf.len());
    let mut p = from;
    while p + 4 <= limit {
        if mpeg_frame_len(buf, p).is_some() {
            return Some(p);
        }
        p += 1;
    }
    None
}

/// Extract the raw MPEG/MP3 elementary stream for sub-sound `idx`.
///
/// The returned bytes begin at the sub-sound's first MPEG sync and contain the
/// concatenated MPEG frames with the FSB sub-block padding removed. Returns
/// `None` for non-MPEG banks, an out-of-range index, malformed input, or when
/// no frame sync can be found. This never panics.
pub fn extract_mpeg_stream(bytes: &[u8], idx: usize) -> Option<Vec<u8>> {
    if !is_mpeg(bytes) {
        return None;
    }
    let (payload, _sample) = extract_sample(bytes, idx)?;
    if payload.len() < 4 {
        return None;
    }
    let first = find_mpeg_sync(payload, 0, payload.len())?;

    let mut out: Vec<u8> = Vec::with_capacity(payload.len().saturating_sub(first));
    let mut off = first;
    loop {
        let len = match mpeg_frame_len(payload, off) {
            Some(l) => l,
            None => break,
        };
        if off + len > payload.len() {
            break; // truncated trailing frame: stop cleanly
        }
        out.extend_from_slice(&payload[off..off + len]);
        match find_mpeg_sync(payload, off + len, MAX_MPEG_PAD) {
            Some(next) => off = next,
            None => break,
        }
    }

    if out.is_empty() {
        // Could not walk frames (e.g. a non-standard stream): fall back to the
        // contiguous compressed bytes starting at the first sync.
        Some(payload[first..].to_vec())
    } else {
        Some(out)
    }
}

/// Best-effort count of MPEG frames in an elementary `stream` by walking frame
/// headers until an invalid sync is seen.
pub fn count_mpeg_frames(stream: &[u8]) -> usize {
    let mut off = 0usize;
    let mut n = 0usize;
    while off + 4 <= stream.len() {
        match mpeg_frame_len(stream, off) {
            Some(len) if len > 0 => {
                n += 1;
                off += len;
            }
            _ => break,
        }
    }
    n
}

/// Dump the extracted MPEG stream for sub-sound `idx` to `path`.
///
/// Returns `Ok(false)` (writing nothing) when [`extract_mpeg_stream`] yields
/// no stream, otherwise `Ok(true)` after a successful write.
pub fn write_mpeg_file(bytes: &[u8], idx: usize, path: &std::path::Path) -> std::io::Result<bool> {
    match extract_mpeg_stream(bytes, idx) {
        Some(stream) => {
            std::fs::write(path, stream)?;
            Ok(true)
        }
        None => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic 2-sub-sound FSB5 bank:
    ///   - sample 0: MPEG, 44100 Hz, 1 ch, 1000 frames, no metadata chunk,
    ///     64 bytes of payload.
    ///   - sample 1: MPEG, 48000 Hz via frequency index 9, 2 ch, 2000 frames,
    ///     one LOOP chunk, 16 bytes of payload.
    fn synthetic() -> Vec<u8> {
        let mode = 11u32; // MPEG

        let raw0: u64 = (8u64 << 1) | (1000u64 << 34); // idx 8, 1ch, off 0, 1000
        let raw1: u64 = 1 // next_chunk
            | (9u64 << 1)          // idx 9 => 48000
            | (1u64 << 5)          // 2 channels
            | (4u64 << 6)          // data offset 4 * 16 = 64
            | (2000u64 << 34); // 2000 frames

        let mut sample_headers: Vec<u8> = Vec::new();
        sample_headers.extend_from_slice(&raw0.to_le_bytes());
        sample_headers.extend_from_slice(&raw1.to_le_bytes());
        // LOOP chunk: next=0, size=8, type=3, then two u32s.
        let chunk: u32 = (8u32 << 1) | (3u32 << 25);
        sample_headers.extend_from_slice(&chunk.to_le_bytes());
        sample_headers.extend_from_slice(&0u32.to_le_bytes());
        sample_headers.extend_from_slice(&0x1d67u32.to_le_bytes());

        // Name table: two u32 offsets (8), "gun\0" at 8, "shot\0" at 12.
        let mut names: Vec<u8> = Vec::new();
        names.extend_from_slice(&8u32.to_le_bytes());
        names.extend_from_slice(&12u32.to_le_bytes());
        names.extend_from_slice(b"gun\0");
        names.extend_from_slice(b"shot\0");

        let data_size = 80usize;
        let sample_header_size = sample_headers.len();
        let name_table_size = names.len();

        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(FSB5_MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes()); // version
        out.extend_from_slice(&2u32.to_le_bytes()); // numSamples
        out.extend_from_slice(&(sample_header_size as u32).to_le_bytes());
        out.extend_from_slice(&(name_table_size as u32).to_le_bytes());
        out.extend_from_slice(&(data_size as u32).to_le_bytes());
        out.extend_from_slice(&mode.to_le_bytes());
        out.extend_from_slice(&[0u8; 8]); // zero
        out.extend_from_slice(&[0xAAu8; 16]); // hash
        out.extend_from_slice(&[0u8; 8]); // dummy
        debug_assert_eq!(out.len(), HEADER_SIZE);
        out.extend_from_slice(&sample_headers);
        out.extend_from_slice(&names);
        out.extend(std::iter::repeat(0x11u8).take(64)); // sample 0 data
        out.extend(std::iter::repeat(0x22u8).take(16)); // sample 1 data
        out
    }

    #[test]
    fn parses_synthetic_bank() {
        let b = synthetic();
        let fsb = parse_fsb(&b).expect("should parse");
        assert_eq!(fsb.version, 1);
        assert_eq!(fsb.num_subsounds, 2);
        assert_eq!(fsb.name.as_deref(), Some("gun"));
        assert_eq!(
            fsb.samples[0],
            FsbSample { num_samples: 1000, sample_rate: 44100, channels: 1, format: 11 }
        );
        assert_eq!(
            fsb.samples[1],
            FsbSample { num_samples: 2000, sample_rate: 48000, channels: 2, format: 11 }
        );
    }

    #[test]
    fn extracts_payload_ranges() {
        let b = synthetic();
        let (d0, s0) = extract_sample(&b, 0).expect("sample 0");
        assert_eq!(d0.len(), 64);
        assert!(d0.iter().all(|&x| x == 0x11));
        assert_eq!(s0.num_samples, 1000);

        let (d1, s1) = extract_sample(&b, 1).expect("sample 1");
        assert_eq!(d1.len(), 16);
        assert!(d1.iter().all(|&x| x == 0x22));
        assert_eq!(s1.channels, 2);

        assert!(extract_sample(&b, 2).is_none());
    }

    #[test]
    fn format_names_cover_common_codecs() {
        assert_eq!(format_name(15), "Vorbis");
        assert_eq!(format_name(16), "FADPCM");
        assert_eq!(format_name(11), "MPEG");
        assert_eq!(format_name(2), "PCM16");
        assert_eq!(format_name(1), "PCM8");
        assert_eq!(format_name(10), "XMA");
        assert_eq!(format_name(999), "Unknown");
    }

    #[test]
    fn malformed_input_does_not_panic() {
        assert!(parse_fsb(&[]).is_none());
        assert!(parse_fsb(b"FSB5").is_none());
        assert!(parse_fsb(b"NOPE..........").is_none());
        assert!(extract_sample(&[], 0).is_none());

        // Valid magic but a truncated header.
        let mut b = synthetic();
        b.truncate(20);
        assert!(parse_fsb(&b).is_none());

        // Corrupt the declared sample-header size to point past EOF.
        let mut b = synthetic();
        b[0x0C..0x10].copy_from_slice(&1_000_000u32.to_le_bytes());
        assert!(parse_fsb(&b).is_none());

        // Absurd sub-sound count.
        let mut b = synthetic();
        b[0x08..0x0C].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse_fsb(&b).is_none());

        // Exhaust every truncation length of a valid file.
        let full = synthetic();
        for n in 0..full.len() {
            let _ = parse_fsb(&full[..n]);
            let _ = extract_sample(&full[..n], 1);
        }
    }

    // -----------------------------------------------------------------------
    // Decoder tests
    // -----------------------------------------------------------------------

    /// Build a minimal single-sub-sound FSB5 bank around `payload`.
    fn build_bank(codec: u32, freq_idx: u64, channels: u16, frames: u32, payload: &[u8]) -> Vec<u8> {
        let ch_code: u64 = match channels {
            1 => 0,
            2 => 1,
            6 => 2,
            8 => 3,
            _ => 0,
        };
        let raw: u64 = (freq_idx << 1) | (ch_code << 5) | ((frames as u64) << 34);

        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(FSB5_MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes()); // version
        out.extend_from_slice(&1u32.to_le_bytes()); // numSamples
        out.extend_from_slice(&8u32.to_le_bytes()); // sampleHeadersSize
        out.extend_from_slice(&0u32.to_le_bytes()); // nameTableSize
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&codec.to_le_bytes());
        out.extend_from_slice(&[0u8; 8]);
        out.extend_from_slice(&[0xAAu8; 16]);
        out.extend_from_slice(&[0u8; 8]);
        out.extend_from_slice(&raw.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn u16_at(b: &[u8], o: usize) -> u16 {
        u16::from_le_bytes([b[o], b[o + 1]])
    }

    #[test]
    fn container_version_reads_magic() {
        assert_eq!(container_version(b"FSB5\x01\x00\x00\x00"), Some(5));
        assert_eq!(container_version(b"FSB4"), Some(4));
        let mut f4 = vec![0u8; 0x14];
        f4[0..4].copy_from_slice(b"FSB4");
        f4[0x10..0x14].copy_from_slice(&0x0004_0000u32.to_le_bytes());
        assert_eq!(container_version(&f4), Some(4));
        assert_eq!(container_version(b"nope"), None);
        assert_eq!(container_version(b"FS"), None);
    }

    #[test]
    fn decode_pcm16_and_wav_roundtrip() {
        let samples: Vec<i16> = vec![0, 100, -100, 32767, -32768, 1234, -5678, 42];
        let mut payload: Vec<u8> = Vec::new();
        for s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }
        let bank = build_bank(CODEC_PCM16, 8, 2, (samples.len() / 2) as u32, &payload);

        let a = decode_sample(&bank, 0).expect("pcm16 decodes");
        assert_eq!(a.sample_rate, 44100);
        assert_eq!(a.channels, 2);
        assert_eq!(a.bits, 16);
        assert_eq!(a.pcm, samples);

        let w = to_wav(&a);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(&w[8..12], b"WAVE");
        assert_eq!(u32le(&w, 4), (w.len() - 8) as u32);
        assert_eq!(&w[12..16], b"fmt ");
        assert_eq!(u32le(&w, 16), 16); // fmt chunk size
        assert_eq!(u16_at(&w, 20), 1); // PCM
        assert_eq!(u16_at(&w, 22), 2); // channels
        assert_eq!(u32le(&w, 24), 44100);
        assert_eq!(u32le(&w, 28), 44100 * 2 * 2); // byte rate
        assert_eq!(u16_at(&w, 32), 4); // block align
        assert_eq!(u16_at(&w, 34), 16); // bits
        assert_eq!(&w[36..40], b"data");
        assert_eq!(u32le(&w, 40), (samples.len() * 2) as u32);
        assert_eq!(w.len(), 44 + samples.len() * 2);
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(i16::from_le_bytes([w[44 + i * 2], w[45 + i * 2]]), *s);
        }
    }

    #[test]
    fn to_wav_handles_hand_built_audio() {
        let a = DecodedAudio { sample_rate: 8000, channels: 1, bits: 16, pcm: vec![1, -1] };
        let w = to_wav(&a);
        assert_eq!(w.len(), 48);
        assert_eq!(u32le(&w, 4), 40);
        assert_eq!(u32le(&w, 24), 8000);
        assert_eq!(u32le(&w, 28), 8000 * 2);
        assert_eq!(u16_at(&w, 32), 2);
        assert_eq!(u32le(&w, 40), 4);
    }

    #[test]
    fn decode_pcm8_is_unsigned_centred() {
        // 2 stereo frames: (128,255), (0,130)
        let payload = vec![128u8, 255, 0, 130];
        let bank = build_bank(CODEC_PCM8, 8, 2, 2, &payload);
        let a = decode_sample(&bank, 0).expect("pcm8 decodes");
        assert_eq!(a.bits, 8);
        assert_eq!(a.pcm, vec![0, (255 - 128) << 8, (0 - 128) << 8, (130 - 128) << 8]);

        // to_wav writes valid 8-bit PCM back.
        let w = to_wav(&a);
        assert_eq!(u16_at(&w, 34), 8);
        assert_eq!(u16_at(&w, 32), 2);
        assert_eq!(&w[44..48], &payload[..]);
    }

    #[test]
    fn decode_ima_mono_block() {
        // One 36-byte block: header (hist, step, reserved) + 32 nibble bytes.
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(&1000i16.to_le_bytes());
        payload.push(10);
        payload.push(0);
        payload.extend(std::iter::repeat(0x00u8).take(32));
        let bank = build_bank(CODEC_IMAADPCM, 8, 1, 64, &payload);

        let a = decode_sample(&bank, 0).expect("ima decodes");
        assert_eq!(a.channels, 1);
        assert_eq!(a.bits, 16);
        // 1 header sample + 63 nibbles, trimmed to the declared 64 frames.
        assert_eq!(a.pcm.len(), 64);
        assert_eq!(a.pcm[0], 1000);
        assert!(!a.pcm.is_empty());
    }

    #[test]
    fn decode_ima_stereo_block() {
        // One 72-byte stereo block: 2x(hist+step+reserved) + 2x32 nibble bytes.
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(&100i16.to_le_bytes());
        payload.push(5);
        payload.push(0);
        payload.extend_from_slice(&(-100i16).to_le_bytes());
        payload.push(5);
        payload.push(0);
        payload.extend(std::iter::repeat(0x11u8).take(64));
        let bank = build_bank(CODEC_IMAADPCM, 8, 2, 64, &payload);

        let a = decode_sample(&bank, 0).expect("ima stereo decodes");
        assert_eq!(a.channels, 2);
        assert_eq!(a.pcm.len(), 128);
        assert_eq!(a.pcm[0], 100);
        assert_eq!(a.pcm[1], -100);
    }

    #[test]
    fn decode_malformed_returns_none_without_panic() {
        assert!(decode_sample(&[], 0).is_none());
        assert!(decode_sample(b"FSB5", 0).is_none());
        assert!(decode_sample(b"not an fsb", 0).is_none());

        let good = build_bank(CODEC_PCM16, 8, 1, 4, &[0u8; 8]);
        assert!(decode_sample(&good, 1).is_none()); // index out of range

        // Unsupported codec (MPEG=11) parses but cannot be decoded here.
        let mpeg = build_bank(11, 8, 1, 4, &[0u8; 40]);
        assert!(decode_sample(&mpeg, 0).is_none());

        // Corrupt table size and absurd counts must not panic.
        let mut bad = good.clone();
        bad[0x0C..0x10].copy_from_slice(&1_000_000u32.to_le_bytes());
        assert!(decode_sample(&bad, 0).is_none());

        // Exhaust every truncation.
        let full = build_bank(CODEC_PCM16, 8, 2, 8, &[0u8; 32]);
        for n in 0..full.len() {
            let _ = decode_sample(&full[..n], 0);
        }
    }

    #[test]
    fn decodes_real_bank_when_supported() {
        // Resolve the repo's extracted assets relative to this crate; skip
        // silently when they are absent (e.g. a detached checkout).
        let rel = "../../../extracted-game/sounds/sounds/weapons/wpn_ar_762_sup.fsb";
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return,
        };

        assert_eq!(container_version(&bytes), Some(5));
        let fsb = parse_fsb(&bytes).expect("real bank parses");
        assert!(fsb.num_subsounds >= 1);

        // All shipped YOSE banks are MPEG; only assert real PCM for codecs we
        // actually implement, otherwise skip.
        let supported = matches!(
            fsb.samples.first().map(|s| s.format),
            Some(CODEC_PCM8) | Some(CODEC_PCM16) | Some(CODEC_IMAADPCM)
        );
        if !supported {
            return;
        }
        let a = decode_sample(&bytes, 0).expect("supported codec decodes");
        assert!(!a.pcm.is_empty());
        assert!(a.sample_rate >= 8000);
        assert!(a.channels >= 1);
    }

    #[test]
    fn real_mpeg_banks_parse_but_do_not_decode() {
        let rel = "../../../extracted-game/sounds/sounds/weapons/wpn_ar_762_sup.fsb";
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let fsb = parse_fsb(&bytes).expect("real bank parses");
        if fsb.samples.first().map(|s| s.format) == Some(11) {
            // MPEG is intentionally unsupported at the container layer here.
            assert!(decode_sample(&bytes, 0).is_none());
        }
    }

    // -----------------------------------------------------------------------
    // MPEG extraction tests
    // -----------------------------------------------------------------------

    /// A syntactically valid MPEG1 Layer III 44.1 kHz 128 kbit/s frame:
    /// `pad_bit` selects the 417- or 418-byte frame size.
    fn mpeg_frame(pad_bit: bool, len: usize) -> Vec<u8> {
        let mut f = vec![0u8; len];
        f[0] = 0xFF;
        f[1] = 0xFB; // MPEG1, Layer III
        f[2] = if pad_bit { 0x92 } else { 0x90 };
        f[3] = 0x00;
        f
    }

    /// Build a bank whose single MPEG sample is three frames on the 420-byte
    /// fixed stride FMOD uses, with zero padding between them.
    fn mpeg_bank() -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&mpeg_frame(false, 417));
        payload.extend_from_slice(&[0u8; 3]); // pad to 420
        payload.extend_from_slice(&mpeg_frame(true, 418));
        payload.extend_from_slice(&[0u8; 2]); // pad to 420
        payload.extend_from_slice(&mpeg_frame(false, 417));
        payload.extend_from_slice(&[0u8; 3]); // pad to 420
        build_bank(CODEC_MPEG, 8, 1, 0, &payload)
    }

    #[test]
    fn is_mpeg_detects_codec() {
        assert!(is_mpeg(&mpeg_bank()));
        assert!(is_mpeg(&synthetic())); // codec 11 synthetic bank
        assert!(!is_mpeg(&build_bank(CODEC_PCM16, 8, 1, 4, &[0u8; 8])));
        assert!(!is_mpeg(&[]));
        assert!(!is_mpeg(b"FSB5"));
        assert!(!is_mpeg(b"not an fsb"));
    }

    #[test]
    fn extract_mpeg_strips_padding_and_counts_frames() {
        let bank = mpeg_bank();
        let stream = extract_mpeg_stream(&bank, 0).expect("synth mpeg stream");
        // 417 + 418 + 417 frame bytes, padding removed.
        assert_eq!(stream.len(), 417 + 418 + 417);
        assert_eq!(stream[0], 0xFF);
        assert_eq!(stream[1] & 0xE0, 0xE0);
        assert_eq!(count_mpeg_frames(&stream), 3);
        // The full padded payload was 1260 bytes; padding must be gone.
        assert!(stream.len() < 1260);
    }

    #[test]
    fn extract_mpeg_returns_none_for_bad_inputs() {
        let bank = mpeg_bank();
        assert!(extract_mpeg_stream(&bank, 1).is_none()); // bad index
        assert!(extract_mpeg_stream(&[], 0).is_none());
        assert!(extract_mpeg_stream(b"FSB5", 0).is_none());
        assert!(extract_mpeg_stream(b"not an fsb", 0).is_none());

        // Non-MPEG bank.
        let pcm = build_bank(CODEC_PCM16, 8, 1, 4, &[0u8; 8]);
        assert!(extract_mpeg_stream(&pcm, 0).is_none());

        // MPEG container whose payload contains no frame sync.
        assert!(extract_mpeg_stream(&synthetic(), 0).is_none());

        // Every truncation must return without panicking.
        for n in 0..bank.len() {
            let _ = extract_mpeg_stream(&bank[..n], 0);
            let _ = count_mpeg_frames(&bank[..n]);
        }
    }

    #[test]
    fn write_mpeg_file_handles_absent_stream() {
        let bank = build_bank(CODEC_PCM16, 8, 1, 4, &[0u8; 8]);
        let p = std::path::Path::new("/tmp/opencode/_sod_never_write.mp3");
        assert_eq!(write_mpeg_file(&bank, 0, p).unwrap(), false);
    }

    #[test]
    fn extracts_real_mpeg_bank_when_present() {
        let rel = "../../../extracted-game/sounds/sounds/weapons/wpn_ar_762_sup.fsb";
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return, // skip silently when assets are absent
        };

        assert!(is_mpeg(&bytes));
        let stream = extract_mpeg_stream(&bytes, 0).expect("real mpeg stream");
        assert!(!stream.is_empty());
        assert_eq!(stream[0], 0xFF);
        assert_eq!(stream[1] & 0xE0, 0xE0);
        assert!(count_mpeg_frames(&stream) > 0);

        // Out-of-range index on a real bank.
        assert!(extract_mpeg_stream(&bytes, 999).is_none());

        // Malformed/truncated real data must not panic.
        for n in [0usize, 4, 0x3C, 100, bytes.len() / 2] {
            let _ = extract_mpeg_stream(&bytes[..n.min(bytes.len())], 0);
        }

        // Dump one .mp3 for optional manual verification; failure is ignored.
        let out = std::path::Path::new("/tmp/opencode/wpn_ar_762_sup.mp3");
        let _ = std::fs::create_dir_all("/tmp/opencode");
        let _ = write_mpeg_file(&bytes, 0, out);
    }
}
