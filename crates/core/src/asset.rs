//! CryEngine asset recognition and parsing (milestone 1).
//!
//! Implemented:
//!   * `.dds`  - DDS header parse (dimensions, mips, pixel format).
//!   * `.mtl`  - CryEngine XML material attribute extraction.
//!   * `.cgf/.cga/.chr/.caf/.anm` - CryEngine chunk container.
//!
//! Chunk layout verified against 7,662 real files (see notes/cgf-format.md and
//! tools/cgf_inspect.py):
//!   header:  "CryTek\0" sig, u32 fileType@0x08, u32 version@0x0C,
//!            u32 chunkTableOffset@0x10 (=0x14), u32 numChunks@0x14
//!   records: at 0x18, 20 bytes each: u32 type, u32 version, u32 fileOffset,
//!            u32 id, u32 size. `type` carries a namespace in its high 16 bits
//!            (0xCCCC/0xACDC/0xAAFC), e.g. 0xCCCC0016 = DataStream.

/// 7-byte CryEngine chunk-file signature; byte 7 is padding.
pub const CRY_CHUNK_SIG: &[u8; 7] = b"CryTek\0";
pub const DDS_MAGIC: &[u8; 4] = b"DDS ";
pub const CHUNK_TABLE_OFFSET: usize = 0x18;
pub const CHUNK_STRIDE: usize = 20;
pub const FORMAT_VERSION: u32 = 0x745;

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

// ---------------------------------------------------------------------------
// DDS
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DdsInfo {
    pub width: u32,
    pub height: u32,
    pub mipmaps: u32,
    pub depth: u32,
    pub fourcc: [u8; 4],
    pub rgb_bits: u32,
}

impl DdsInfo {
    pub fn format_name(&self) -> String {
        match &self.fourcc {
            b"DXT1" => "DXT1/BC1".into(),
            b"DXT3" => "DXT3/BC2".into(),
            b"DXT5" => "DXT5/BC3".into(),
            b"ATI2" | b"ATI1" => "ATI2/BC5".into(),
            b"BC4U" | b"BC4S" => "BC4".into(),
            b"DX10" => "DX10 (extended)".into(),
            [0, 0, 0, 0] => format!("uncompressed {}bpp", self.rgb_bits),
            other => String::from_utf8_lossy(other).into_owned(),
        }
    }
}

pub fn parse_dds(b: &[u8]) -> Option<DdsInfo> {
    if b.len() < 128 || &b[0..4] != DDS_MAGIC {
        return None;
    }
    Some(DdsInfo {
        height: u32le(b, 12),
        width: u32le(b, 16),
        depth: u32le(b, 24),
        mipmaps: u32le(b, 28).max(1),
        fourcc: [b[84], b[85], b[86], b[87]],
        rgb_bits: u32le(b, 88),
    })
}

// ---------------------------------------------------------------------------
// CryEngine chunk files (.cgf/.cga/.chr/.caf/.anm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ChunkRecord {
    /// Full namespaced type, e.g. 0xCCCC0016 (DataStream).
    pub chunk_type: u32,
    pub version: u32,
    pub file_offset: u32,
    pub id: u32,
    pub size: u32,
}

impl ChunkRecord {
    /// Low 16 bits identify the chunk kind within its namespace.
    pub fn kind(&self) -> u16 {
        (self.chunk_type & 0xFFFF) as u16
    }
    pub fn namespace(&self) -> u16 {
        (self.chunk_type >> 16) as u16
    }

    /// Payload bytes (chunk body, after the 16-byte embedded header).
    /// `size` is the whole chunk size including that header.
    pub fn payload<'a>(&self, file: &'a [u8]) -> Option<&'a [u8]> {
        let start = (self.file_offset as usize).checked_add(16)?;
        let end = (self.file_offset as usize).checked_add(self.size as usize)?;
        if start <= end && end <= file.len() {
            Some(&file[start..end])
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct CryChunkFile {
    pub file_type: u32,
    pub format_version: u32,
    pub num_chunks: u32,
    pub chunks: Vec<ChunkRecord>,
}

/// Best-effort chunk names. The `0xCCCC` namespace carries the mesh chunks;
/// the tail ids present in State of Decay are 0x0E, 0x13..0x18 (see notes).
pub fn chunk_type_name(kind: u16) -> &'static str {
    match kind {
        0x00 => "Mesh",
        0x01 => "Helper",
        0x02 => "VertAnim",
        0x03 => "BoneAnim",
        0x04 => "GeomNameList",
        0x05 => "BoneNameList",
        0x06 => "MtlName/SceneProps",
        0x08 => "Controller",
        0x09 => "Timing",
        0x0a => "BoneMesh",
        0x0b => "Node",
        0x0e => "Timing",
        0x13 => "SourceInfo",
        0x14 => "MtlName",
        0x15 => "ExportFlags",
        0x16 => "DataStream",
        0x17 => "MeshSubsets",
        0x18 => "MeshPhysicsData",
        _ => "Unknown",
    }
}

pub fn is_cry_chunk_file(b: &[u8]) -> bool {
    b.len() >= 0x18 && &b[0..7] == CRY_CHUNK_SIG
}

pub fn parse_cry_chunk_file(b: &[u8]) -> Option<CryChunkFile> {
    if !is_cry_chunk_file(b) {
        return None;
    }
    let file_type = u32le(b, 8);
    let format_version = u32le(b, 0x0C);
    let num_chunks = u32le(b, 0x14);

    // Version >= 0x745 uses 20-byte records (with size); older uses 16 bytes
    // (size must be derived from the next chunk's offset).
    let stride = if format_version >= FORMAT_VERSION { 20 } else { 16 };
    let mut chunks = Vec::new();
    let mut offsets = Vec::new();
    let mut o = CHUNK_TABLE_OFFSET;
    while (chunks.len() as u32) < num_chunks && o + stride <= b.len() {
        let file_offset = u32le(b, o + 8);
        let size = if stride == 20 { u32le(b, o + 16) } else { 0 };
        chunks.push(ChunkRecord {
            chunk_type: u32le(b, o),
            version: u32le(b, o + 4),
            file_offset,
            id: u32le(b, o + 12),
            size,
        });
        offsets.push(file_offset);
        o += stride;
    }
    // Resolve any missing sizes (pre-0x745 files) from neighbour offsets.
    let mut sorted = offsets.clone();
    sorted.sort_unstable();
    for c in chunks.iter_mut() {
        if c.size == 0 {
            let next = sorted
                .iter()
                .find(|&&x| x > c.file_offset)
                .copied()
                .unwrap_or(b.len() as u32);
            c.size = next.saturating_sub(c.file_offset);
        }
    }
    Some(CryChunkFile { file_type, format_version, num_chunks, chunks })
}

// ---------------------------------------------------------------------------
// CryEngine XML material (.mtl)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Material {
    pub shader: String,
    pub surface_type: String,
    pub diffuse: String,
    pub specular: String,
    pub emissive: String,
    pub opacity: String,
    pub textures: Vec<(String, String)>, // (map, file)
}

fn attr<'a>(line: &'a str, key: &str) -> Option<String> {
    let pat = format!("{key}=\"");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

pub fn parse_mtl(b: &[u8]) -> Option<Material> {
    let text = String::from_utf8_lossy(b);
    let mut m = Material::default();
    if let Some(line) = text.lines().find(|l| l.trim_start().starts_with("<Material")) {
        m.shader = attr(line, "Shader").unwrap_or_default();
        m.surface_type = attr(line, "SurfaceType").unwrap_or_default();
        m.diffuse = attr(line, "Diffuse").unwrap_or_default();
        m.specular = attr(line, "Specular").unwrap_or_default();
        m.emissive = attr(line, "Emissive").unwrap_or_default();
        m.opacity = attr(line, "Opacity").unwrap_or_default();
    }
    for line in text.lines() {
        if line.contains("<Texture ") {
            if let (Some(map), Some(file)) = (attr(line, "Map"), attr(line, "File")) {
                m.textures.push((map, file));
            }
        }
    }
    Some(m)
}

// ---------------------------------------------------------------------------
// DDS -> RGBA8 decode (CPU, mip level 0 only)
// ---------------------------------------------------------------------------
//
// The game's textures are DirectDraw surfaces: mostly DXT1/BC1, DXT5/BC3,
// ATI2/BC5 (normal maps) plus uncompressed 32-bit BGRA/RGBA and 24-bit RGB.
// Decoding to a tight, top-left, RGBA8 buffer lets the renderer upload one
// fixed format instead of matching a dozen GPU formats.
//
// Colour formats are decoded identically to Pillow (verified byte-for-byte on
// every shipped DXT1/DXT5/BGRA/RGB24/L8 surface). Single-channel, alpha-only
// (`DDPF_ALPHA`) surfaces are the one case Pillow cannot open; they carry no
// colour, so the channel is replicated into RGB *and* kept as alpha. Emitting
// opaque white there (the previous behaviour) turned every coverage mask into a
// solid white card when its RGB was sampled as albedo.

/// A decoded, uncompressed image: row-major, 4 bytes/pixel, top-left origin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DdsImageRgba {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

const DDPF_ALPHAPIXELS: u32 = 0x0000_0001;
const DDPF_ALPHA: u32 = 0x0000_0002;
const DDPF_FOURCC: u32 = 0x0000_0004;
const DDPF_RGB: u32 = 0x0000_0040;
const DDPF_LUMINANCE: u32 = 0x0002_0000;

#[inline]
fn rgb565_to_rgb(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 0x1f) as u32;
    let g = ((c >> 5) & 0x3f) as u32;
    let b = (c & 0x1f) as u32;
    [
        ((r << 3) | (r >> 2)) as u8,
        ((g << 2) | (g >> 4)) as u8,
        ((b << 3) | (b >> 2)) as u8,
    ]
}

/// BC1 colour half (8 bytes). When `allow_1bit_alpha`, `c0 <= c1` selects the
/// 3-colour + transparent-black palette (used by BC1); otherwise the 4-colour
/// palette is always used (BC2/BC3).
fn bc1_color(block: &[u8], out: &mut [[u8; 4]; 16], allow_1bit_alpha: bool) {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let e0 = rgb565_to_rgb(c0);
    let e1 = rgb565_to_rgb(c1);
    let mut pal = [[0u8; 4]; 4];
    pal[0] = [e0[0], e0[1], e0[2], 255];
    pal[1] = [e1[0], e1[1], e1[2], 255];
    if c0 > c1 || !allow_1bit_alpha {
        for k in 0..3 {
            pal[2][k] = ((2 * e0[k] as u32 + e1[k] as u32) / 3) as u8;
            pal[3][k] = ((e0[k] as u32 + 2 * e1[k] as u32) / 3) as u8;
        }
        pal[2][3] = 255;
        pal[3][3] = 255;
    } else {
        for k in 0..3 {
            pal[2][k] = ((e0[k] as u32 + e1[k] as u32) / 2) as u8;
        }
        pal[2][3] = 255;
        pal[3] = [0, 0, 0, 0];
    }
    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for i in 0..16 {
        let idx = ((bits >> (2 * i)) & 0x3) as usize;
        out[i] = pal[idx];
    }
}

/// BC2/DXT3 colour half (16 bytes): 8 bytes of 4-bit alpha then an 8-byte
/// BC1-style colour block that always uses the 4-colour palette.
fn bc2_color(block: &[u8], out: &mut [[u8; 4]; 16]) {
    bc1_color(&block[8..16], out, false);
    let mut abits: u64 = 0;
    for i in 0..8 {
        abits |= (block[i] as u64) << (8 * i);
    }
    for i in 0..16 {
        let a = ((abits >> (4 * i)) & 0xF) as u32;
        out[i][3] = (a * 17) as u8;
    }
}

/// BC3/BC4 alpha half (8 bytes): two endpoints + 16 3-bit indices, LSB-first.
fn bc3_alpha(block: &[u8]) -> [u8; 16] {
    let a0 = block[0] as u32;
    let a1 = block[1] as u32;
    let mut pal = [0u8; 8];
    pal[0] = a0 as u8;
    pal[1] = a1 as u8;
    if a0 > a1 {
        for i in 1..=6u32 {
            pal[(i + 1) as usize] = (((7 - i) * a0 + i * a1) / 7) as u8;
        }
    } else {
        for i in 1..=4u32 {
            pal[(i + 1) as usize] = (((5 - i) * a0 + i * a1) / 5) as u8;
        }
        pal[6] = 0;
        pal[7] = 255;
    }
    let mut bits: u64 = 0;
    for i in 0..6 {
        bits |= (block[2 + i] as u64) << (8 * i);
    }
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = pal[((bits >> (3 * i)) & 0x7) as usize];
    }
    out
}

/// Decode a block-compressed surface into tight RGBA8 using `f` per block.
fn decode_blocks<F: FnMut(&[u8], &mut [[u8; 4]; 16])>(
    data: &[u8],
    width: u32,
    height: u32,
    block_bytes: usize,
    mut f: F,
) -> Option<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let bw = (w + 3) / 4;
    let bh = (h + 3) / 4;
    let need = bw.checked_mul(bh)?.checked_mul(block_bytes)?;
    if data.len() < need {
        return None;
    }
    let mut out = vec![0u8; w.checked_mul(h)?.checked_mul(4)?];
    let mut px = [[0u8; 4]; 16];
    for by in 0..bh {
        for bx in 0..bw {
            let off = (by * bw + bx) * block_bytes;
            f(&data[off..off + block_bytes], &mut px);
            for y in 0..4 {
                let py = by * 4 + y;
                if py >= h {
                    break;
                }
                for x in 0..4 {
                    let pxx = bx * 4 + x;
                    if pxx >= w {
                        break;
                    }
                    let d = (py * w + pxx) * 4;
                    out[d..d + 4].copy_from_slice(&px[y * 4 + x]);
                }
            }
        }
    }
    Some(out)
}

/// Unpack a tightly packed 4-byte RGBA8 (`is_rgba`) or BGRA8 surface.
fn decode_rgba_tight(data: &[u8], width: u32, height: u32, is_rgba: bool) -> Option<Vec<u8>> {
    let n = (width as usize).checked_mul(height as usize)?;
    if data.len() < n.checked_mul(4)? {
        return None;
    }
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        let p = &data[i * 4..i * 4 + 4];
        if is_rgba {
            out[i * 4..i * 4 + 4].copy_from_slice(p);
        } else {
            out[i * 4] = p[2];
            out[i * 4 + 1] = p[1];
            out[i * 4 + 2] = p[0];
            out[i * 4 + 3] = p[3];
        }
    }
    Some(out)
}

fn decode_fourcc(fourcc: &[u8], full: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let data = &full[128..];
    match fourcc {
        b"DXT1" => decode_blocks(data, width, height, 8, |b, px| bc1_color(b, px, true)),
        b"DXT3" => decode_blocks(data, width, height, 16, bc2_color),
        b"DXT5" => decode_blocks(data, width, height, 16, |b, px| {
            let a = bc3_alpha(&b[0..8]);
            bc1_color(&b[8..16], px, false);
            for i in 0..16 {
                px[i][3] = a[i];
            }
        }),
        // BC4: single channel replicated to RGB, opaque.
        b"ATI1" | b"BC4U" | b"BC4S" => decode_blocks(data, width, height, 8, |b, px| {
            let v = bc3_alpha(b);
            for i in 0..16 {
                px[i] = [v[i], v[i], v[i], 255];
            }
        }),
        // BC5: R = first channel, G = second; B = 128 (flat normal Z), A = 255.
        b"ATI2" | b"BC5U" | b"BC5S" => decode_blocks(data, width, height, 16, |b, px| {
            let r = bc3_alpha(&b[0..8]);
            let g = bc3_alpha(&b[8..16]);
            for i in 0..16 {
                px[i] = [r[i], g[i], 128, 255];
            }
        }),
        b"DX10" => decode_dx10(full, width, height),
        _ => None,
    }
}

fn decode_dx10(full: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    if full.len() < 148 {
        return None;
    }
    let fmt = u32le(full, 128);
    let data = &full[148..];
    match fmt {
        71 | 72 => decode_blocks(data, width, height, 8, |b, px| bc1_color(b, px, true)),
        74 | 75 => decode_blocks(data, width, height, 16, bc2_color),
        77 | 78 => decode_blocks(data, width, height, 16, |b, px| {
            let a = bc3_alpha(&b[0..8]);
            bc1_color(&b[8..16], px, false);
            for i in 0..16 {
                px[i][3] = a[i];
            }
        }),
        80 | 81 => decode_blocks(data, width, height, 8, |b, px| {
            let v = bc3_alpha(b);
            for i in 0..16 {
                px[i] = [v[i], v[i], v[i], 255];
            }
        }),
        83 | 84 => decode_blocks(data, width, height, 16, |b, px| {
            let r = bc3_alpha(&b[0..8]);
            let g = bc3_alpha(&b[8..16]);
            for i in 0..16 {
                px[i] = [r[i], g[i], 128, 255];
            }
        }),
        28 | 29 => decode_rgba_tight(data, width, height, true),
        87 | 88 | 91 | 93 => decode_rgba_tight(data, width, height, false),
        _ => None,
    }
}

/// Scale a masked, contiguously packed channel to 8 bits.
fn extract_channel(word: u32, mask: u32) -> u8 {
    if mask == 0 {
        return 0;
    }
    let shift = mask.trailing_zeros();
    let max = mask >> shift;
    let v = (word & mask) >> shift;
    ((v * 255) / max) as u8
}

fn decode_uncompressed(
    flags: u32,
    full: &[u8],
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    let bits = u32le(full, 88);
    let rm = u32le(full, 92);
    let gm = u32le(full, 96);
    let bm = u32le(full, 100);
    let am = u32le(full, 104);
    let data = &full[128..];
    let n = (width as usize).checked_mul(height as usize)?;
    let bpp = ((bits + 7) / 8) as usize;
    if bpp == 0 || bpp > 4 || data.len() < n.checked_mul(bpp)? {
        return None;
    }
    let mut out = vec![0u8; n * 4];
    let alpha_only = flags & DDPF_ALPHA != 0;
    let luminance = flags & DDPF_LUMINANCE != 0;
    for i in 0..n {
        let o = i * bpp;
        let mut word: u32 = 0;
        for k in 0..bpp {
            word |= (data[o + k] as u32) << (8 * k);
        }
        let has_alpha = flags & DDPF_ALPHAPIXELS != 0 && am != 0;
        let (r, g, b, a);
        if alpha_only && rm == 0 && gm == 0 && bm == 0 {
            // Alpha-only (`DDPF_ALPHA`) surface (e.g. an opacity/coverage mask).
            // It carries exactly one channel and no colour. Do not force RGB to
            // opaque white: that turns every single-channel mask into a solid
            // white "card" the moment its RGB is sampled as albedo. Replicate the
            // channel into RGB (exactly like the zero-mask / BC4 single-channel
            // paths) and keep it as the alpha, so the mask decodes as a grayscale
            // RGBA image instead of a white rectangle.
            let v = extract_channel(word, if am != 0 { am } else { 0xFF });
            r = v;
            g = v;
            b = v;
            a = v;
        } else if rm == 0 && gm == 0 && bm == 0 {
            // Luminance (or zero-mask fallback): replicate the single channel.
            let m = if luminance && am != 0 { am } else { 0xFF };
            let v = extract_channel(word, m);
            r = v;
            g = v;
            b = v;
            a = if has_alpha { extract_channel(word, am) } else { 255 };
        } else {
            r = extract_channel(word, rm);
            g = extract_channel(word, gm);
            b = extract_channel(word, bm);
            a = if has_alpha { extract_channel(word, am) } else { 255 };
        }
        out[i * 4] = r;
        out[i * 4 + 1] = g;
        out[i * 4 + 2] = b;
        out[i * 4 + 3] = a;
    }
    Some(out)
}

/// Decode mip level 0 of a `.dds` file to a tight RGBA8 image. Returns `None`
/// for unsupported formats or truncated data; never panics.
pub fn decode_dds_rgba(bytes: &[u8]) -> Option<DdsImageRgba> {
    if bytes.len() < 128 || &bytes[0..4] != DDS_MAGIC {
        return None;
    }
    let height = u32le(bytes, 12);
    let width = u32le(bytes, 16);
    if width == 0 || height == 0 {
        return None;
    }
    let flags = u32le(bytes, 80);
    let fourcc = &bytes[84..88];
    let rgba = if flags & DDPF_FOURCC != 0 {
        decode_fourcc(fourcc, bytes, width, height)?
    } else if flags & (DDPF_RGB | DDPF_LUMINANCE | DDPF_ALPHA | DDPF_ALPHAPIXELS) != 0 {
        decode_uncompressed(flags, bytes, width, height)?
    } else {
        return None;
    };
    Some(DdsImageRgba { width, height, rgba })
}

// ---------------------------------------------------------------------------
// Minimal PNG writer (stored deflate, std only)
// ---------------------------------------------------------------------------

fn crc32(bytes: &[u8]) -> u32 {
    let mut c: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            let m = (c & 1).wrapping_neg();
            c = (c >> 1) ^ (0xEDB8_8320 & m);
        }
    }
    !c
}

fn adler32(bytes: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &x in bytes {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_in = Vec::with_capacity(4 + data.len());
    crc_in.extend_from_slice(kind);
    crc_in.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
}

/// Encode a decoded image as an 8-bit RGBA PNG using only stored (uncompressed)
/// deflate blocks. Returns `None` if `image.rgba` does not match its dimensions.
pub fn write_dds_as_png(image: &DdsImageRgba) -> Option<Vec<u8>> {
    let w = image.width as usize;
    let h = image.height as usize;
    if image.rgba.len() != w.checked_mul(h)?.checked_mul(4)? || w == 0 || h == 0 {
        return None;
    }
    let stride = w * 4;
    let mut raw = Vec::with_capacity(h * (1 + stride));
    for y in 0..h {
        raw.push(0); // filter type: none
        raw.extend_from_slice(&image.rgba[y * stride..(y + 1) * stride]);
    }

    // zlib wrapper + stored deflate blocks.
    let mut z = Vec::with_capacity(raw.len() + 64);
    z.extend_from_slice(&[0x78, 0x01]);
    if raw.is_empty() {
        z.extend_from_slice(&[0x01, 0x00, 0x00, 0xFF, 0xFF]);
    }
    let mut pos = 0;
    while pos < raw.len() {
        let n = (raw.len() - pos).min(0xFFFF);
        let last = pos + n == raw.len();
        z.push(if last { 0x01 } else { 0x00 });
        z.extend_from_slice(&(n as u16).to_le_bytes());
        z.extend_from_slice(&(!(n as u16)).to_le_bytes());
        z.extend_from_slice(&raw[pos..pos + n]);
        pos += n;
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&image.width.to_be_bytes());
    ihdr.extend_from_slice(&image.height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA, deflate, no interlace
    png_chunk(&mut out, b"IHDR", &ihdr);
    png_chunk(&mut out, b"IDAT", &z);
    png_chunk(&mut out, b"IEND", &[]);
    Some(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 128-byte DDS header followed by caller-supplied mip-0 data.
    fn header(w: u32, h: u32, fourcc: &[u8; 4], flags: u32, bits: u32, masks: [u32; 4]) -> Vec<u8> {
        let mut b = vec![0u8; 128];
        b[0..4].copy_from_slice(b"DDS ");
        b[4..8].copy_from_slice(&124u32.to_le_bytes());
        b[8..12].copy_from_slice(&0x0002_1007u32.to_le_bytes());
        b[12..16].copy_from_slice(&h.to_le_bytes());
        b[16..20].copy_from_slice(&w.to_le_bytes());
        b[28..32].copy_from_slice(&1u32.to_le_bytes());
        b[76..80].copy_from_slice(&32u32.to_le_bytes());
        b[80..84].copy_from_slice(&flags.to_le_bytes());
        b[84..88].copy_from_slice(fourcc);
        b[88..92].copy_from_slice(&bits.to_le_bytes());
        for (i, m) in masks.iter().enumerate() {
            b[92 + i * 4..96 + i * 4].copy_from_slice(&m.to_le_bytes());
        }
        b
    }

    fn header_c(w: u32, h: u32, fourcc: &[u8; 4]) -> Vec<u8> {
        header(w, h, fourcc, DDPF_FOURCC, 0, [0; 4])
    }

    #[test]
    fn bc1_opaque_4x4() {
        let mut b = header_c(4, 4, b"DXT1");
        b.extend_from_slice(&0xF800u16.to_le_bytes()); // red
        b.extend_from_slice(&0x001Fu16.to_le_bytes()); // blue
        b.extend_from_slice(&[0xE4, 0, 0, 0]); // row 0 indices 0,1,2,3
        let img = decode_dds_rgba(&b).expect("decode bc1");
        assert_eq!((img.width, img.height), (4, 4));
        assert_eq!(img.rgba.len(), 64);
        assert_eq!(&img.rgba[0..4], &[255, 0, 0, 255]);
        assert_eq!(&img.rgba[4..8], &[0, 0, 255, 255]);
        assert_eq!(&img.rgba[8..12], &[170, 0, 85, 255]);
        assert_eq!(&img.rgba[12..16], &[85, 0, 170, 255]);
        // Remaining pixels are index 0 (red).
        assert_eq!(&img.rgba[16..20], &[255, 0, 0, 255]);
    }

    #[test]
    fn bc1_transparent_4x4() {
        let mut b = header_c(4, 4, b"DXT1");
        b.extend_from_slice(&0x0000u16.to_le_bytes()); // black (c0)
        b.extend_from_slice(&0xFFFFu16.to_le_bytes()); // white (c1); c0 <= c1
        b.extend_from_slice(&[0xE4, 0, 0, 0]);
        let img = decode_dds_rgba(&b).expect("decode bc1 alpha");
        assert_eq!(&img.rgba[0..4], &[0, 0, 0, 255]);
        assert_eq!(&img.rgba[4..8], &[255, 255, 255, 255]);
        assert_eq!(&img.rgba[8..12], &[127, 127, 127, 255]);
        assert_eq!(&img.rgba[12..16], &[0, 0, 0, 0]); // index 3 transparent
    }

    #[test]
    fn bc2_alpha_nibbles() {
        let mut b = header_c(4, 4, b"DXT3");
        b.extend_from_slice(&[0x0F, 0, 0, 0, 0, 0, 0, 0]); // pixel0 a=15, pixel1 a=0
        b.extend_from_slice(&0xF800u16.to_le_bytes()); // red
        b.extend_from_slice(&0x001Fu16.to_le_bytes()); // blue
        b.extend_from_slice(&[0xE4, 0, 0, 0]);
        let img = decode_dds_rgba(&b).expect("decode bc2");
        assert_eq!(&img.rgba[0..4], &[255, 0, 0, 255]);
        assert_eq!(&img.rgba[4..8], &[0, 0, 255, 0]);
    }

    #[test]
    fn bc3_alpha_and_color() {
        let mut b = header_c(4, 4, b"DXT5");
        // a0=255, a1=0, pixel 0 index 1 -> alpha 0, others index 0 -> 255.
        b.extend_from_slice(&[255, 0, 0x01, 0, 0, 0, 0, 0]);
        b.extend_from_slice(&0xFFFFu16.to_le_bytes()); // white
        b.extend_from_slice(&0x0000u16.to_le_bytes()); // black
        b.extend_from_slice(&[0xE4, 0, 0, 0]);
        let img = decode_dds_rgba(&b).expect("decode bc3");
        assert_eq!(&img.rgba[0..4], &[255, 255, 255, 0]);
        assert_eq!(&img.rgba[4..8], &[0, 0, 0, 255]);
        assert_eq!(&img.rgba[8..12], &[170, 170, 170, 255]);
        assert_eq!(&img.rgba[12..16], &[85, 85, 85, 255]);
        // Second row untouched by index bytes -> alpha 255.
        assert_eq!(img.rgba[4 * 4 + 3], 255);
    }

    #[test]
    fn bc4_luma_replicated() {
        let mut b = header_c(4, 4, b"ATI1");
        b.extend_from_slice(&[77, 0, 0, 0, 0, 0, 0, 0]);
        let img = decode_dds_rgba(&b).expect("decode bc4");
        for p in img.rgba.chunks(4) {
            assert_eq!(p, &[77, 77, 77, 255]);
        }
    }

    #[test]
    fn bc5_two_channel() {
        let mut b = header_c(4, 4, b"ATI2");
        b.extend_from_slice(&[100, 0, 0, 0, 0, 0, 0, 0]); // R = 100
        b.extend_from_slice(&[200, 0, 0, 0, 0, 0, 0, 0]); // G = 200
        let img = decode_dds_rgba(&b).expect("decode bc5");
        for p in img.rgba.chunks(4) {
            assert_eq!(p, &[100, 200, 128, 255]);
        }
    }

    #[test]
    fn uncompressed_rgba_2x2() {
        let mut b = header(
            2,
            2,
            b"\0\0\0\0",
            DDPF_RGB | DDPF_ALPHAPIXELS,
            32,
            [0x00FF_0000, 0x0000_FF00, 0x0000_00FF, 0xFF00_0000],
        );
        let px = [
            (10u8, 20u8, 30u8, 255u8),
            (40, 50, 60, 128),
            (70, 80, 90, 0),
            (100, 110, 120, 255),
        ];
        for (r, g, bl, a) in px {
            b.push(bl);
            b.push(g);
            b.push(r);
            b.push(a);
        }
        let img = decode_dds_rgba(&b).expect("decode rgba32");
        assert_eq!(
            img.rgba,
            vec![10, 20, 30, 255, 40, 50, 60, 128, 70, 80, 90, 0, 100, 110, 120, 255]
        );
    }

    #[test]
    fn uncompressed_rgb24_1x1() {
        let mut b = header(
            1,
            1,
            b"\0\0\0\0",
            DDPF_RGB,
            24,
            [0x00FF_0000, 0x0000_FF00, 0x0000_00FF, 0],
        );
        b.extend_from_slice(&[30, 20, 10]); // B, G, R -> (10, 20, 30)
        let img = decode_dds_rgba(&b).expect("decode rgb24");
        assert_eq!(img.rgba, vec![10, 20, 30, 255]);
    }

    #[test]
    fn truncated_and_unsupported_return_none() {
        assert!(decode_dds_rgba(&[]).is_none());
        assert!(decode_dds_rgba(&[0u8; 64]).is_none());

        // Unsupported FourCC.
        let mut bad = header_c(4, 4, b"ZZZZ");
        bad.extend_from_slice(&[0u8; 8]);
        assert!(decode_dds_rgba(&bad).is_none());

        // BC1 header but no block data -> truncated.
        let t = header_c(4, 4, b"DXT1");
        assert!(decode_dds_rgba(&t).is_none());

        // Four blocks declared, only one present.
        let mut short = header_c(8, 8, b"DXT1");
        short.extend_from_slice(&[0u8; 8]);
        assert!(decode_dds_rgba(&short).is_none());

        // Zero dimension.
        let z = header_c(0, 4, b"DXT1");
        assert!(decode_dds_rgba(&z).is_none());
    }

    #[test]
    fn png_writer_is_well_formed() {
        let img = DdsImageRgba {
            width: 2,
            height: 1,
            rgba: vec![255, 0, 0, 255, 0, 255, 0, 128],
        };
        let png = write_dds_as_png(&img).expect("encode png");
        assert_eq!(&png[0..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..20], &2u32.to_be_bytes());
        assert_eq!(&png[20..24], &1u32.to_be_bytes());
        assert_eq!(&png[24..26], &[8, 6]);
        assert!(png.ends_with(&[0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82]));
    }

    fn find_dds(dir: &std::path::Path, depth: u32) -> Option<std::path::PathBuf> {
        if depth > 6 {
            return None;
        }
        let rd = std::fs::read_dir(dir).ok()?;
        let mut subdirs = Vec::new();
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                subdirs.push(p);
            } else if p
                .extension()
                .map(|x| x.eq_ignore_ascii_case("dds"))
                .unwrap_or(false)
            {
                return Some(p);
            }
        }
        for d in subdirs {
            if let Some(p) = find_dds(&d, depth + 1) {
                return Some(p);
            }
        }
        None
    }

    /// Recursively find a foliage/vegetation `*diffuse*.dds` sample, if the
    /// extracted game is available.
    fn find_vegetation_diffuse(dir: &std::path::Path, depth: u32) -> Option<std::path::PathBuf> {
        if depth > 8 {
            return None;
        }
        let rd = std::fs::read_dir(dir).ok()?;
        let mut subdirs = Vec::new();
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name.eq_ignore_ascii_case("vegetation") || name.eq_ignore_ascii_case("foliage") {
                        if let Some(hit) = find_diffuse_in(&p, 0) {
                            return Some(hit);
                        }
                    }
                }
                subdirs.push(p);
            }
        }
        for d in subdirs {
            if let Some(p) = find_vegetation_diffuse(&d, depth + 1) {
                return Some(p);
            }
        }
        None
    }

    fn find_diffuse_in(dir: &std::path::Path, depth: u32) -> Option<std::path::PathBuf> {
        if depth > 8 {
            return None;
        }
        let rd = std::fs::read_dir(dir).ok()?;
        let mut subdirs = Vec::new();
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                subdirs.push(p);
            } else {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                let lower = name.to_ascii_lowercase();
                if lower.ends_with(".dds") && lower.contains("diffuse") {
                    return Some(p);
                }
            }
        }
        for d in subdirs {
            if let Some(p) = find_diffuse_in(&d, depth + 1) {
                return Some(p);
            }
        }
        None
    }

    /// Regression: a single-channel alpha-only (`DDPF_ALPHA`) surface must not be
    /// turned into an opaque white image. It should carry the channel in RGB as
    /// well as alpha, matching the other single-channel paths.
    #[test]
    fn alpha_only_surface_is_not_white() {
        let mut b = header(2, 1, b"\0\0\0\0", DDPF_ALPHA, 8, [0, 0, 0, 0xFF]);
        b.extend_from_slice(&[10, 200]);
        let img = decode_dds_rgba(&b).expect("decode alpha-only");
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.rgba.len(), 2 * 4);
        assert_eq!(img.rgba, vec![10, 10, 10, 10, 200, 200, 200, 200]);
    }

    /// Same defect on a real shipped one-channel asset: before the fix every
    /// texel decoded to opaque white (the `normalsfitting` mask is only ~4%
    /// 255-valued), so a mask rendered as albedo became a blank white card.
    #[test]
    fn alpha_only_engine_asset_not_white() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game");
        let path = root.join("shaders/shaders/engineassets/screenspace/normalsfitting.dds");
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let img = decode_dds_rgba(&bytes).expect("decode alpha-only asset");
        assert_eq!(img.rgba.len(), img.width as usize * img.height as usize * 4);
        let n = img.width as usize * img.height as usize;
        let white = img
            .rgba
            .chunks(4)
            .filter(|p| p[0] == 255 && p[1] == 255 && p[2] == 255)
            .count();
        assert!(white * 2 < n, "alpha-only asset decoded blank: {white}/{n}");
    }

    /// Real foliage diffuse maps must decode to actual leaf colour, not a blank
    /// white card. Pixels for `poplartree.dds` were hand-verified with Pillow
    /// (`Image.open(path).convert('RGBA')`, identical byte-for-byte).
    #[test]
    fn foliage_diffuse_decodes_to_colour() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game");
        let exact = root.join("objects/objects/landscape/vegetation/trees/poplar/poplartree.dds");
        let path = if exact.is_file() {
            exact
        } else {
            match find_vegetation_diffuse(&root, 0) {
                Some(p) => p,
                None => return, // no sample on disk: skip silently
            }
        };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let img = decode_dds_rgba(&bytes)
            .unwrap_or_else(|| panic!("foliage decode failed for {:?}", path));
        assert_eq!(
            img.rgba.len(),
            img.width as usize * img.height as usize * 4,
            "dimension/payload mismatch for {:?}",
            path
        );

        // The whole point: the surface must not be overwhelmingly pure white.
        let n = img.width as usize * img.height as usize;
        let white = img
            .rgba
            .chunks(4)
            .filter(|p| p[0] == 255 && p[1] == 255 && p[2] == 255)
            .count();
        assert!(
            white * 20 < n,
            "foliage is blank/white: {white}/{n} pure-white texels for {:?}",
            path
        );

        // Exact Pillow-verified pixels for the DXT1 poplar diffuse.
        let is_poplar = path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.eq_ignore_ascii_case("poplartree.dds"))
            .unwrap_or(false);
        if is_poplar {
            let px = |x: u32, y: u32| -> [u8; 4] {
                let o = ((y * img.width + x) * 4) as usize;
                [img.rgba[o], img.rgba[o + 1], img.rgba[o + 2], img.rgba[o + 3]]
            };
            assert_eq!(px(0, 0), [0, 0, 0, 0], "transparent border");
            assert_eq!(px(256, 256), [32, 75, 27, 255], "mid leaf");
            assert_eq!(px(257, 256), [49, 105, 33, 255], "mid leaf");
            assert_eq!(px(300, 300), [68, 132, 54, 255], "leaf");
        }
    }

    /// A DXT5 foliage diffuse (8-bit alpha) also decodes to leaf colour and is
    /// fully consumed by the PNG writer, exercising both paths together.
    #[test]
    fn dxt5_foliage_diffuse_and_png() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game");
        let path = root.join("objects/objects/landscape/vegetation/bushes/azalea/azalea_white_diffuse.dds");
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let img = decode_dds_rgba(&bytes).expect("decode DXT5 foliage");
        assert_eq!(img.rgba.len(), img.width as usize * img.height as usize * 4);
        // Hand-verified with Pillow.
        assert_eq!(&img.rgba[0..4], &[132, 117, 123, 0]);
        assert_eq!(&img.rgba[4..8], &[126, 115, 120, 0]);
        let png = write_dds_as_png(&img).expect("encode foliage png");
        assert_eq!(&png[0..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    }

    #[test]
    fn decode_real_dds_if_available() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game");
        let path = match find_dds(&root, 0) {
            Some(p) => p,
            None => return, // no sample on disk: skip silently
        };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let img = decode_dds_rgba(&bytes)
            .unwrap_or_else(|| panic!("decode failed for real file {:?}", path));
        assert!(img.width > 0 && img.height > 0);
        assert_eq!(
            img.rgba.len(),
            img.width as usize * img.height as usize * 4,
            "dimension/payload mismatch for {:?}",
            path
        );
    }
}
