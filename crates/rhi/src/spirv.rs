//! SPIR-V module framing shared by the backends (finding RVK-18).
//!
//! Backends hand SPIR-V to drivers (Vulkan) or a translator (OpenGL), and drivers may crash on
//! malformed modules. [`ShaderDescriptor::spirv`] therefore checks the framing of every module
//! before a backend sees it:
//! - the byte order, from the magic number;
//! - the header;
//! - that every instruction lies within the module;
//! - that an entry point exists for the stage and name.
//!
//! It does not validate the module's semantics; tools such as `spirv-val` do that offline.

use crate::descriptor::{ShaderDescriptor, ShaderSource, ShaderStage};
use crate::error::RhiError;
use std::borrow::Cow;

/// SPIR-V's magic number, the first word of every module.
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Words in the module header: magic, version, generator, bound, schema.
const HEADER_WORDS: usize = 5;

/// `OpEntryPoint`.
const OP_ENTRY_POINT: u32 = 15;

fn invalid(message: String) -> RhiError {
    RhiError::ShaderCompilationFailed(format!("invalid SPIR-V module: {message}"))
}

/// `words` in native byte order. A module stored as bytes may be in either byte order and its
/// magic number tells which (SPIR-V §2.3); words built from bytes of the other order are swapped.
///
/// # Errors
/// `RhiError::ShaderCompilationFailed` if the module is shorter than its header or does not
/// start with the magic number in either byte order.
pub fn spirv_native_words(words: &[u32]) -> Result<Cow<'_, [u32]>, RhiError> {
    if words.len() < HEADER_WORDS {
        return Err(invalid(format!(
            "{} words, shorter than the {HEADER_WORDS}-word header",
            words.len()
        )));
    }
    match words[0] {
        SPIRV_MAGIC => Ok(Cow::Borrowed(words)),
        magic if magic.swap_bytes() == SPIRV_MAGIC => {
            Ok(Cow::Owned(words.iter().map(|w| w.swap_bytes()).collect()))
        }
        magic => Err(invalid(format!(
            "starts with {magic:#010x}, not the magic number {SPIRV_MAGIC:#010x}"
        ))),
    }
}

/// The words of a module given as bytes, in native byte order (see [`spirv_native_words`]).
///
/// # Errors
/// `RhiError::ShaderCompilationFailed` if the length is not a whole number of words, or as
/// [`spirv_native_words`].
pub fn spirv_words_from_bytes(bytes: &[u8]) -> Result<Vec<u32>, RhiError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(invalid(format!(
            "{} bytes, not a whole number of 32-bit words",
            bytes.len()
        )));
    }
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_ne_bytes(*w))
        .collect();
    Ok(match spirv_native_words(&words)? {
        Cow::Borrowed(_) => words,
        Cow::Owned(swapped) => swapped,
    })
}

/// The SPIR-V execution model of `stage`.
fn execution_model(stage: ShaderStage) -> u32 {
    match stage {
        ShaderStage::Vertex => 0,
        ShaderStage::Fragment => 4,
        ShaderStage::Compute => 5,
    }
}

/// Decodes a SPIR-V literal string (UTF-8, nul-terminated, packed little-end first into words).
fn literal_string(words: &[u32]) -> Option<String> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let end = bytes.iter().position(|&b| b == 0)?;
    String::from_utf8(bytes[..end].to_vec()).ok()
}

/// Checks the instruction stream of native-order `words` and that it declares `entry_point` for
/// `stage`.
///
/// # Errors
/// `RhiError::ShaderCompilationFailed` for an instruction with a zero word count or running
/// past the end of the module, or a missing entry point.
pub fn validate_spirv_structure(
    words: &[u32],
    stage: ShaderStage,
    entry_point: &str,
) -> Result<(), RhiError> {
    let model = execution_model(stage);
    let mut found = false;
    let mut at = HEADER_WORDS;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let opcode = words[at] & 0xFFFF;
        if count == 0 || at + count > words.len() {
            return Err(invalid(format!(
                "instruction at word {at} (opcode {opcode}, {count} words) runs past the end \
                 of the {}-word module",
                words.len()
            )));
        }
        // OpEntryPoint: execution model, function id, name, interface ids.
        if opcode == OP_ENTRY_POINT
            && count >= 4
            && words[at + 1] == model
            && literal_string(&words[at + 3..at + count]).as_deref() == Some(entry_point)
        {
            found = true;
        }
        at += count;
    }
    if found {
        Ok(())
    } else {
        Err(RhiError::ShaderCompilationFailed(format!(
            "the SPIR-V module has no {stage:?} entry point named `{entry_point}`"
        )))
    }
}

impl ShaderDescriptor<'_> {
    /// The SPIR-V module in native byte order with its framing checked (byte order, header,
    /// instruction bounds, the entry point for `stage`), or `None` for GLSL.
    ///
    /// # Errors
    /// `RhiError::ShaderCompilationFailed` as [`spirv_words_from_bytes`],
    /// [`spirv_native_words`] and [`validate_spirv_structure`] describe.
    pub fn spirv(&self) -> Result<Option<Cow<'_, [u32]>>, RhiError> {
        let words = match &self.source {
            ShaderSource::Spirv(words) => spirv_native_words(words)?,
            ShaderSource::SpirvBytes(bytes) => Cow::Owned(spirv_words_from_bytes(bytes)?),
            ShaderSource::Glsl(_) => return Ok(None),
        };
        validate_spirv_structure(&words, self.stage, self.entry_point)?;
        Ok(Some(words))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal module: header, `OpEntryPoint GLCompute %1 "main"`.
    fn module() -> Vec<u32> {
        let name = u32::from_le_bytes(*b"main");
        vec![
            SPIRV_MAGIC,
            0x0001_0000,
            0,
            2,
            0,
            (5 << 16) | OP_ENTRY_POINT,
            5,
            1,
            name,
            0,
        ]
    }

    #[test]
    fn byte_order_follows_the_magic_number() {
        let native = module();
        let swapped: Vec<u32> = native.iter().map(|w| w.swap_bytes()).collect();
        assert_eq!(spirv_native_words(&swapped).unwrap(), native);
        assert!(matches!(
            spirv_native_words(&native).unwrap(),
            Cow::Borrowed(_)
        ));
        let bytes: Vec<u8> = swapped.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(spirv_words_from_bytes(&bytes).unwrap(), native);
    }

    #[test]
    fn framing_errors_are_reported() {
        let native = module();
        assert!(validate_spirv_structure(&native, ShaderStage::Compute, "main").is_ok());
        assert!(validate_spirv_structure(&native, ShaderStage::Vertex, "main").is_err());
        assert!(validate_spirv_structure(&native, ShaderStage::Compute, "mai").is_err());
        let truncated = &native[..native.len() - 1];
        assert!(validate_spirv_structure(truncated, ShaderStage::Compute, "main").is_err());
        let mut zero_count = native.clone();
        zero_count.push(0);
        assert!(validate_spirv_structure(&zero_count, ShaderStage::Compute, "main").is_err());
        assert!(spirv_words_from_bytes(&[3, 2, 0x23, 7, 1]).is_err());
        assert!(spirv_native_words(&native[..4]).is_err());
        let mut no_magic = native;
        no_magic[0] = 0xDEAD_BEEF;
        assert!(spirv_native_words(&no_magic).is_err());
    }
}
