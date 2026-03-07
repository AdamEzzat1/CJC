//! PNG export — behind the `png` feature flag.
//!
//! Uses the `png` crate for proper PNG output.
//! Reuses the shared rasterizer from `raster.rs`.

#[cfg(feature = "png")]
use crate::raster::rasterize;
#[cfg(feature = "png")]
use crate::scene::Scene;

/// Render a scene to PNG bytes.
///
/// Only available when the `png` feature is enabled.
#[cfg(feature = "png")]
pub fn render_png(scene: &Scene) -> Result<Vec<u8>, String> {
    let buf = rasterize(scene);
    let mut out = Vec::new();

    {
        let mut encoder = png::Encoder::new(&mut out, buf.width, buf.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Default);

        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("PNG header error: {}", e))?;

        writer
            .write_image_data(&buf.data)
            .map_err(|e| format!("PNG write error: {}", e))?;
    }

    Ok(out)
}

/// Stub when PNG feature is not enabled.
#[cfg(not(feature = "png"))]
pub fn render_png(_scene: &crate::scene::Scene) -> Result<Vec<u8>, String> {
    Err("PNG export requires the 'png' feature flag. Compile with --features png".to_string())
}
