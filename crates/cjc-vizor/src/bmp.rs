//! BMP raster writer — uncompressed BMP format.
//!
//! Zero external dependencies. 54-byte header + raw pixels.
//! Deterministic: same scene → identical bytes.

use crate::raster::{rasterize, PixelBuffer};
use crate::scene::Scene;

/// Render a scene to an uncompressed BMP byte buffer.
pub fn render_bmp(scene: &Scene) -> Vec<u8> {
    let buf = rasterize(scene);
    pixel_buffer_to_bmp(&buf)
}

/// Convert a pixel buffer to BMP format.
pub fn pixel_buffer_to_bmp(buf: &PixelBuffer) -> Vec<u8> {
    let w = buf.width as usize;
    let h = buf.height as usize;

    // BMP rows are padded to 4-byte boundaries.
    let row_bytes = w * 3;
    let row_padded = (row_bytes + 3) & !3;
    let pixel_data_size = row_padded * h;
    let file_size = 54 + pixel_data_size;

    let mut out = Vec::with_capacity(file_size);

    // ── BMP file header (14 bytes) ──
    out.push(b'B');
    out.push(b'M');
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&[0u8; 4]); // reserved
    out.extend_from_slice(&54u32.to_le_bytes()); // pixel data offset

    // ── DIB header (BITMAPINFOHEADER, 40 bytes) ──
    out.extend_from_slice(&40u32.to_le_bytes()); // header size
    out.extend_from_slice(&(w as i32).to_le_bytes()); // width
    out.extend_from_slice(&(h as i32).to_le_bytes()); // height (positive = bottom-up)
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&24u16.to_le_bytes()); // bits per pixel
    out.extend_from_slice(&0u32.to_le_bytes()); // compression (none)
    out.extend_from_slice(&(pixel_data_size as u32).to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes()); // x pixels per meter (~72 DPI)
    out.extend_from_slice(&2835u32.to_le_bytes()); // y pixels per meter
    out.extend_from_slice(&0u32.to_le_bytes()); // colors used
    out.extend_from_slice(&0u32.to_le_bytes()); // important colors

    // ── Pixel data (bottom-up, BGR) ──
    for row in (0..h).rev() {
        for col in 0..w {
            let idx = (row * w + col) * 4;
            out.push(buf.data[idx + 2]); // B
            out.push(buf.data[idx + 1]); // G
            out.push(buf.data[idx]);     // R
        }
        // Padding to 4-byte boundary.
        for _ in 0..(row_padded - row_bytes) {
            out.push(0);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::scene::{Scene, SceneElement};

    #[test]
    fn test_render_bmp_header() {
        let scene = Scene::new(10, 10);
        let bmp = render_bmp(&scene);
        assert_eq!(bmp[0], b'B');
        assert_eq!(bmp[1], b'M');
        // File size at offset 2 (little-endian u32).
        let file_size = u32::from_le_bytes([bmp[2], bmp[3], bmp[4], bmp[5]]);
        assert_eq!(file_size as usize, bmp.len());
    }

    #[test]
    fn test_render_bmp_deterministic() {
        let mut scene = Scene::new(50, 50);
        scene.push(SceneElement::Circle {
            cx: 25.0, cy: 25.0, r: 10.0,
            fill: Color::rgb(255, 0, 0),
            stroke: None,
        });
        let b1 = render_bmp(&scene);
        let b2 = render_bmp(&scene);
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_bmp_dimensions() {
        let scene = Scene::new(100, 80);
        let bmp = render_bmp(&scene);
        // Width at offset 18, height at offset 22 (i32 LE).
        let w = i32::from_le_bytes([bmp[18], bmp[19], bmp[20], bmp[21]]);
        let h = i32::from_le_bytes([bmp[22], bmp[23], bmp[24], bmp[25]]);
        assert_eq!(w, 100);
        assert_eq!(h, 80);
    }
}
