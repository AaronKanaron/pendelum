use image::{Rgb, RgbImage};

fn main() {
    let width = 1024;
    let height = 1024;
    let mut img = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let r = generate_color_value(x, y, 0);
            let g = generate_color_value(x + 3, y + 11, 1);
            let b = generate_color_value(x+ 17, y+ 7, 2);
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    img.save("output.png").unwrap();
}

//deterministic function to generate a color value
fn generate_color_value(x: u32, y: u32, channel: u32) -> u8 {
    // Normalize coordinates roughly centered
    let nx = (x as f64 / 1024.0) * 6.28; // 0..2π
    let ny = (y as f64 / 1024.0) * 6.28;

    // Bitwise seed mixing for channel offset
    let seed = ((x ^ y.rotate_left(channel * 7)) & 0xFF) as f64 / 255.0;

    // Create warped orbits with iterative sin/cos chaos
    let mut cx = nx + seed * 3.0;
    let mut cy = ny - seed * 3.0;

    for _ in 0..3 {
        let tx = cx.sin() * cy.cos() + seed * 1.7;
        let ty = cy.sin() * cx.cos() - seed * 1.3;
        cx = tx;
        cy = ty;
    }

    // Combine to a raw value
    let raw = (cx * cy * 10.0 + seed * 5.0).sin();

    // Modulate by channel differently to maximize alien difference
    let val = match channel {
        0 => (raw * 127.0 + 128.0) as u8,
        1 => ((raw + seed).fract() * 255.0) as u8,
        2 => (((raw * seed * 3.14).abs().fract()) * 255.0) as u8,
        _ => 0,
    };

    val
}

// fn generate_color_value(a: u32, b: u32) -> u8 {
//     let x = a.wrapping_mul(37) ^ b.wrapping_mul(73);
//     let y = a.rotate_left(5) ^ b.rotate_right(3);
//     let mix = (x ^ y).wrapping_mul(31).rotate_left((a ^ b) % 32);
//     (mix ^ 0xA5A5A5A5).wrapping_add(a.wrapping_sub(b)) as u8 % 255
// }



// fn clamp_color(value: u32) -> u8 {
//     let min = 0;
//     let max = 255;
//     if value < min {
//         return min as u8;
//     }
//     if value > max {
//         return max as u8;
//     }
//     value as u8
// }
