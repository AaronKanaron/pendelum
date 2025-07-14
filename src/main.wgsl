struct SimulationParams {
    width: u32,
    height: u32,
    time_steps: u32,
    dt: f32,
    // theta1_min: f32,
    // theta1_max: f32,
    // theta2_min: f32,
    // theta2_max: f32,
    center_theta1: f32,
    center_theta2: f32,
    half_span1: f32,
    half_span2: f32,
    gravity: f32,
    length1: f32,
    length2: f32,
    mass1: f32,
    mass2: f32,
    damping: f32,
    _padding: vec3<f32>,
}

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> params: SimulationParams;

fn hash(p: vec2<u32>) -> f32 {
    var h = (p.x * 374761393u + p.y * 668265263u) ^ (p.x * 1274126177u);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h) * 2.3283064e-10;
}

fn simulate_pendulum(theta1_init: f32, theta2_init: f32) -> vec2<f32> {
    var theta1 = theta1_init;
    var theta2 = theta2_init;
    var omega1 = 0.0;
    var omega2 = 0.0;

    let g = params.gravity;
    let l1 = params.length1;
    let l2 = params.length2;
    let m1 = params.mass1;
    let m2 = params.mass2;
    let dt = params.dt;
    let damping = params.damping;

    for (var i = 0u; i < params.time_steps; i++) {
        let cos_diff = cos(theta1 - theta2);
        let sin_diff = sin(theta1 - theta2);

        let denominator = l1 * (2.0 * m1 + m2 - m2 * cos(2.0 * theta1 - 2.0 * theta2));

        let numerator1 = -m2 * g * sin(theta1 - 2.0 * theta2) - 2.0 * sin_diff * m2 * (omega2 * omega2 * l2 + omega1 * omega1 * l1 * cos_diff) - (m1 + m2) * g * sin(theta1);

        let numerator2 = 2.0 * sin_diff * (omega1 * omega1 * l1 * (m1 + m2) + g * (m1 + m2) * cos(theta1) + omega2 * omega2 * l2 * m2 * cos_diff);

        let alpha1 = numerator1 / denominator;
        let alpha2 = numerator2 / (l2 * denominator);

        omega1 += alpha1 * dt;
        omega2 += alpha2 * dt;

        omega1 *= damping;
        omega2 *= damping;

        theta1 += omega1 * dt;
        theta2 += omega2 * dt;
        
        // Check for chaotic divergence
        if abs(omega1) > 50.0 || abs(omega2) > 50.0 {
            break;
        }
    }

    let v1 = omega1 * params.length1;
    let v2 = omega2 * params.length2;

    return vec2<f32>(v1, v2);
}

fn velocity_to_color(v: vec2<f32>) -> vec3<f32> {
    let speed = length(v);
    let normalized_speed = min(speed / 10.0, 1.0);

    if normalized_speed < 0.5 {
        // White to blue transition (slow to medium speeds)
        let t = normalized_speed * 2.0; // Map 0-0.5 to 0-1
        let white = vec3<f32>(1.0, 1.0, 1.0);
        let blue = vec3<f32>(0.0, 0.0, 1.0);
        return mix(white, blue, t);
    } else {
        // Blue to black transition (medium to fast speeds)
        let t = (normalized_speed - 0.5) * 2.0; // Map 0.5-1 to 0-1
        let blue = vec3<f32>(0.0, 0.0, 1.0);
        let black = vec3<f32>(0.0, 0.0, 0.0);
        return mix(blue, black, t);
    }
    // let angle = atan2(v.y, v.x);
    // let hue = (angle + 3.14159) / (2.0 * 3.14159);
    
    // let saturation = normalized_speed;
    // let value = 0.5 + 0.5 * normalized_speed;
    
    // return hsv_to_rgb(vec3<f32>(hue, saturation, value));
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x * 6.0;
    let s = hsv.y;
    let v = hsv.z;

    let c = v * s;
    let x = c * (1.0 - abs((h % 2.0) - 1.0));
    let m = v - c;

    var rgb: vec3<f32>;

    if h < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if h < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if h < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if h < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if h < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    return rgb + vec3<f32>(m, m, m);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let index = y * params.width + x;
    
    // Map pixel coordinates to initial angles
    let nx = (f32(x) / f32(params.width) - 0.5) * 2.0;
    let ny = (f32(y) / f32(params.height) - 0.5) * 2.0;

    let theta1 = params.center_theta1 + nx * params.half_span1;
    let theta2 = params.center_theta2 + ny * params.half_span2;

    // Add small random perturbation to break symmetry
    let random_offset = hash(vec2<u32>(x, y)) * 0.01;
    let theta1_perturbed = theta1 + random_offset;
    let theta2_perturbed = theta2 + random_offset;
    
    // Simulate the double pendulum
    let velocities = simulate_pendulum(theta1_perturbed, theta2_perturbed);
    
    // Convert velocities to color
    let color = velocity_to_color(velocities);
    
    // Pack color into RGBA u32
    let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
    let a = 255u;

    output[index] = (a << 24u) | (b << 16u) | (g << 8u) | r;
}