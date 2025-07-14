struct SimulationParams {
    width: u32,
    height: u32,
    current_pass: u32,
    total_passes: u32,
    steps_per_pass: u32,
    _padding_u: vec3<u32>,  // Aligns u32 section to 32 bytes
    dt: f32,
    theta1_min: f32,
    theta1_max: f32,
    theta2_min: f32,
    theta2_max: f32,
    gravity: f32,
    length1: f32,
    length2: f32,
    mass1: f32,
    mass2: f32,
    damping: f32,
    _padding_f: vec3<f32>,  // Pads f32 section to 64-byte alignment
}

struct ParticleState {
    theta1: f32,
    theta2: f32,
    omega1: f32,
    omega2: f32,
}

@group(0) @binding(0) var<storage, read_write> state: array<ParticleState>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: SimulationParams;

fn hash(p: vec2<u32>) -> f32 {
    var h = (p.x * 374761393u + p.y * 668265263u) ^ (p.x * 1274126177u);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h) * 2.3283064e-10;
}

fn simulate_pendulum(s: ptr<function, ParticleState>, steps: u32) {
    // Dereference state pointers
    var theta1 = (*s).theta1;
    var theta2 = (*s).theta2;
    var omega1 = (*s).omega1;
    var omega2 = (*s).omega2;
    
    let g = params.gravity;
    let l1 = params.length1;
    let l2 = params.length2;
    let m1 = params.mass1;
    let m2 = params.mass2;
    let dt = params.dt;
    let damping = params.damping;
    
    for (var i = 0u; i < steps; i++) {
        // Break early if diverged
        if abs(omega1) > 50.0 || abs(omega2) > 50.0 {
            break;
        }
        
        let cos_diff = cos(theta1 - theta2);
        let sin_diff = sin(theta1 - theta2);
        let denominator = l1 * (2.0 * m1 + m2 - m2 * cos(2.0 * theta1 - 2.0 * theta2));
        
        let numerator1 = -m2 * g * sin(theta1 - 2.0 * theta2) - 2.0 * sin_diff * m2 * 
                        (omega2 * omega2 * l2 + omega1 * omega1 * l1 * cos_diff) - 
                        (m1 + m2) * g * sin(theta1);
        
        let numerator2 = 2.0 * sin_diff * (omega1 * omega1 * l1 * (m1 + m2) + 
                        g * (m1 + m2) * cos(theta1) + omega2 * omega2 * l2 * m2 * cos_diff);
        
        let alpha1 = numerator1 / denominator;
        let alpha2 = numerator2 / (l2 * denominator);
        
        omega1 += alpha1 * dt;
        omega2 += alpha2 * dt;
        
        omega1 *= damping;
        omega2 *= damping;
        
        theta1 += omega1 * dt;
        theta2 += omega2 * dt;
    }
    
    // Update state
    (*s).theta1 = theta1;
    (*s).theta2 = theta2;
    (*s).omega1 = omega1;
    (*s).omega2 = omega2;
}

fn velocity_to_color(v: vec2<f32>) -> vec3<f32> {
    let speed = length(v);
    let normalized_speed = min(speed / 10.0, 1.0);
    let angle = atan2(v.y, v.x);
    let hue = (angle + 3.14159) / (2.0 * 3.14159);
    let saturation = normalized_speed;
    let value = 0.5 + 0.5 * normalized_speed;
    return hsv_to_rgb(vec3<f32>(hue, saturation, value));
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x * 6.0;
    let s = hsv.y;
    let v = hsv.z;
    let c = v * s;
    let x = c * (1.0 - abs((h % 2.0) - 1.0));
    let m = v - c;
    
    var rgb: vec3<f32>;
    if (h < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h < 5.0) {
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
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let index = y * params.width + x;
    var particle_state: ParticleState;
    
    // Initialize or load state
    if (params.current_pass == 0u) {
        let theta1 = params.theta1_min + (f32(x) / f32(params.width)) * 
                    (params.theta1_max - params.theta1_min);
        let theta2 = params.theta2_min + (f32(y) / f32(params.height)) * 
                    (params.theta2_max - params.theta2_min);
        let random_offset = hash(vec2<u32>(x, y)) * 0.01;
        
        particle_state = ParticleState(
            theta1 + random_offset,
            theta2 + random_offset,
            0.0,
            0.0
        );
    } else {
        particle_state = state[index];
    }
    
    // Run simulation steps
    simulate_pendulum(&particle_state, params.steps_per_pass);
    
    // Final pass: compute and store color
    if (params.current_pass == params.total_passes - 1u) {
        let v1 = particle_state.omega1 * params.length1;
        let v2 = particle_state.omega2 * params.length2;
        let color = velocity_to_color(vec2<f32>(v1, v2));
        
        let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
        let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
        let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
        let a = 255u;
        output[index] = (a << 24u) | (b << 16u) | (g << 8u) | r;
    }
    
    // Persist state for next pass
    state[index] = particle_state;
}