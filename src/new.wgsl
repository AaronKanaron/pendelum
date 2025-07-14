// Replace the simulate_pendulum function with this version:
fn simulate_pendulum(s: ptr<function, ParticleState>, steps: u32) -> bool {
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
        // Check for chaotic divergence
        if abs(omega1) > 50.0 || abs(omega2) > 50.0 {
            // Update state before returning
            (*s).theta1 = theta1;
            (*s).theta2 = theta2;
            (*s).omega1 = omega1;
            (*s).omega2 = omega2;
            return true; // Indicates chaotic divergence
        }

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
    }
    
    // Update state
    (*s).theta1 = theta1;
    (*s).theta2 = theta2;
    (*s).omega1 = omega1;
    (*s).omega2 = omega2;

    return false; // No chaotic divergence
}

// Replace the velocity_to_color function with this version:
fn velocity_to_color(v: vec2<f32>) -> vec3<f32> {
    let speed = length(v);
    let normalized_speed = min(speed / 10.0, 1.0);
    
    // Create white-to-blue-to-black gradient
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
}


@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let index = y * params.width + x;
    
    // Map pixel coordinates to initial angles
    let theta1 = params.theta1_min + (f32(x) / f32(params.width)) * (params.theta1_max - params.theta1_min);
    let theta2 = params.theta2_min + (f32(y) / f32(params.height)) * (params.theta2_max - params.theta2_min);
    
    // Add small random perturbation to break symmetry
    let random_offset = hash(vec2<u32>(x, y)) * 0.01;
    let theta1_perturbed = theta1 + random_offset;
    let theta2_perturbed = theta2 + random_offset;
    
    // Simulate the double pendulum
    let velocities = simulate_pendulum(theta1_perturbed, theta2_perturbed);
    let diverged = simulate_pendulum(&particle_state, params.steps_per_pass);
    
    // Convert velocities to color
    let color = velocity_to_color(velocities);
    if (params.current_pass == params.total_passes - 1u) {
        let v1 = particle_state.omega1 * params.length1;
        let v2 = particle_state.omega2 * params.length2;
        let velocities = vec2<f32>(v1, v2);
        
        var color: vec3<f32>;
        
        // Check if this particle experienced chaotic divergence
        if abs(particle_state.omega1) > 50.0 || abs(particle_state.omega2) > 50.0 {
            color = vec3<f32>(1.0, 0.0, 0.0); // Red for chaotic divergence
        } else {
            color = velocity_to_color(velocities);
        }
        
        let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
        let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
        let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
        let a = 255u;
        output[index] = (a << 24u) | (b << 16u) | (g << 8u) | r;
    }

    
    // Pack color into RGBA u32
    let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
    let a = 255u;
    
    output[index] = (a << 24u) | (b << 16u) | (g << 8u) | r;
}

// Also update the simulate_pendulum call in the main function:
    // Run simulation steps
    