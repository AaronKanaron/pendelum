use pixels::{Error, Pixels, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use std::time::Instant;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, SampleRate, Stream, StreamConfig};
use std::sync::{Arc, Mutex};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 800;

// Audio configuration
const SAMPLE_RATE: u32 = 44100;
const LOW_FREQ: f32 = 100.0;   // Minimum frequency (Hz)
const HIGH_FREQ: f32 = 1000.0; // Maximum frequency (Hz)

#[derive(Clone)]
struct DoublePendulum {
    // Angles
    theta1: f32,
    theta2: f32,
    // Angular velocities
    omega1: f32,
    omega2: f32,
    // Physical parameters
    m1: f32,    // mass 1
    m2: f32,    // mass 2
    l1: f32,    // length 1
    l2: f32,    // length 2
    g: f32,     // gravity
}

impl DoublePendulum {
    fn new() -> Self {
        Self {
            theta1: std::f32::consts::PI / 2.0,  // Start at 90 degrees
            theta2: std::f32::consts::PI / 2.0,
            omega1: 0.0,
            omega2: 0.0,
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            g: 9.81,
        }
    }

    fn update(&mut self, dt: f32) {
        // Calculate the derivatives using the double pendulum equations
        let (dtheta1, dtheta2, domega1, domega2) = self.derivatives();
        
        // Update using Runge-Kutta 4th order integration
        let k1 = (dtheta1, dtheta2, domega1, domega2);
        
        let temp_theta1 = self.theta1 + k1.0 * dt * 0.5;
        let temp_theta2 = self.theta2 + k1.1 * dt * 0.5;
        let temp_omega1 = self.omega1 + k1.2 * dt * 0.5;
        let temp_omega2 = self.omega2 + k1.3 * dt * 0.5;
        
        let temp_pendulum = DoublePendulum {
            theta1: temp_theta1,
            theta2: temp_theta2,
            omega1: temp_omega1,
            omega2: temp_omega2,
            ..*self
        };
        let k2 = temp_pendulum.derivatives();
        
        let temp_theta1 = self.theta1 + k2.0 * dt * 0.5;
        let temp_theta2 = self.theta2 + k2.1 * dt * 0.5;
        let temp_omega1 = self.omega1 + k2.2 * dt * 0.5;
        let temp_omega2 = self.omega2 + k2.3 * dt * 0.5;
        
        let temp_pendulum = DoublePendulum {
            theta1: temp_theta1,
            theta2: temp_theta2,
            omega1: temp_omega1,
            omega2: temp_omega2,
            ..*self
        };
        let k3 = temp_pendulum.derivatives();
        
        let temp_theta1 = self.theta1 + k3.0 * dt;
        let temp_theta2 = self.theta2 + k3.1 * dt;
        let temp_omega1 = self.omega1 + k3.2 * dt;
        let temp_omega2 = self.omega2 + k3.3 * dt;
        
        let temp_pendulum = DoublePendulum {
            theta1: temp_theta1,
            theta2: temp_theta2,
            omega1: temp_omega1,
            omega2: temp_omega2,
            ..*self
        };
        let k4 = temp_pendulum.derivatives();
        
        // Apply the weighted average
        self.theta1 += (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) * dt / 6.0;
        self.theta2 += (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) * dt / 6.0;
        self.omega1 += (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) * dt / 6.0;
        self.omega2 += (k1.3 + 2.0 * k2.3 + 2.0 * k3.3 + k4.3) * dt / 6.0;
    }

    fn derivatives(&self) -> (f32, f32, f32, f32) {
        let dtheta1 = self.omega1;
        let dtheta2 = self.omega2;
        
        // Simplified double pendulum equations
        let delta = self.theta2 - self.theta1;
        let den1 = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * delta.cos() * delta.cos();
        let den2 = (self.l2 / self.l1) * den1;
        
        let num1 = -self.m2 * self.l1 * self.omega1 * self.omega1 * delta.sin() * delta.cos()
                 + self.m2 * self.g * self.theta2.sin() * delta.cos()
                 + self.m2 * self.l2 * self.omega2 * self.omega2 * delta.sin()
                 - (self.m1 + self.m2) * self.g * self.theta1.sin();
        
        let num2 = -self.m2 * self.l2 * self.omega2 * self.omega2 * delta.sin() * delta.cos()
                 + (self.m1 + self.m2) * self.g * self.theta1.sin() * delta.cos()
                 + (self.m1 + self.m2) * self.l1 * self.omega1 * self.omega1 * delta.sin()
                 - (self.m1 + self.m2) * self.g * self.theta2.sin();
        
        let domega1 = num1 / den1;
        let domega2 = num2 / den2;
        
        (dtheta1, dtheta2, domega1, domega2)
    }

    fn get_linear_velocities(&self) -> (f32, f32) {
        // Convert angular velocities to linear velocities of the masses
        let v1 = self.omega1 * self.l1;
        
        // For the second mass, we need to consider both pendulum segments
        let v2x = self.l1 * self.omega1 * (-self.theta1).sin() + self.l2 * self.omega2 * (-self.theta2).sin();
        let v2y = self.l1 * self.omega1 * self.theta1.cos() + self.l2 * self.omega2 * self.theta2.cos();
        let v2 = (v2x * v2x + v2y * v2y).sqrt();
        
        (v1, v2)
    }
}

#[derive(Clone)]
struct AudioState {
    left_frequency: f32,
    right_frequency: f32,
    volume: f32,
    phase_left: f32,
    phase_right: f32,
}

impl AudioState {
    fn new() -> Self {
        Self {
            left_frequency: 440.0,
            right_frequency: 440.0,
            volume: 0.1,
            phase_left: 0.0,
            phase_right: 0.0,
        }
    }
}

struct AudioEngine {
    _stream: Stream,
    audio_state: Arc<Mutex<AudioState>>,
}

impl AudioEngine {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = host.default_output_device()
            .ok_or("No output device available")?;
        
        let config = device.default_output_config()?;
        let sample_rate = config.sample_rate().0;
        
        let audio_state = Arc::new(Mutex::new(AudioState::new()));
        let audio_state_clone = audio_state.clone();
        
        let stream = match config.sample_format() {
            SampleFormat::F32 => Self::run_stream::<f32>(&device, &config.into(), audio_state_clone)?,
            SampleFormat::I16 => Self::run_stream::<i16>(&device, &config.into(), audio_state_clone)?,
            SampleFormat::U16 => Self::run_stream::<u16>(&device, &config.into(), audio_state_clone)?,
            _ => return Err("Unsupported sample format".into()),
        };
        
        stream.play()?;
        
        Ok(Self {
            _stream: stream,
            audio_state,
        })
    }
    
    fn run_stream<T>(
        device: &Device,
        config: &StreamConfig,
        audio_state: Arc<Mutex<AudioState>>,
    ) -> Result<Stream, Box<dyn std::error::Error>>
    where
        T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
    {
        let sample_rate = config.sample_rate.0 as f32;
        let channels = config.channels as usize;
        
        let stream = device.build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                if let Ok(mut state) = audio_state.try_lock() {
                    for frames in data.chunks_mut(channels) {
                        // Generate left channel sample
                        let left_sample = (state.phase_left * 2.0 * std::f32::consts::PI).sin() * state.volume;
                        state.phase_left += state.left_frequency / sample_rate;
                        if state.phase_left >= 1.0 {
                            state.phase_left -= 1.0;
                        }
                        
                        // Generate right channel sample
                        let right_sample = (state.phase_right * 2.0 * std::f32::consts::PI).sin() * state.volume;
                        state.phase_right += state.right_frequency / sample_rate;
                        if state.phase_right >= 1.0 {
                            state.phase_right -= 1.0;
                        }
                        
                        // Fill the frame with stereo samples
                        if frames.len() >= 2 {
                            frames[0] = T::from_sample(left_sample);
                            frames[1] = T::from_sample(right_sample);
                        } else if frames.len() == 1 {
                            // Mono output - mix both channels
                            frames[0] = T::from_sample((left_sample + right_sample) * 0.5);
                        }
                    }
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?;
        
        Ok(stream)
    }
    
    fn update_frequencies(&self, v1: f32, v2: f32, max_velocity: f32) {
        if let Ok(mut state) = self.audio_state.try_lock() {
            // Map velocities to frequency range
            let normalized_v1 = (v1 / max_velocity).clamp(-1.0, 1.0);
            let normalized_v2 = (v2 / max_velocity).clamp(-1.0, 1.0);
            
            // Convert to positive frequency range
            state.left_frequency = Self::velocity_to_frequency(normalized_v1);
            state.right_frequency = Self::velocity_to_frequency(normalized_v2);
            
            // Calculate volume based on velocity magnitude
            let velocity_magnitude = (v1 * v1 + v2 * v2).sqrt();
            let normalized_magnitude = (velocity_magnitude / (max_velocity * std::f32::consts::SQRT_2)).clamp(0.0, 1.0);
            state.volume = normalized_magnitude * 0.2; // Scale volume
        }
    }
    
    fn velocity_to_frequency(normalized_velocity: f32) -> f32 {
        // Map [-1, 1] to [LOW_FREQ, HIGH_FREQ]
        let t = (normalized_velocity + 1.0) * 0.5; // Map to [0, 1]
        LOW_FREQ + t * (HIGH_FREQ - LOW_FREQ)
    }
    
    fn set_volume(&self, volume: f32) {
        if let Ok(mut state) = self.audio_state.try_lock() {
            state.volume = volume.clamp(0.0, 1.0);
        }
    }
}

struct VelocityVisualizer {
    pendulum: DoublePendulum,
    velocity_map: Vec<f32>,
    max_velocity: f32,
    time: f32,
    audio_engine: AudioEngine,
}

impl VelocityVisualizer {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            pendulum: DoublePendulum::new(),
            velocity_map: vec![0.0; (WIDTH * HEIGHT) as usize],
            max_velocity: 10.0,
            time: 0.0,
            audio_engine: AudioEngine::new()?,
        })
    }

    fn update(&mut self, delta_time: f32) {
        self.time += delta_time;
        
        // Update pendulum with smaller timesteps for stability
        let substeps = 4;
        let dt = delta_time / substeps as f32;
        
        for _ in 0..substeps {
            self.pendulum.update(dt);
        }
        
        // Get linear velocities
        let (v1, v2) = self.pendulum.get_linear_velocities();
        
        // Update audio frequencies
        self.audio_engine.update_frequencies(v1, v2, self.max_velocity);
        
        // Map velocities to pixel coordinates for visualization
        let x = self.velocity_to_pixel(v1, WIDTH);
        let y = self.velocity_to_pixel(v2, HEIGHT);
        
        // Update the velocity map
        if x < WIDTH && y < HEIGHT {
            let index = (y * WIDTH + x) as usize;
            if index < self.velocity_map.len() {
                self.velocity_map[index] += 1.0;
            }
        }
    }

    fn velocity_to_pixel(&self, velocity: f32, dimension: u32) -> u32 {
        let normalized = (velocity / self.max_velocity + 1.0) * 0.5; // Map [-max, max] to [0, 1]
        let pixel = (normalized * dimension as f32) as u32;
        pixel.min(dimension - 1)
    }

    fn generate_visualization(&mut self, frame: &mut [u8]) {
        // Apply decay to create trailing effect
        for value in &mut self.velocity_map {
            *value *= 0.995;
        }
        
        // Find the maximum value for normalization
        let max_intensity = self.velocity_map.iter().fold(0.0f32, |a, &b| a.max(b));
        let max_intensity = max_intensity.max(1.0f32);
        
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let intensity = self.velocity_map[i] / max_intensity;
            
            // Create a colorful visualization
            let r = (intensity * 255.0) as u8;
            let g = ((intensity * 2.0).min(1.0) * 255.0) as u8;
            let b = ((intensity * 4.0).min(1.0) * 255.0) as u8;
            
            pixel[0] = r; // R
            pixel[1] = g; // G
            pixel[2] = b; // B
            pixel[3] = 255; // A
        }
        
        // Draw coordinate axes
        self.draw_axes(frame);
        
        // Draw current velocity position
        let (v1, v2) = self.pendulum.get_linear_velocities();
        self.draw_current_position(frame, v1, v2);
    }

    fn draw_axes(&self, frame: &mut [u8]) {
        let center_x = WIDTH / 2;
        let center_y = HEIGHT / 2;
        
        // Draw vertical axis (v2 = 0)
        for y in 0..HEIGHT {
            let index = (y * WIDTH + center_x) as usize * 4;
            if index + 3 < frame.len() {
                frame[index] = 128;     // R
                frame[index + 1] = 128; // G
                frame[index + 2] = 128; // B
            }
        }
        
        // Draw horizontal axis (v1 = 0)
        for x in 0..WIDTH {
            let index = (center_y * WIDTH + x) as usize * 4;
            if index + 3 < frame.len() {
                frame[index] = 128;     // R
                frame[index + 1] = 128; // G
                frame[index + 2] = 128; // B
            }
        }
        
        // Draw center point
        let center_index = (center_y * WIDTH + center_x) as usize * 4;
        if center_index + 3 < frame.len() {
            frame[center_index] = 255;     // R
            frame[center_index + 1] = 255; // G
            frame[center_index + 2] = 255; // B
        }
    }
    
    fn draw_current_position(&self, frame: &mut [u8], v1: f32, v2: f32) {
        let x = self.velocity_to_pixel(v1, WIDTH);
        let y = self.velocity_to_pixel(v2, HEIGHT);
        
        // Draw a cross at the current position
        for dx in -2..=2 {
            for dy in -2..=2 {
                if dx == 0 || dy == 0 {
                    let px = (x as i32 + dx) as u32;
                    let py = (y as i32 + dy) as u32;
                    
                    if px < WIDTH && py < HEIGHT {
                        let index = (py * WIDTH + px) as usize * 4;
                        if index + 3 < frame.len() {
                            frame[index] = 255;     // R
                            frame[index + 1] = 255; // G
                            frame[index + 2] = 0;   // B (yellow)
                        }
                    }
                }
            }
        }
    }
    
    fn reset(&mut self) {
        self.pendulum = DoublePendulum::new();
        self.velocity_map.fill(0.0);
    }
    
    fn reset_random(&mut self) {
        self.pendulum = DoublePendulum::new();
        self.pendulum.theta1 = rand::random::<f32>() * std::f32::consts::PI;
        self.pendulum.theta2 = rand::random::<f32>() * std::f32::consts::PI;
        self.pendulum.omega1 = (rand::random::<f32>() - 0.5) * 2.0;
        self.pendulum.omega2 = (rand::random::<f32>() - 0.5) * 2.0;
        self.velocity_map.fill(0.0);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let event_loop = EventLoop::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Double Pendulum Velocity Visualization with Audio")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let window_size = window.inner_size();
    let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
    let mut pixels = Pixels::new(WIDTH, HEIGHT, surface_texture)?;
    
    let mut visualizer = VelocityVisualizer::new()?;
    let mut last_frame_time = Instant::now();
    let mut audio_enabled = true;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                if input.state == winit::event::ElementState::Pressed {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::Escape) => {
                            *control_flow = ControlFlow::Exit;
                        }
                        Some(VirtualKeyCode::Space) => {
                            visualizer.reset();
                        }
                        Some(VirtualKeyCode::R) => {
                            visualizer.reset_random();
                        }
                        Some(VirtualKeyCode::M) => {
                            // Toggle audio (mute/unmute)
                            audio_enabled = !audio_enabled;
                            visualizer.audio_engine.set_volume(if audio_enabled { 0.2 } else { 0.0 });
                        }
                        Some(VirtualKeyCode::Up) => {
                            // Increase volume
                            if audio_enabled {
                                visualizer.audio_engine.set_volume(0.3);
                            }
                        }
                        Some(VirtualKeyCode::Down) => {
                            // Decrease volume
                            if audio_enabled {
                                visualizer.audio_engine.set_volume(0.1);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                pixels.resize_surface(size.width, size.height).ok();
            }
            Event::MainEventsCleared => {
                let now = Instant::now();
                let delta_time = now.duration_since(last_frame_time).as_secs_f32();
                last_frame_time = now;
                
                visualizer.update(delta_time);
                visualizer.generate_visualization(pixels.frame_mut());
                
                if let Err(err) = pixels.render() {
                    eprintln!("pixels.render() failed: {}", err);
                    *control_flow = ControlFlow::Exit;
                }
            }
            _ => {}
        }
    });
}