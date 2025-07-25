use atomic_float::AtomicF32;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use pixels::{Error, Pixels, SurfaceTexture};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};
use wgpu::util::DeviceExt;
use winit::{
    dpi::LogicalSize,
    event::{
        ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
        WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 800;

struct DoublePendulumRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    time: f32,
    params: SimulationParams,
    sensitivity: f32,

    /// Trajectory points in normalized coordinates [0..1]
    trajectory: Option<Vec<[f32; 2]>>, // [x1, y1, x2, y2] per frame
    traj_velocities: Option<Vec<[f32; 2]>>,
    /// Animation progress index
    traj_index: usize,
    /// Whether we animate over time (vs. static draw)
    animate_traj: bool,

    click_angles: Option<(f32, f32)>,

    screen_w: f32,
    screen_h: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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
    _padding: [f32; 6],
}

impl DoublePendulumRenderer {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Double Pendulum Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (WIDTH * HEIGHT * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (WIDTH * HEIGHT * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let params = SimulationParams {
            width: WIDTH,
            height: HEIGHT,
            time_steps: 1000,
            dt: 0.01,
            // theta1_min: -std::f32::consts::PI,
            // theta1_max: std::f32::consts::PI,
            // theta2_min: -std::f32::consts::PI,
            // theta2_max: std::f32::consts::PI,
            center_theta1: 0.0,
            center_theta2: 0.0,
            half_span1: std::f32::consts::PI,
            half_span2: std::f32::consts::PI,

            gravity: 9.81,
            length1: 1.0,
            length2: 1.0,
            mass1: 1.0,
            mass2: 1.0,
            damping: 0.999,
            _padding: [0.0; 6],
        };
        // let current_params = params.clone(); // spara en kopia

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });
        Self {
            device,
            queue,
            compute_pipeline,
            bind_group,
            output_buffer,
            staging_buffer,
            params_buffer,
            time: 0.0,
            params,
            sensitivity: 0.1, // Default sensitivity
            trajectory: None,
            traj_velocities: None,
            traj_index: 0,
            animate_traj: false,
            click_angles: None,
            screen_w: WIDTH as f32,
            screen_h: HEIGHT as f32,
        }
    }

    fn update(&mut self, delta_time: f32) {
        self.time += delta_time;
    }

    async fn render(&mut self, frame: &mut [u8]) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups((WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            (WIDTH * HEIGHT * 4) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver).unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        frame.copy_from_slice(&data);
        drop(data);
        self.staging_buffer.unmap();
    }

    fn handle_key_input(&mut self, input: KeyboardInput) -> bool {
        if let Some(key) = input.virtual_keycode {
            let change = if input.state == ElementState::Pressed {
                0.1
            } else {
                0.0
            };
            let mut updated = false;

            match key {
                //SET SENSITIVITY OF KEYBOARD INPUT BY RANGING FROM 0-9
                VirtualKeyCode::Key0 => {
                    self.sensitivity = 0.0001;
                    println!("Sensitivity set to 0.0001");
                }
                VirtualKeyCode::Key1 => {
                    self.sensitivity = 0.001;
                    println!("Sensitivity set to 0.001");
                }
                VirtualKeyCode::Key2 => {
                    self.sensitivity = 0.01;
                    println!("Sensitivity set to 0.01");
                }
                VirtualKeyCode::Key3 => {
                    self.sensitivity = 0.1;
                    println!("Sensitivity set to 0.1");
                }
                VirtualKeyCode::Key4 => {
                    self.sensitivity = 0.25;
                    println!("Sensitivity set to 0.25");
                }
                VirtualKeyCode::Key5 => {
                    self.sensitivity = 0.5;
                    println!("Sensitivity set to 0.5");
                }
                VirtualKeyCode::Key6 => {
                    self.sensitivity = 1.0;
                    println!("Sensitivity set to 1.0");
                }
                VirtualKeyCode::Key7 => {
                    self.sensitivity = 2.0;
                    println!("Sensitivity set to 2.0");
                }
                VirtualKeyCode::Key8 => {
                    self.sensitivity = 5.0;
                    println!("Sensitivity set to 5.0");
                }
                VirtualKeyCode::Key9 => {
                    self.sensitivity = 10.0;
                    println!("Sensitivity set to 10.0");
                }

                VirtualKeyCode::G => {
                    self.params.gravity += change * self.sensitivity;
                    println!("Gravity: {}", self.params.gravity);
                    updated = true;
                }
                VirtualKeyCode::H => {
                    self.params.gravity -= change * self.sensitivity;
                    updated = true;
                }
                VirtualKeyCode::L => {
                    self.params.length1 += change * self.sensitivity;
                    updated = true;
                }
                VirtualKeyCode::K => {
                    self.params.length1 -= change * self.sensitivity;
                    updated = true;
                }
                VirtualKeyCode::J => {
                    self.params.length2 += change * self.sensitivity;
                    updated = true;
                }
                VirtualKeyCode::U => {
                    self.params.length2 -= change * self.sensitivity;
                    updated = true;
                }
                VirtualKeyCode::M => {
                    self.params.mass1 += change * self.sensitivity;
                    updated = true;
                }
                VirtualKeyCode::N => {
                    self.params.mass1 -= change * self.sensitivity;
                    updated = true;
                }
                VirtualKeyCode::I => {
                    self.params.mass2 += change * self.sensitivity;
                    updated = true;
                }
                VirtualKeyCode::O => {
                    self.params.mass2 -= change * self.sensitivity;
                    updated = true;
                }
                VirtualKeyCode::T => {
                    self.params.time_steps += (100.0 * self.sensitivity) as u32;
                    updated = true;
                }
                VirtualKeyCode::Y => {
                    self.params.time_steps = self
                        .params
                        .time_steps
                        .saturating_sub((100.0 * self.sensitivity) as u32);
                    updated = true;
                }
                VirtualKeyCode::D => {
                    self.params.dt += 0.001 * self.sensitivity;
                    updated = true;
                }
                VirtualKeyCode::S => {
                    self.params.dt = (self.params.dt - 0.001).max(0.001) * self.sensitivity;
                    updated = true;
                }
                _ => {}
            }

            if updated {
                self.queue.write_buffer(
                    &self.params_buffer,
                    0,
                    bytemuck::cast_slice(&[self.params]),
                );
                self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bind Group"),
                    layout: &self.compute_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.params_buffer.as_entire_binding(),
                        },
                    ],
                });
                println!(
                    "Updated params: gravity={}, length1={}, length2={}, mass1={}, mass2={}, time_steps={}, dt={}",
                    self.params.gravity, self.params.length1, self.params.length2,
                    self.params.mass1, self.params.mass2, self.params.time_steps, self.params.dt
                );
            }

            return updated;
        }

        false
    }

    /// Simulate one pendulum for `time_steps` and return the final angular velocities.
    fn simulate_single(
        &self,
        theta1: f32,
        theta2: f32,
    ) -> (Vec<[f32; 2]>, Vec<[f32; 2]>, f32, f32) {
        let mut traj = Vec::with_capacity(self.params.time_steps as usize);

        // state
        let mut th1 = theta1;
        let mut th2 = theta2;
        let mut om1 = 0.0;
        let mut om2 = 0.0;

        // aliases
        let dt = self.params.dt;
        let g = self.params.gravity;
        let l1 = self.params.length1;
        let l2 = self.params.length2;
        let m1 = self.params.mass1;
        let m2 = self.params.mass2;
        let damping = self.params.damping;

        let mut vel_hist = Vec::with_capacity(self.params.time_steps as usize);
        for _ in 0..self.params.time_steps {
            // compute accelerations
            let c = (th1 - th2).cos();
            let s = (th1 - th2).sin();
            let denom = l1 * (2.0 * m1 + m2 - m2 * (2.0 * th1 - 2.0 * th2).cos());

            let num1 = -m2 * g * (th1 - 2.0 * th2).sin()
                - 2.0 * s * m2 * (om2 * om2 * l2 + om1 * om1 * l1 * c)
                - (m1 + m2) * g * th1.sin();

            let num2 = 2.0
                * s
                * (om1 * om1 * l1 * (m1 + m2)
                    + g * (m1 + m2) * th1.cos()
                    + om2 * om2 * l2 * m2 * c);

            let a1 = num1 / denom;
            let a2 = num2 / (l2 * denom);

            // integrate & damp
            om1 = (om1 + a1 * dt) * damping;
            om2 = (om2 + a2 * dt) * damping;
            th1 += om1 * dt;
            th2 += om2 * dt;

            // record angles for your fractal trace
            traj.push([th1, th2]);
            vel_hist.push([om1, om2]);
        }

        // return the trajectory plus the final omegas
        (traj, vel_hist, om1, om2)
    }

    /// Call on every frame (after computing the fractal), so the trace
    /// will re‐project if you pan/zoom.
    fn overlay_trajectory(&mut self, frame: &mut [u8]) {
        let (traj, (_, _)) =
            if let (Some(traj), Some(angles)) = (&self.trajectory, &self.click_angles) {
                (traj, *angles)
            } else {
                return;
            };

        // number of points to draw
        let pts = if self.animate_traj {
            self.traj_index = (self.traj_index + 4).min(traj.len());
            self.traj_index
        } else {
            traj.len()
        };

        let buf_w = self.params.width as i32;
        let buf_h = self.params.height as i32;

        // helper to draw a 3×3 red block in the pixel buffer
        fn draw_block(buf: &mut [u8], x: i32, y: i32, W: i32, H: i32) {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let px = x + dx;
                    let py = y + dy;
                    if (0..W).contains(&px) && (0..H).contains(&py) {
                        // buffer is exactly W*H*4 bytes long
                        let idx = ((py as usize) * W as usize + px as usize) * 4;

                        buf[idx + 0] = 255; // R
                        buf[idx + 1] = 0; // G
                        buf[idx + 2] = 0; // B
                        buf[idx + 3] = 255; // A
                    }
                }
            }
        }

        let mut prev: Option<(i32, i32)> = None;

        // plot each (θ₁,θ₂) in the same way your fractal uses angles→pixels
        for &[th1, th2] in traj.iter().take(pts) {
            // let nx = (th1 - self.params.center_theta1) / self.params.half_span1;
            // let ny = (th2 - self.params.center_theta2) / self.params.half_span2;
            // let sx = ((nx * 0.5 + 0.5) * w) as i32;
            // let sy = ((ny * 0.5 + 0.5) * h) as i32;
            let (sx, sy) = self.world_to_screen(th1, th2);

            draw_block(frame, sx, sy, buf_w, buf_h);

            if let Some((px, py)) = prev {
                let dx = sx - px;
                let dy = sy - py;
                let steps = dx.abs().max(dy.abs()).max(1);
                for i in 1..=steps {
                    let ix = px + dx * i / steps;
                    let iy = py + dy * i / steps;
                    draw_block(frame, ix, iy, buf_w, buf_h);
                }
            }
            prev = Some((sx, sy));
        }
    }

    fn screen_to_world(&self, x: f32, y: f32) -> (f32, f32) {
        // use real on‐screen dims, not params.width/height
        let nx = (x / self.screen_w - 0.5) * 2.0;
        let ny = (y / self.screen_h - 0.5) * 2.0;
        let t1 = self.params.center_theta1 + nx * self.params.half_span1;
        let t2 = self.params.center_theta2 + ny * self.params.half_span2;
        (t1, t2)
    }

    fn world_to_screen(&self, theta1: f32, theta2: f32) -> (i32, i32) {
        let nx = (theta1 - self.params.center_theta1) / self.params.half_span1;
        let ny = (theta2 - self.params.center_theta2) / self.params.half_span2;
        // map into the *fixed* buffer dimensions (WIDTH×HEIGHT)
        let sx = ((nx * 0.5 + 0.5) * (self.params.width as f32)) as i32;
        let sy = ((ny * 0.5 + 0.5) * (self.params.height as f32)) as i32;
        (sx, sy)
    }
}

struct AudioState {
    freq_left: Arc<AtomicF32>,
    freq_right: Arc<AtomicF32>,
    vel_left: Arc<AtomicF32>, // last angular velocity
    vel_right: Arc<AtomicF32>,
    mute: Arc<AtomicBool>,
    _stream: cpal::Stream,
}

const COMPUTE_SHADER: &str = include_str!("main.wgsl");

fn setup_audio(low_hz: f32, high_hz: f32) -> AudioState {
    let host = cpal::default_host();
    let device = host.default_output_device().expect("no output device");
    let config = device.default_output_config().unwrap();
    let sample_rate = config.sample_rate().0 as f32;
    let channels = config.channels() as usize;

    // Shared frequency targets:
    let freq_left = Arc::new(AtomicF32::new((low_hz + high_hz) / 2.0));
    let freq_right = Arc::new(AtomicF32::new((low_hz + high_hz) / 2.0));
    let vel_left = Arc::new(AtomicF32::new(0.0));
    let vel_right = Arc::new(AtomicF32::new(0.0));
    let mute = Arc::new(AtomicBool::new(false));

    // Clone for callback:
    let fl = freq_left.clone();
    let fr = freq_right.clone();
    let vl = vel_left.clone();
    let vr = vel_right.clone();
    let m = mute.clone();

    // Phase accumulators:
    let mut phase_l = 0.0_f32;
    let mut phase_r = 0.0_f32;

    // Build the stream:
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &config.into(),
            move |data: &mut [f32], _| {
                for frame in data.chunks_mut(channels) {
                    if m.load(Ordering::Relaxed) {
                        for sample in frame.iter_mut() {
                            *sample = 0.0;
                        }
                        continue;
                    }
                    // Read target freqs
                    let target_l = fl.load(Ordering::Relaxed);
                    let target_r = fr.load(Ordering::Relaxed);

                    let vel_l = vl.load(Ordering::Relaxed).abs();
                    let vel_r = vr.load(Ordering::Relaxed).abs();

                    // Increment phases
                    // phase_l += target_l * 2.0 * std::f32::consts::PI / sample_rate;
                    // phase_r += target_r * 2.0 * std::f32::consts::PI / sample_rate;
                    fn harmonic_mix(phase: f32, vel: f32) -> f32 {
                        // define thresholds
                        let t_low = 2.0;
                        let t_high = 8.0;
                        // interpolation factor in [0,1]
                        let u = ((vel - t_low) / (t_high - t_low)).clamp(0.0, 1.0);
                        let mut sum = 0.0;
                        let mut norm = 0.0;
                        for h in 1..=5 {
                            // weight harmonics: lower velocity favors fundamental/harmonic 2, high adds 3-5
                            let w = if h <= 2 {
                                1.0 - u * 0.5 // reduce only slightly at high vel
                            } else {
                                u * (h as f32 / 5.0)
                            };
                            sum += w * (phase * h as f32).sin();
                            norm += w;
                        }
                        sum / norm
                    }

                    phase_l += target_l * 2.0 * std::f32::consts::PI / sample_rate;
                    phase_r += target_r * 2.0 * std::f32::consts::PI / sample_rate;

                    // Wrap back into [0..2π]
                    if phase_l > std::f32::consts::TAU {
                        phase_l -= std::f32::consts::TAU;
                    }
                    if phase_r > std::f32::consts::TAU {
                        phase_r -= std::f32::consts::TAU;
                    }

                    let sample_l = harmonic_mix(phase_l, vel_l);
                    let sample_r = harmonic_mix(phase_r, vel_r);

                    // Wrap
                    // if phase_l > std::f32::consts::TAU {
                    //     phase_l -= std::f32::consts::TAU
                    // }
                    // if phase_r > std::f32::consts::TAU {
                    //     phase_r -= std::f32::consts::TAU
                    // }

                    // let sample_l = phase_l.sin();
                    // let sample_r = phase_r.sin();

                    // Stereo
                    frame[0] = sample_l;
                    if channels > 1 {
                        frame[1] = sample_r;
                    }
                }
            },
            move |err| eprintln!("Audio error: {}", err),
            None,
        ),
        _ => panic!("Unsupported sample format"),
    }
    .unwrap();

    stream.play().unwrap();

    AudioState {
        freq_left,
        freq_right,
        vel_left,
        vel_right,
        mute,
        _stream: stream,
    }
}

fn map_vel_to_freq(v: f32, v_max: f32, low: f32, high: f32) -> f32 {
    let v_clamped = v.max(-v_max).min(v_max);
    let t = (v_clamped + v_max) / (2.0 * v_max);
    low + t * (high - low)
}

fn main() -> Result<(), Error> {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Double Pendulum Fractal - GPU Accelerated")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let window_size = window.inner_size();
    let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
    let mut pixels = Pixels::new(WIDTH, HEIGHT, surface_texture)?;

    let mut renderer = pollster::block_on(DoublePendulumRenderer::new());
    let audio = setup_audio(100.0, 1000.0);
    renderer.screen_w = window_size.width as f32;
    renderer.screen_h = window_size.height as f32;
    let mut last_frame_time = Instant::now();
    let mut frame_rendered = false;

    // let mut dragging = false;
    // let mut last_cursor: Option<(f64, f64)> = None;

    // let mut dragging = false;
    // let mut last_cursor: Option<(f64, f64)> = None;
    // renderer.click_pos = None;
    let mut dragging = false;
    let mut last_cursor: Option<(f64, f64)> = None;
    let mut click_pos: Option<(f64, f64)> = None;
    let mut pan_prev: Option<(f64, f64)> = None;
    let mut shift_down = false;

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
                if let Some(key) = input.virtual_keycode {
                    if key == VirtualKeyCode::LShift || key == VirtualKeyCode::RShift {
                        shift_down = input.state == ElementState::Pressed;
                        println!("Shift key state: {:?}", input.state);
                    }
                }

                // 2) Then early‐exit on Escape/R (or pass to your existing handler):
                if input.state == ElementState::Pressed {
                    if let Some(VirtualKeyCode::Escape) = input.virtual_keycode {
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                    if let Some(VirtualKeyCode::X) = input.virtual_keycode {
                        let new = !audio.mute.load(Ordering::Relaxed);
                        audio.mute.store(new, Ordering::Relaxed);
                        println!("Audio {}", if new { "muted" } else { "unmuted" });
                        frame_rendered = false;
                        return;
                    }

                    // 3) Finally, non‐Shift & non‐special keys for pendulum params:
                    if renderer.handle_key_input(input) {
                        frame_rendered = false;
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                pixels.resize_surface(size.width, size.height).ok();
                renderer.screen_w = size.width as f32;
                renderer.screen_h = size.height as f32;
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::MouseWheel { delta, .. } => {
                    // compute scroll direction & strength
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y as f32,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                    };
                    let zoom_factor = (1.0 + scroll * 0.1).max(0.1);

                    // **ONLY** shrink/grow your world‐span
                    renderer.params.half_span1 /= zoom_factor;
                    renderer.params.half_span2 /= zoom_factor;

                    // push the new spans up to the GPU
                    renderer.queue.write_buffer(
                        &renderer.params_buffer,
                        0,
                        bytemuck::cast_slice(&[renderer.params]),
                    );
                    frame_rendered = false;
                }

                WindowEvent::CursorMoved { position, .. } => {
                    last_cursor = Some((position.x, position.y));
                    if let Some((ex, ey)) = last_cursor {
                        // If left mouse is down and Shift is NOT pressed -> update preview
                        if dragging && !shift_down {
                            // Map to normalized coords
                            let (theta1, theta2) = renderer.screen_to_world(ex as f32, ey as f32);
                            // let (trajectory, final_om1, final_om2) =
                            //     renderer.simulate_single(theta1, theta2);

                            let (traj, vel_hist, f1, f2) = renderer.simulate_single(theta1, theta2);
                            renderer.traj_velocities = Some(vel_hist);
                            (traj, f1, f2);

                            // let f1 = map_vel_to_freq(final_om1, 10.0, 100.0, 1000.0);
                            // let f2 = map_vel_to_freq(final_om2, 10.0, 100.0, 1000.0);
                            // audio.freq_left.store(f1, Ordering::Relaxed);
                            // audio.freq_right.store(f2, Ordering::Relaxed);
                            let f1 = map_vel_to_freq(f1, 10.0, 100.0, 1000.0);
                            let f2 = map_vel_to_freq(f2, 10.0, 100.0, 1000.0);
                            // store base freq and last velocity
                            audio.freq_left.store(f1, Ordering::Relaxed);
                            audio.freq_right.store(f2, Ordering::Relaxed);
                            audio.vel_left.store(f1, Ordering::Relaxed);
                            audio.vel_right.store(f2, Ordering::Relaxed);

                            renderer.click_angles = Some((theta1, theta2));
                            // renderer.trajectory = Some(traj);
                            renderer.traj_index = 0;
                            renderer.animate_traj = true;
                            frame_rendered = false;
                        }

                        // If Shift is down and left-button held -> pan
                        if dragging && shift_down {
                            if let Some((px0, py0)) = pan_prev {
                                // world before vs. after
                                let (t10, t20) = renderer.screen_to_world(px0 as f32, py0 as f32);
                                let (t1n, t2n) = renderer.screen_to_world(ex as f32, ey as f32);

                                renderer.params.center_theta1 -= t1n - t10;
                                renderer.params.center_theta2 -= t2n - t20;

                                renderer.queue.write_buffer(
                                    &renderer.params_buffer,
                                    0,
                                    bytemuck::cast_slice(&[renderer.params]),
                                );
                                frame_rendered = false;
                            }
                            pan_prev = last_cursor;
                        }
                    }
                }

                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    match state {
                        ElementState::Pressed => {
                            dragging = true;
                            click_pos = last_cursor;
                            pan_prev = last_cursor;
                        }
                        ElementState::Released => {
                            dragging = false;
                            pan_prev = None;
                            // On release, only finalize click if it was a true click (not drag)
                            if let (Some((sx, sy)), Some((ex, ey))) = (click_pos, last_cursor) {
                                let dist2 = (sx - ex).powi(2) + (sy - ey).powi(2);
                                if dist2 < 25.0 {
                                    // compute and store final angles & sim
                                    // let nx = (ex as f32 / WIDTH as f32 - 0.5) * 2.0;
                                    // let ny = (ey as f32 / HEIGHT as f32 - 0.5) * 2.0;
                                    // let theta1 = renderer.params.center_theta1
                                    //     + nx * renderer.params.half_span1;
                                    // let theta2 = renderer.params.center_theta2
                                    //     + ny * renderer.params.half_span2;
                                    let (theta1, theta2) =
                                        renderer.screen_to_world(ex as f32, ey as f32);
                                    renderer.click_angles = Some((theta1, theta2));
                                    // let (trajectory, final_om1, final_om2) =
                                    //     renderer.simulate_single(theta1, theta2);

                                    let (traj, vel_hist, f1, f2) =
                                        renderer.simulate_single(theta1, theta2);
                                    renderer.trajectory = Some(traj);
                                    renderer.traj_velocities = Some(vel_hist);
                                    // (trajectory, final_om1, final_om2) =
                                    //     (renderer.trajectory.as_ref().unwrap().clone(), f1, f2);

                                    // renderer.trajectory = Some(traj);
                                    renderer.traj_index = 0;
                                    renderer.animate_traj = true;
                                    frame_rendered = false;

                                    let f1 = map_vel_to_freq(f1, 10.0, 100.0, 1000.0);
                                    let f2 = map_vel_to_freq(f2, 10.0, 100.0, 1000.0);
                                    // store base freq and last velocity
                                    audio.freq_left.store(f1, Ordering::Relaxed);
                                    audio.freq_right.store(f2, Ordering::Relaxed);
                                    audio.vel_left.store(f1, Ordering::Relaxed);
                                    audio.vel_right.store(f2, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }

                _ => {}
            },

            Event::MainEventsCleared => {
                let now = Instant::now();
                let delta_time = now.duration_since(last_frame_time).as_secs_f32();
                last_frame_time = now;

                renderer.update(delta_time);

                let mut need_redraw = !frame_rendered;
                if renderer.animate_traj {
                    // as long as we haven't drawn the entire trajectory, keep animating
                    need_redraw = true;
                }
                if need_redraw {
                    pollster::block_on(renderer.render(pixels.frame_mut()));
                    renderer.overlay_trajectory(pixels.frame_mut());
                    frame_rendered = true;

                    // if we're animating, also drive audio from the current step
                    if renderer.animate_traj {
                        if let (Some(vels), Some(idx)) =
                            (&renderer.traj_velocities, Some(renderer.traj_index.saturating_sub(1)))
                        {
                            let [om1, om2] = vels[idx.min(vels.len().saturating_sub(1))];
                            let f1 = map_vel_to_freq(om1, 10.0, 100.0, 1000.0);
                            let f2 = map_vel_to_freq(om2, 10.0, 100.0, 1000.0);
                            audio.freq_left.store(f1, Ordering::Relaxed);
                            audio.freq_right.store(f2, Ordering::Relaxed);
                            audio.vel_left.store(om1, Ordering::Relaxed);
                            audio.vel_right.store(om2, Ordering::Relaxed);
                        }

                        // keep animating until we hit the end
                        if let Some(traj) = &renderer.trajectory {
                            if renderer.traj_index < traj.len() {
                                frame_rendered = false;
                            } else {
                                renderer.animate_traj = false;
                            }
                        }
                    }
                }

                if pixels.render().is_err() {
                    *control_flow = ControlFlow::Exit;
                }
            }
            _ => {}
        }
    });
}
