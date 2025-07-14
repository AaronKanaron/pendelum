use pixels::{Error, Pixels, SurfaceTexture};
use std::time::Instant;
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
}

const COMPUTE_SHADER: &str = include_str!("main.wgsl");

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
    let mut last_frame_time = Instant::now();
    let mut frame_rendered = false;

    let mut dragging = false;
    let mut last_cursor: Option<(f64, f64)> = None;

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
                        Some(VirtualKeyCode::R) => {
                            // Trigger re-render
                            frame_rendered = false;
                        }
                        _ => {
                            if renderer.handle_key_input(input) {
                                frame_rendered = false;
                            }
                        }
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                pixels.resize_surface(size.width, size.height).ok();
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::MouseWheel { delta, .. } => {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y as f32,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                    };
                    // zoom factor per tick:
                    let scale = (1.0 + scroll * 0.1).max(0.1);
                    renderer.params.half_span1 /= scale;
                    renderer.params.half_span2 /= scale;
                    // renderer.params.half_span1 *= zoom_amount;
                    // renderer.params.half_span2 *= zoom_amount;
                    renderer.queue.write_buffer(
                        &renderer.params_buffer,
                        0,
                        bytemuck::cast_slice(&[renderer.params]),
                    );
                    frame_rendered = false;
                }
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    dragging = state == ElementState::Pressed;
                    if !dragging {
                        last_cursor = None;
                    }
                }
                WindowEvent::CursorMoved { position, .. } if dragging => {
                    if let Some((lx, ly)) = last_cursor {
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        // Convert pixel delta → angle delta:
                        // note: window size → params.half_span
                        let (w, h) = (renderer.params.width as f32, renderer.params.height as f32);
                        let ang_dx = -(dx as f32 / w) * (2.0 * renderer.params.half_span1);
                        let ang_dy = -(dy as f32 / h) * (2.0 * renderer.params.half_span2);
                        renderer.params.center_theta1 += ang_dx;
                        renderer.params.center_theta2 += ang_dy;
                        renderer.queue.write_buffer(
                            &renderer.params_buffer,
                            0,
                            bytemuck::cast_slice(&[renderer.params]),
                        );
                        frame_rendered = false;
                    }
                    last_cursor = Some((position.x, position.y));
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                let now = Instant::now();
                let delta_time = now.duration_since(last_frame_time).as_secs_f32();
                last_frame_time = now;

                renderer.update(delta_time);

                if !frame_rendered {
                    pollster::block_on(renderer.render(pixels.frame_mut()));
                    frame_rendered = true;
                }

                if pixels.render().is_err() {
                    *control_flow = ControlFlow::Exit;
                }
            }
            _ => {}
        }
    });
}
