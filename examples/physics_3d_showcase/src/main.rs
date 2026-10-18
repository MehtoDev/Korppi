//! Main executable for the 3D Physics Showcase.

use korppi_engine::{run_windowed, PresentOptions};
use korppi_window::config::WindowConfig;
use physics_3d_showcase_lib::app::Physics3dShowcaseApp;
use std::env;
use tracing_subscriber::EnvFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Log at `info` unless RUST_LOG says otherwise; winit/calloop trace output stays hidden
    // (ISSUE-001).
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = env::args().collect();
    let is_headless = args.iter().any(|arg| arg == "--headless");

    if is_headless {
        let mut ticks: u32 = 120;
        let mut target_gallery: Option<usize> = None;

        let mut i = 1;
        while i < args.len() {
            if args[i] == "--headless" && i + 1 < args.len() {
                if let Ok(t) = args[i + 1].parse::<u32>() {
                    ticks = t;
                    i += 1;
                }
            } else if args[i] == "--gallery" && i + 1 < args.len() {
                if let Ok(g) = args[i + 1].parse::<usize>() {
                    target_gallery = Some(g);
                    i += 1;
                }
            }
            i += 1;
        }

        println!("=== Korppi 3D Physics Showcase: Headless Mode ===");
        println!("Simulating {ticks} ticks per gallery...");

        let mut app = Physics3dShowcaseApp::new();
        let gallery_indices: Vec<usize> = match target_gallery {
            Some(idx) => vec![idx],
            None => (0..app.gallery_count()).collect(),
        };

        for &gallery_idx in &gallery_indices {
            app.load_gallery(gallery_idx);
            println!(
                "\n--- Simulating Gallery {}: {} ---",
                gallery_idx + 1,
                app.current_gallery_name()
            );
            println!("Description: {}", app.current_gallery_description());

            let dt = 1.0 / 60.0;
            for step in 0..ticks {
                app.step_simulation(dt);

                // Sanity check body stability
                for id in 0..app.world().body_count() as u32 {
                    if let Some(body) = app.world().body(id) {
                        assert!(
                            body.position.x.is_finite()
                                && body.position.y.is_finite()
                                && body.position.z.is_finite(),
                            "Non-finite position at tick {step} for body {id}: {:?}",
                            body.position
                        );
                        assert!(
                            body.linear_velocity.x.is_finite()
                                && body.linear_velocity.y.is_finite()
                                && body.linear_velocity.z.is_finite(),
                            "Non-finite velocity at tick {step} for body {id}: {:?}",
                            body.linear_velocity
                        );
                        assert!(
                            body.angular_velocity.x.is_finite()
                                && body.angular_velocity.y.is_finite()
                                && body.angular_velocity.z.is_finite(),
                            "Non-finite angular velocity at tick {step} for body {id}: {:?}",
                            body.angular_velocity
                        );
                    }
                }
            }

            let sleeping_count = (0..app.world().body_count() as u32)
                .filter(|&id| app.world().body(id).is_some_and(|b| b.is_sleeping))
                .count();

            println!(
                "Gallery {} Completed Successfully: {} bodies ({} sleeping), {} joints, {} contacts",
                gallery_idx + 1,
                app.world().body_count(),
                sleeping_count,
                app.world().joint_count(),
                app.world().contact_count()
            );
        }

        println!("\nAll Headless Physics 3D Galleries Passed With Zero Numerical Anomalies!");
        return Ok(());
    }

    // Interactive windowed execution: `--backend vulkan|opengl`, `--frames N`,
    // `--present-mode fifo|mailbox|immediate|fifo-relaxed`.
    let options = PresentOptions::from_args(&args)?;
    let backend = options.backend;
    let app = Physics3dShowcaseApp::new();
    let config = WindowConfig::default()
        .with_title("Korppi 3D Physics Showcase")
        .with_dimensions(1280, 720);

    let frames = run_windowed(app, config, options)?;
    println!("Presented {frames} frames with {backend:?}");
    Ok(())
}
