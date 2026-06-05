//! Phase 3d demo: compare the four pipeline schedules side by side
//! on an 8-GPU / 4-stage training workload.
//!
//! Variants compared:
//! 1. **GPipe** (baseline) — Phase 2c default
//! 2. **1F1B** — same bubble, lower activation memory
//! 3. **Interleaved(factor=2)** — smaller bubble, more comms
//! 4. **GPipe + activation checkpointing** — lower memory, more CPU
//!
//! Run with:
//! ```text
//! cargo run --example pipeline_schedule_comparison -p cjc-nss
//! ```

use cjc_nss::{
    ClusterTopology, GpuTrainingConfig, GpuTrainingSimulator, NssSeed, PipelineSchedule,
    PressureKind,
};

fn main() {
    println!("Phase 3d — pipeline-schedule comparison on 8-GPU / 4-stage training");
    println!("(common config: 4 stages, 8 microbatches/iter, jitter=0.05)\n");

    let base = GpuTrainingConfig {
        n_gpus: 8,
        pipeline_stages: 4,
        microbatches_per_iteration: 8,
        service_mean: 1.0,
        service_jitter: 0.05,
        memory_per_microbatch: 0.04,
        gc_interval: 100_000, // disable GC for the experiment
        fragmentation_growth: 0.0,
        ..GpuTrainingConfig::default()
    };

    let variants: Vec<(&str, GpuTrainingConfig)> = vec![
        ("GPipe (baseline)", base),
        (
            "1F1B",
            GpuTrainingConfig {
                pipeline_schedule: PipelineSchedule::OneForwardOneBackward,
                ..base
            },
        ),
        (
            "Interleaved(2)",
            GpuTrainingConfig {
                pipeline_schedule: PipelineSchedule::Interleaved { factor: 2 },
                ..base
            },
        ),
        (
            "GPipe + activation checkpoint",
            GpuTrainingConfig {
                activation_checkpointing: true,
                checkpoint_memory_factor: 0.4,
                checkpoint_recompute_overhead: 0.33,
                ..base
            },
        ),
    ];

    println!("[design-time metrics] derived from config alone (no simulation):");
    println!(
        "  {:<30} {:>13} {:>12} {:>10}",
        "schedule", "bubble (s0)", "mem mult.", "comm mult."
    );
    for (name, cfg) in &variants {
        let bubble = cfg.bubble_fraction();
        let mem = cfg.stage_activation_memory_multiplier(0);
        let comm = cfg.communication_multiplier();
        println!(
            "  {:<30} {:>13.4} {:>12.4} {:>10.4}",
            name, bubble, mem, comm
        );
    }

    let top = ClusterTopology::complete(8, 16, 0.5).unwrap();
    let seed = NssSeed(2026);
    println!("\n[simulation metrics] after 32 microbatch ticks (sum across all GPUs):");
    println!(
        "  {:<30} {:>10} {:>10} {:>10} {:>10}",
        "schedule", "Σ mem", "Σ sync", "Σ cpu", "Σ net"
    );
    for (name, cfg) in &variants {
        let mut sim = GpuTrainingSimulator::new(*cfg, top.clone(), seed, vec![]).unwrap();
        let traj = sim.run(32).unwrap();
        let mut mem_sum = 0.0f64;
        let mut sync_sum = 0.0f64;
        let mut cpu_sum = 0.0f64;
        let mut net_sum = 0.0f64;
        for ev in traj.iter() {
            for s in ev.state.nodes.values() {
                mem_sum += s.pressures.get(PressureKind::Memory).map(|p| p.saturation()).unwrap_or(0.0);
                sync_sum += s.pressures.get(PressureKind::Sync).map(|p| p.saturation()).unwrap_or(0.0);
                cpu_sum += s.pressures.get(PressureKind::Cpu).map(|p| p.saturation()).unwrap_or(0.0);
                net_sum += s.pressures.get(PressureKind::Network).map(|p| p.saturation()).unwrap_or(0.0);
            }
        }
        println!(
            "  {:<30} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            name, mem_sum, sync_sum, cpu_sum, net_sum
        );
    }

    println!("\n[takeaways]");
    println!("  • 1F1B keeps the same Σ sync as GPipe (same bubble) but cuts Σ memory.");
    println!("  • Interleaved(2) cuts Σ sync (smaller bubble) at the cost of higher Σ net.");
    println!("  • Activation checkpointing cuts Σ memory at the cost of higher Σ cpu.");
    println!("  • All four configs produce deterministic trajectories — same seed, same bytes.");
}
