use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tpx_core::types::OpenMode;
use tpx_engine::database::TPXDatabase;
use tpx_shard::engine::ShardedEngine;

#[derive(Parser, Debug)]
#[command(
    name = "tpx",
    author,
    version = "1.0.0",
    about = "TPX Storage Engine CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Show archive metadata, version, flags, and chunk statistics
    Info {
        /// Path to .tpx file
        path: PathBuf,
    },
    /// Display hierarchical tree of tensor keys and shapes
    Tree {
        /// Path to .tpx file
        path: PathBuf,
        /// Optional prefix filter
        #[arg(short, long)]
        prefix: Option<String>,
    },
    /// Verify footer xxHash64 and chunk CRC32C checksums
    Verify {
        /// Path to .tpx file
        path: PathBuf,
    },
    /// Compact archive by purging tombstoned and dead chunks
    Compact {
        /// Path to .tpx file
        path: PathBuf,
    },
    /// Display deduplication ratio and savings statistics
    DedupStats {
        /// Path to .tpx file
        path: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { path } => {
            let engine = ShardedEngine::open(&path, OpenMode::ReadOnly)?;
            let file_size = engine.io.len();
            let toc = engine.master_toc.read();
            let live_keys = toc.entries.iter().filter(|e| e.tombstone == 0).count();
            let tombstoned_keys = toc.entries.len() - live_keys;

            println!("======================================================================");
            println!("TPX ARCHIVE METADATA: {}", path.display());
            println!("======================================================================");
            println!(
                "Format Version         : {:#06x}",
                engine.header.format_version
            );
            println!("Flags                  : {:#06x}", engine.header.flags);
            println!("Shard Count            : {}", engine.header.shard_count);
            println!(
                "Total File Size        : {} Bytes ({:.2} MB)",
                file_size,
                file_size as f64 / (1024.0 * 1024.0)
            );
            println!(
                "Live Keys              : {} ({} tombstoned)",
                live_keys, tombstoned_keys
            );
            println!("Total TOC Entries      : {}", toc.entries.len());
            println!("Archive Integrity      : CRC32C / xxHash64 enabled");
            println!("======================================================================");
        }
        Commands::Tree { path, prefix } => {
            let db = TPXDatabase::open(&path, OpenMode::ReadOnly)?;
            let pfx = prefix.unwrap_or_default();
            let tensors = db.get_prefix(&pfx)?;
            println!("Keys under '{}': ({} found)", pfx, tensors.len());
            for (key, tensor) in tensors {
                println!("  ├── {} {} {:?}", key, tensor.dtype.as_str(), tensor.shape);
            }
        }
        Commands::Verify { path } => match ShardedEngine::open(&path, OpenMode::ReadOnly) {
            Ok(engine) => {
                let mut all_valid = true;
                let toc = engine.master_toc.read();
                for entry in &toc.entries {
                    if entry.tombstone == 0 {
                        if let Err(e) = engine.get(&entry.key) {
                            eprintln!("[ERROR] Key '{}' failed validation: {e}", entry.key);
                            all_valid = false;
                        }
                    }
                }
                if all_valid {
                    println!("[OK] Footer checksum and chunk CRC32C checks passed for all keys.");
                } else {
                    eprintln!("[FAIL] Integrity verification encountered corruption.");
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("[FAIL] Could not open archive: {e}");
                std::process::exit(1);
            }
        },
        Commands::Compact { path } => {
            let db = TPXDatabase::open(&path, OpenMode::ReadWrite)?;
            db.compact()?;
            println!("[DONE] Compacted archive at {}", path.display());
        }
        Commands::DedupStats { path } => {
            let engine = ShardedEngine::open(&path, OpenMode::ReadOnly)?;
            let file_size = engine.io.len();
            let toc = engine.master_toc.read();
            let mut logical_bytes: usize = 0;
            for entry in &toc.entries {
                if entry.tombstone == 0 {
                    let elem_count: usize = entry.shape.iter().map(|&x| x as usize).product();
                    logical_bytes += elem_count * entry.dtype_id.element_size();
                }
            }

            let ratio = if file_size > 0 {
                logical_bytes as f64 / file_size as f64
            } else {
                1.0
            };

            println!(
                "Logical uncompressed volume: {} Bytes ({:.2} MB)",
                logical_bytes,
                logical_bytes as f64 / (1024.0 * 1024.0)
            );
            println!(
                "Physical on-disk size      : {} Bytes ({:.2} MB)",
                file_size,
                file_size as f64 / (1024.0 * 1024.0)
            );
            println!("Effective dedup & comp ratio: {:.2}x", ratio);
        }
    }

    Ok(())
}
