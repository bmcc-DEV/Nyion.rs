use std::time::Instant;
use swamp_kernels::fused_gemv_q6k::fused_gemv_q6k;

const BLOCK_BYTES: usize = 210;
const BLOCK_SIZE: usize = 256;

fn main() {
    let dims = [
        (2048, 5632),
        (2048, 2048),
        (1024, 2816),
    ];

    for &(n_rows, n_cols) in &dims {
        let n_blocks = n_cols / BLOCK_SIZE;
        let raw_size = n_rows * n_blocks * BLOCK_BYTES;
        let mut raw = vec![0u8; raw_size];
        let x: Vec<f32> = (0..n_cols).map(|i| ((i as f32) * 0.01).sin()).collect();

        for row in 0..n_rows {
            for blk in 0..n_blocks {
                let base = (row * n_blocks + blk) * BLOCK_BYTES;
                let block = &mut raw[base..base + BLOCK_BYTES];
                for i in 0..128 {
                    let lo = (i as u8) & 0x0F;
                    let hi = ((i + 128) as u8) & 0x0F;
                    block[i] = lo | (hi << 4);
                }
                for i in 0..64 {
                    block[128 + i] = ((i * 3 + row as usize * 5) as u8) & 0xFF;
                }
                for i in 0..16 {
                    block[192 + i] = ((row * 16 + i + 1) as u8) & 0x7F;
                }
                let d: u16 = 0x3800;
                block[208..210].copy_from_slice(&d.to_le_bytes());
            }
        }

        let mut out = vec![0.0f32; n_rows];
        let trials = 5;

        fused_gemv_q6k(&raw, &x, &mut out, n_rows, n_cols);

        let start = Instant::now();
        for _ in 0..trials {
            fused_gemv_q6k(&raw, &x, &mut out, n_rows, n_cols);
        }
        let elapsed = start.elapsed().as_secs_f64() / trials as f64;

        let bytes_read = raw_size as f64;
        let bw = bytes_read / elapsed / 1e9;

        println!("Q6_K {:>4}x{:<5} {:>8.3}ms {:>8.2} GB/s",
                 n_rows, n_cols, elapsed * 1000.0, bw);
    }
}
