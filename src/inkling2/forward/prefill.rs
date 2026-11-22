use super::*;
use crate::backend::{Backend, Buf};

/// x holds Q, relative codes, sink and bias; y holds the result. Stage as many KV heads as fit at a time.
pub(super) fn scores((be, x, y, stage): (&dyn Backend, &Buf, &Buf, &Buf), q: &[f32], r: &[f32], sink: &[f32], bias: &[f32], slot: &Slot,
    nt: usize, nh: usize, hd: usize, nr: usize, ext: usize, window: usize) -> Result<Vec<f32>> {
    ensure!(be.name() == "cuda", "non-CUDA backend");
    let nkv = slot.k.len();
    ensure!(nt > 1 && nh > 0 && nh <= 65535 && nkv > 0 && nh % nkv == 0 && (1..=128).contains(&hd) && window > 0, "unsupported shape");
    ensure!(q.len() <= y.len() / 4 && [q.len(), r.len(), sink.len(), bias.len()].iter().sum::<usize>() <= x.len() / 4, "activation scratch too small");
    ensure!(nt.checked_mul(nh) == Some(sink.len()) && sink.len().checked_mul(hd) == Some(q.len()) && sink.len().checked_mul(nr) == Some(r.len()) && nr.checked_mul(ext) == Some(bias.len()), "invalid projection shape");
    let len = slot.k[0].len();
    ensure!(slot.v.len() == nkv && slot.k.iter().chain(&slot.v).all(|h| h.len() == len) && len % hd == 0 && len / hd >= nt, "invalid KV shape");
    let rows = len / hd;
    let window = window.min(rows);
    // Rows before the oldest query's window aren't needed even when the host hasn't trimmed them yet.
    let first = (rows - nt + 1).saturating_sub(window);
    let rows = rows - first;
    let stride = rows * hd;
    let batch = nkv.min(stage.len() / 4 / stride);
    ensure!(batch > 0, "KV scratch too small for one head");
    let mut off = 0;
    let mut put = |a: &[f32]| -> Result<u64> {
        let addr = x.addr() + 4 * off as u64;
        be.put(a, x, off)?;
        off += a.len();
        Ok(addr)
    };
    let (qptr, rptr, sptr, bptr) = (put(q)?, put(r)?, put(sink)?, put(bias)?);
    for kh in (0..nkv).step_by(batch) {
        let heads = batch.min(nkv - kh);
        for (j, h) in slot.k[kh..kh+heads].iter().chain(&slot.v[kh..kh+heads]).enumerate() {
            be.put(h[first * hd..].reinterpret_cast(), stage, j * stride)?;
        }
        be.ink_prefill(qptr, rptr, sptr, bptr, stage.addr(), stage.addr() + (2 * heads * stride) as u64, y.addr(), nt, nh, nh / nkv, kh, heads, rows, window, hd, nr, ext)?;
    }
    be.get(y, 0, q.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA and NVRTC"]
    fn cuda_prefill_reference() -> Result<()> {
        let be = crate::backend::open(Some("cuda"), 0)?;
        let (nt, nh, nkv, hd, nr, ext, rows) = (7, 6, 3, 64, 5, 13, 43);
        let vals = |n, seed| (0..n).map(|i: usize| ((i * 73 + i / 19 + seed) % 127) as f32 / 32. - 2.).collect::<Vec<_>>();
        let q = vals(nt * nh * hd, 33);
        let r = vals(nt * nh * nr, 17);
        let sink = vals(nt * nh, 8);
        let bias = vals(nr * ext, 28);
        let half = |s| (0..nkv).map(|h| vals(rows * hd, s + h * 47).into_iter().map(f16::from_f32).collect()).collect();
        let slot = Slot { k: half(4), v: half(77), ..Slot::default() };
        let x = be.zeros::<f32>(q.len() + r.len() + sink.len() + bias.len())?;
        let y = be.zeros::<f32>(q.len())?;
        let stage = be.alloc(4 * rows * hd)?; // full attention must use three head batches
        for window in [1, 21, rows] {
            let got = scores((&*be, &x, &y, &stage), &q, &r, &sink, &bias, &slot, nt, nh, hd, nr, ext, window)?;
            let mut max_err = 0.0f64;
            for t in 0..nt { for h in 0..nh {
                let i = t * nh + h;
                let kh = h / (nh / nkv);
                let end = rows - nt + t + 1;
                let mut z = 1.;
                let mut acc = vec![0.; hd];
                for j in end.saturating_sub(window)..end {
                    let d = end - 1 - j;
                    let mut sc = (0..hd).map(|c| q[i*hd+c] as f64 * slot.k[kh][j*hd+c].to_f64()).sum::<f64>() / (hd as f64).sqrt() - sink[i] as f64;
                    if d < ext { for c in 0..nr { sc += r[i*nr+c] as f64 * bias[(c+1)*ext-1-d] as f64; } }
                    let w = sc.exp(); z += w;
                    for c in 0..hd { acc[c] += w * slot.v[kh][j*hd+c].to_f64(); }
                }
                for c in 0..hd {
                    let err = (got[i*hd+c] as f64 - acc[c] / z).abs();
                    assert!(err < 2e-5, "window={window} t={t} head={h} channel={c} err={err}");
                    max_err = max_err.max(err);
                }
            } }
            eprintln!("CUDA prefill window={window} max_abs_error={max_err:e}");
        }
        let small = be.alloc(4 * rows * hd - 1)?;
        let err = scores((&*be, &x, &y, &small), &q, &r, &sink, &bias, &slot, nt, nh, hd, nr, ext, rows).unwrap_err();
        assert_eq!(err.to_string(), "KV scratch too small for one head");
        Ok(())
    }
}
