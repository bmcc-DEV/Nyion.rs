// swamp-kernels/src/simd.rs
// Kernels SIMD para operacoes densas f32 com dispatch automatico

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =========================================================================
// DOT PRODUCT: out = Σ a[i] * b[i]
// =========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
        i += 8;
    }
    let mut tail = 0.0f32;
    while i < n {
        tail += *a.add(i) * *b.add(i);
        i += 1;
    }
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    _mm_cvtss_f32(sum128) + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let va = _mm512_loadu_ps(a.add(i));
        let vb = _mm512_loadu_ps(b.add(i));
        sum = _mm512_fmadd_ps(va, vb, sum);
        i += 16;
    }
    // Handle remainder with AVX2
    let mut tail = 0.0f32;
    if i + 8 <= n {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        sum = _mm512_fmadd_ps(_mm512_castps256_ps512(va), _mm512_castps256_ps512(vb), sum);
        i += 8;
    }
    while i < n {
        tail += *a.add(i) * *b.add(i);
        i += 1;
    }
    let lo = _mm512_extractf32x4_ps(sum, 0);
    let hi = _mm512_extractf32x4_ps(sum, 1);
    let hi2 = _mm512_extractf32x4_ps(sum, 2);
    let hi3 = _mm512_extractf32x4_ps(sum, 3);
    let s1 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(hi2, hi3);
    let s = _mm_add_ps(s1, s2);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s) + tail
}

#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { dot_product_avx512(a.as_ptr(), b.as_ptr(), n) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2(a.as_ptr(), b.as_ptr(), n) };
        }
    }
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// =========================================================================
// WEIGHTED SUM: out[i] += score * v[i]
// =========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn weighted_sum_avx2(out: *mut f32, v: *const f32, score: f32, n: usize) {
    let score_bcast = _mm256_set1_ps(score);
    let mut i = 0;
    while i + 8 <= n {
        let vv = _mm256_loadu_ps(v.add(i));
        let ov = _mm256_loadu_ps(out.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_fmadd_ps(score_bcast, vv, ov));
        i += 8;
    }
    while i < n {
        *out.add(i) += score * *v.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn weighted_sum_avx512(out: *mut f32, v: *const f32, score: f32, n: usize) {
    let score_bcast = _mm512_set1_ps(score);
    let mut i = 0;
    while i + 16 <= n {
        let vv = _mm512_loadu_ps(v.add(i));
        let ov = _mm512_loadu_ps(out.add(i));
        _mm512_storeu_ps(out.add(i), _mm512_fmadd_ps(score_bcast, vv, ov));
        i += 16;
    }
    while i < n {
        *out.add(i) += score * *v.add(i);
        i += 1;
    }
}

#[inline]
pub fn weighted_sum(out: &mut [f32], v: &[f32], score: f32) {
    let n = v.len().min(out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { weighted_sum_avx512(out.as_mut_ptr(), v.as_ptr(), score, n) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { weighted_sum_avx2(out.as_mut_ptr(), v.as_ptr(), score, n) };
        }
    }
    for i in 0..n {
        out[i] += score * v[i];
    }
}

// =========================================================================
// RAW POINTER VERSIONS (zero overhead, sem criação de slices)
// =========================================================================

#[inline]
pub unsafe fn dot_product_raw(a: *const f32, b: *const f32, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return dot_product_avx512(a, b, n);
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return dot_product_avx2(a, b, n);
        }
    }
    let mut sum = 0.0f32;
    let mut i = 0;
    while i < n {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

#[inline]
pub unsafe fn weighted_sum_raw(out: *mut f32, v: *const f32, score: f32, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return weighted_sum_avx512(out, v, score, n);
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return weighted_sum_avx2(out, v, score, n);
        }
    }
    let mut i = 0;
    while i < n {
        *out.add(i) += score * *v.add(i);
        i += 1;
    }
}

// =========================================================================
// TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = 1.0*8.0 + 2.0*7.0 + 3.0*6.0 + 4.0*5.0 + 5.0*4.0 + 6.0*3.0 + 7.0*2.0 + 8.0*1.0;
        let diff = (dot_product(&a, &b) - expected).abs();
        assert!(diff < 1e-6);
    }

    #[test]
    fn test_dot_product_odd() {
        let a = vec![1.0f32, 2.0, 3.0, 5.0, 7.0, 9.0];
        let b = vec![2.0f32, 3.0, 5.0, 7.0, 11.0, 13.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let diff = (dot_product(&a, &b) - expected).abs();
        assert!(diff < 1e-6);
    }

    #[test]
    fn test_weighted_sum() {
        let mut out = vec![1.0f32, 2.0, 3.0, 4.0];
        let v = vec![5.0f32, 6.0, 7.0, 8.0];
        let score = 2.0;
        weighted_sum(&mut out, &v, score);
        // out = [1+10, 2+12, 3+14, 4+16] = [11, 14, 17, 20]
        assert!((out[0] - 11.0f32).abs() < 1e-6);
        assert!((out[1] - 14.0f32).abs() < 1e-6);
        assert!((out[2] - 17.0f32).abs() < 1e-6);
        assert!((out[3] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_large() {
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let diff = (dot_product(&a, &b) - expected).abs();
        assert!(diff < 1e-4);
    }
}
