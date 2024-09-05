use ark_bls12_381::Fr;
use ark_std::{cfg_iter, UniformRand};
use itertools::{zip_eq, Itertools};
use rand::distributions::{Distribution, Uniform};
use rand::{CryptoRng, RngCore};
use std::fmt;
use std::iter::{zip, Sum};
use std::ops::{AddAssign, Mul};
use std::time::Duration;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

/// Utility struct for displaying human-readable duration of the form "10.5 ms",
/// "350 μs", or "27 ns" from fhe.rs.
pub(crate) struct DisplayDuration(pub Duration);

impl fmt::Display for DisplayDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let duration_ns = self.0.as_nanos();
        if duration_ns < 1_000_u128 {
            write!(f, "{duration_ns} ns")
        } else if duration_ns < 1_000_000_u128 {
            write!(f, "{} μs", (duration_ns + 500) / 1_000)
        } else {
            let duration_ms_times_10 = (duration_ns + 50_000) / (100_000);
            write!(f, "{} ms", (duration_ms_times_10 as f64) / 10.0)
        }
    }
}

#[macro_export]
macro_rules! timeit {
    ($name:expr, $code:expr) => {{
        use crate::utils::DisplayDuration;
        let start = std::time::Instant::now();
        let r = $code;
        println!("⏱  {}: {}", $name, DisplayDuration(start.elapsed()));
        r
    }};
}
#[macro_export]
macro_rules! timeit_return_t {
    ($name:expr, $code:expr) => {{
        use crate::utils::DisplayDuration;
        let start = std::time::Instant::now();
        let r = $code;
        let elapsed = start.elapsed();
        let t = DisplayDuration(elapsed);
        println!("⏱  {}: {}", $name, t);
        (r, elapsed)
    }};
}

#[macro_export]
macro_rules! time {
    ($name:expr, $code:expr) => {{
        let start = std::time::Instant::now();
        let r = $code;
        (r, start.elapsed())
    }};
}

pub(crate) fn linear_combination(a: &Vec<Fr>, x: &Vec<Vec<Fr>>) -> Vec<Fr> {
    assert_eq!(a.len(), x.len());

    zip(a, x).fold(vec![Fr::from(0); x[0].len()], |acc, (ai, xi)| {
        acc.iter()
            .zip(xi)
            .map(|(acc_i, xi_i)| acc_i + &(ai * &xi_i))
            .collect()
    })
}

pub(crate) fn linear_combination_gen<T>(a: &Vec<T>, x: &Vec<Vec<T>>) -> Vec<T>
where
    T: Mul<T, Output = T> + Copy + Default + AddAssign + Sum + std::ops::Add<Output = T>,
{
    assert_eq!(a.len(), x.len());

    zip_eq(a, x).fold(vec![T::default(); x[0].len()], |acc, (ai, xi)| {
        acc.iter()
            .zip_eq(xi)
            .map(|(acc_i, xij)| *acc_i + *ai * *xij)
            .collect()
    })
}

// Generic dot product between two vectors of the same length.
pub(crate) fn dot_product<T, U, O>(v1: &Vec<T>, v2: &Vec<U>) -> O
where
    T: Mul<U, Output = O> + Copy + std::marker::Sync,
    O: Default + AddAssign + Copy + Sum + std::marker::Send,
    U: Copy + std::marker::Sync,
{
    assert_eq!(v1.len(), v2.len());
    let pairs = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a.into(), b.into()))
        .collect::<Vec<_>>();
    cfg_iter!(pairs).map(|(&x, &y)| x.mul(y)).sum()
}

// Generic random vector generator, that supports different size outputs from type range maximum.

pub(crate) fn random_vector_in_range<R>(len: usize, max: usize, rng: &mut R) -> Vec<u64>
where
    R: RngCore + CryptoRng,
{
    // random number from 0 to 10
    let distribution: Uniform<u64> = Uniform::new(0, max as u64);
    (0..len).map(|_| distribution.sample(rng)).collect()
}

pub(crate) fn rand_matrix<R>(len: usize, max: usize, rng: &mut R) -> Vec<Vec<u64>>
where
    R: RngCore + CryptoRng,
{
    (0..len)
        .map(|_| random_vector_in_range::<R>(len, max, rng))
        .collect()
}

pub(crate) fn random_vector<T, U, R>(len: usize, rng: &mut R) -> Vec<U>
where
    T: UniformRand,
    U: From<T>,
    R: RngCore + CryptoRng,
{
    (0..len).map(|_| U::from(T::rand(rng))).collect()
}

// Generic random square matrix generator, that supports different size outputs from type range maximum.
pub(crate) fn random_square_matrix<T, U, R>(len: usize, rng: &mut R) -> Vec<Vec<U>>
where
    T: UniformRand,
    U: From<T>,
    R: RngCore + CryptoRng,
{
    (0..len)
        .map(|_| random_vector::<T, U, _>(len, rng))
        .collect()
}

// Diagonally encode the matrix.
pub(crate) fn diagonal_encode_square_matrix<T>(len: usize, matrix: Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: fmt::Display + Copy,
{
    let mut result = Vec::new();
    for i in 0..len {
        let mut a_i = Vec::new();
        let mut base_row = 0;
        let mut base_col = i;
        for _j in 0..len {
            if base_col == len {
                // base_row += 1;
                base_col = 0;
            }

            a_i.push(matrix[base_row][base_col].clone());

            base_row += 1;
            base_col += 1;
        }
        result.push(a_i);
    }
    result
}

pub(crate) fn flip_matrix(matrix: Vec<Vec<u64>>) -> Vec<Vec<u64>> {
    let n = matrix.len();
    let mut flipped: Vec<Vec<u64>> = vec![vec![0; n]; n];

    for i in 0..n {
        for j in 0..n {
            flipped[i][j] = matrix[j][i].clone();
        }
    }

    flipped
}

pub(crate) fn print_matrix<T>(matrix: &Vec<Vec<T>>)
where
    T: fmt::Display + Copy,
{
    for row in matrix {
        for &val in row {
            print!("{:<4} ", val);
        }
        println!();
    }
}

pub(crate) fn print_vector(vector: Vec<u64>) {
    for val in vector {
        print!("{:<4} ", val);
    }
    println!();
}
