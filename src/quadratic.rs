use crate::utils::{
    diagonal_encode_square_matrix, dot_product, linear_combination_gen, random_square_matrix,
    random_vector, random_vector_in_range,
};
use crate::{timeit, timeit_return_t};
use ark_bls12_381::{Bls12_381, G2Projective};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::VariableBaseMSM;
use ark_ec::{AffineRepr, CurveGroup, Group};
use ark_ff::{UniformRand};
use ark_std::cfg_iter;
use ark_std::rand::distributions::Distribution;
use fhe::bfv::{
    BfvParametersBuilder, Ciphertext, Encoding, EvaluationKey, EvaluationKeyBuilder,
    Plaintext, RelinearizationKey, SecretKey,
};
use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};
use itertools::Itertools;
use rand::distributions::Standard;
use rand::{thread_rng};
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use std::any::TypeId;
use std::error::Error;
use std::iter::Sum;
use std::ops::{Mul, Sub};

type G1 = <Bls12_381 as Pairing>::G1;
type G2 = <Bls12_381 as Pairing>::G2;
type Scalar = <Bls12_381 as Pairing>::ScalarField;
type GT = PairingOutput<Bls12_381>;

/// Homomorphically evaluate vector matrix vector product xAy.
/// Assumes that A is diagonally encoded.
pub fn fhe_quadratic(
    n: usize,
    x_ct: &Ciphertext,
    mut y_ct: Ciphertext,
    A: &Vec<Plaintext>,
    rk: &RelinearizationKey,
    ek: &Vec<EvaluationKey>,
) -> Result<Ciphertext, Box<dyn Error>> {
    // Compute a * y as result = sum(a_i * rot_i(x))
    let a_0_pt = A.get(0).unwrap();
    let mut a_y = &y_ct * a_0_pt;
    let ek_0 = ek.get(0).unwrap();
    for i in 1..n {
        let a_i_pt = A.get(i).unwrap();
        y_ct = ek_0.rotates_columns_by(&y_ct, 1)?;

        a_y = &a_y + &(&y_ct * a_i_pt);
    }

    // Dot product step 1: Hadamard product (x * a_y).
    let mut result = x_ct * &a_y;

    // Relinarize result.
    rk.relinearizes(&mut result)?;

    // Dot product step 2: Rotate and sum.

    // Make sure we have a power of 2 as length.
    assert_eq!(n.count_ones(), 1);

    // Calculate log2 of length.
    let log2_length: usize = n.trailing_zeros() as usize;
    assert_eq!(ek.len(), log2_length as usize);

    // Do first rotation & sum.
    let mut rot: Ciphertext;
    // Do the rest of the rotations & sums.
    for i in 0..log2_length {
        // Create the evaluation key.
        let ek_i: &EvaluationKey = ek.get(i).unwrap();
        rot = ek_i.rotates_columns_by(&result, 1 << i)?;
        result = &result + &rot;
    }

    Ok(result)
}

/// Homomorphically evaluate inner product xy with rotate and sum.
pub fn fhe_inner_product(
    n: usize,
    x_ct: &Ciphertext,
    y_ct: Ciphertext,
    rk: &RelinearizationKey,
    ek: &Vec<EvaluationKey>,
) -> Result<Ciphertext, Box<dyn Error>> {
    let mut result = x_ct * &y_ct;

    // Relinarize result.
    rk.relinearizes(&mut result)?;

    // Dot product step 2: Rotate and sum.

    // Make sure we have a power of 2 as length.
    assert_eq!(n.count_ones(), 1);

    // Calculate log2 of length.
    let log2_length: usize = n.trailing_zeros() as usize;
    assert_eq!(ek.len(), log2_length as usize);

    // Do first rotation & sum.
    let mut rot: Ciphertext;
    // Do the rest of the rotations & sums.
    for i in 0..log2_length {
        // Create the evaluation key.
        let ek_i: &EvaluationKey = ek.get(i).unwrap();
        rot = ek_i.rotates_columns_by(&result, 1 << i)?;
        result = &result + &rot;
    }

    Ok(result)
}

// Compute xAy when x in G1, y in G2, A in Fr.
pub fn vec_matrix_vec_pairing(x: &Vec<G1>, y: &Vec<G2>, A: &Vec<Vec<Scalar>>) -> GT {
    let Ay = cfg_iter!(A)
        .map(|row| {
            <G2Projective as VariableBaseMSM>::msm(
                &y.iter().map(|y_i| y_i.into_affine()).collect::<Vec<_>>(),
                &row,
            )
            .unwrap()
        })
        .collect::<Vec<G2>>();
    let pairings = cfg_iter!(x)
        .zip_eq(Ay)
        .map(|(x_i, Ay_i)| Bls12_381::pairing(x_i, Ay_i))
        .collect::<Vec<_>>();
    cfg_iter!(pairings).sum()
}

/// Compute xAy over integers.
pub fn local_vec_matrix_vec(x: Vec<u64>, y: Vec<u64>, A: Vec<Vec<u64>>) -> u64 {
    let Ay: Vec<u64> = cfg_iter!(A).map(|row| dot_product(&y, row)).collect();
    dot_product(&x, &Ay)
}

/// Batch FHE linear function evaluations.
/// l: Number of inputs (x_i, y_i)
/// n: Input length
/// d: Degree of ciphertext
/// p: Plaintext modulus
/// q_sizes: Moduli sizes
pub fn vQHE_batch<T: 'static>(
    l: usize,
    n: usize,
    d: usize,
    p: u64,
    q_sizes: [usize; 5],
) -> Result<(), Box<dyn Error>>
where
    u64: From<T>,
    Standard: Distribution<T>,
{
    println!(
        "vQHE_batch: l: {:?}, n: {:?}, d: {:?}, p: {:?}, q_sizes: {:?}",
        l, n, d, p, q_sizes
    );

    assert_eq!(n * 2, d);

    // Setup.
    // Create random number generator.
    let mut rng = thread_rng();

    // Sample common reference string.
    let h = random_vector::<G1, G1, _>(n, &mut rng);
    let r = random_vector::<G2, G2, _>(n, &mut rng);

    // Server: Offline.
    // Generate quadratic function Q_A.
    let Q_A = random_square_matrix::<T, u64, _>(n, &mut rng);
    let Q_A_diag = diagonal_encode_square_matrix(n, Q_A.clone());

    let A = Q_A
        .iter()
        .map(|row| {
            row.iter()
                .map(|c| Scalar::from(c.clone()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<Vec<_>>>();

    // Pairing commitment to A.
    let mut cm = timeit!("create cm", vec_matrix_vec_pairing(&h, &r, &A));

    // Client: Upload.
    // Initialize BFV instance.
    let params = BfvParametersBuilder::new()
        .set_degree(d)
        .set_plaintext_modulus(p)
        .set_moduli_sizes(&q_sizes)
        .build_arc()
        .unwrap();

    // Create private key.
    let sk = SecretKey::random(&params, &mut rng);

    // Create relinearization key.
    let rk = RelinearizationKey::new(&sk, &mut rng).unwrap();

    // Create the evaluation key.
    let log2_length: usize = n.trailing_zeros() as usize;
    let ek = (0..log2_length)
        .map(|i| {
            EvaluationKeyBuilder::new_leveled(&sk, 0, 0)
                .unwrap()
                .enable_column_rotation(1 << i)
                .unwrap()
                .build(&mut rng)
                .unwrap()
        })
        .collect::<Vec<EvaluationKey>>();

    // Sample randomness.
    let alpha = random_vector::<Scalar, Scalar, _>(l + 1, &mut rng);
    let beta = random_vector::<Scalar, Scalar, _>(l + 1, &mut rng);

    // Trim 1 bit each, since fhe.rs limits the plaintext modulus to 62 bits.
    let max = if TypeId::of::<T>() == TypeId::of::<u8>() {
        1 << 8
    } else {
        1 << 15
    };

    // Generate inputs.
    let mut x: Vec<Vec<Scalar>> = Vec::with_capacity(l);
    let mut x_int: Vec<Vec<u64>> = Vec::with_capacity(l);
    let mut p_x: Vec<Vec<Scalar>> = Vec::with_capacity(l);
    let mut y: Vec<Vec<Scalar>> = Vec::with_capacity(l);
    let mut y_int: Vec<Vec<u64>> = Vec::with_capacity(l);
    let mut p_y: Vec<Vec<Scalar>> = Vec::with_capacity(l);
    (0..l).for_each(|_| {
        // Initialize the input.
        let x_i_int: Vec<u64> = random_vector_in_range(n, max, &mut thread_rng());
        let y_i_int: Vec<u64> = random_vector_in_range(n, max, &mut thread_rng());
        x_int.push(x_i_int.clone());
        y_int.push(y_i_int.clone());

        // Input randomizer.
        let p_x_i = random_vector::<Scalar, Scalar, _>(n, &mut thread_rng());
        p_x.push(p_x_i.clone());
        let p_y_i = random_vector::<Scalar, Scalar, _>(n, &mut thread_rng());
        p_y.push(p_y_i.clone());

        // Create input component of public key.
        let x_i: Vec<Scalar> = x_i_int
            .iter()
            .map(|&x_i_field| Scalar::from(x_i_field))
            .collect();
        let y_i: Vec<Scalar> = y_i_int
            .iter()
            .map(|&y_i_field| Scalar::from(y_i_field))
            .collect();

        // Append to the list.
        x.push(x_i);
        y.push(y_i);
    });

    let _x_cts: Vec<Ciphertext> = Vec::with_capacity(l);
    let _y_cts: Vec<Ciphertext> = Vec::with_capacity(l);

    let t_client_ct_create_start = std::time::Instant::now();
    let x_cts: Vec<Ciphertext> = cfg_iter!(x_int)
        .map(|x_i_int| {
            // Create the ciphertext.
            let x_padded = [x_i_int.clone(), vec![0; d - n]].concat();
            let x_pt = Plaintext::try_encode(&x_padded, Encoding::simd(), &params).unwrap();
            sk.try_encrypt(&x_pt, &mut thread_rng()).unwrap()
        })
        .collect::<Vec<Ciphertext>>();
    let y_cts: Vec<Ciphertext> = cfg_iter!(y_int)
        .map(|y_i_int| {
            // Create the ciphertext.
            let y_padded = [y_i_int.clone(), vec![0; d - n]].concat();
            let y_pt = Plaintext::try_encode(&y_padded, Encoding::simd(), &params).unwrap();
            sk.try_encrypt(&y_pt, &mut thread_rng()).unwrap()
        })
        .collect::<Vec<Ciphertext>>();
    let t_client_ct_create = t_client_ct_create_start.elapsed();

    // Compute batches.
    let x_batch = linear_combination_gen(&alpha[1..].to_vec(), &x);
    let y_batch = linear_combination_gen(&beta[1..].to_vec(), &y);

    let t_client_start = std::time::Instant::now();

    // Compute proving keys.
    let x_pairs = cfg_iter!(h)
        .zip_eq(x_batch)
        .map(|(a, b)| ((*a).into(), b))
        .collect::<Vec<(G1, Scalar)>>();
    let y_pairs = cfg_iter!(r)
        .zip_eq(y_batch)
        .map(|(a, b)| ((*a).into(), b))
        .collect::<Vec<(G2, Scalar)>>();
    let alpha_0 = alpha.get(0).unwrap();
    let beta_0 = beta.get(0).unwrap();
    let c_x_batch = cfg_iter!(x_pairs)
        .map(|(h_i, x_i): &(G1, Scalar)| (h_i.mul(alpha_0) + G1::generator().mul(*x_i)))
        .collect::<Vec<_>>();
    let c_y_batch = cfg_iter!(y_pairs)
        .map(|(r_i, y_i): &(G2, Scalar)| (r_i.mul(beta_0) + G2::generator().mul(*y_i)))
        .collect::<Vec<_>>();

    // Create cross term verifying keys.
    let s_x = Scalar::rand(&mut rng);
    let s_y = Scalar::rand(&mut rng);

    // Cross term proving keys.
    let c_x = cfg_iter!(x)
        .zip_eq(p_x.clone())
        .map(|(x_i, p_x_i)| {
            cfg_iter!(x_i)
                .zip_eq(p_x_i)
                .map(|(x_i_j, p_x_i_j)| G1::generator().mul(s_x * &p_x_i_j + x_i_j))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let c_y = cfg_iter!(y)
        .zip_eq(p_y.clone())
        .map(|(y_i, p_y_i)| {
            cfg_iter!(y_i)
                .zip_eq(p_y_i)
                .map(|(y_i_j, p_y_i_j)| G2::generator().mul(s_y * &p_y_i_j + y_i_j))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let t_client_create_pk = t_client_start.elapsed();
    println!("client create pk: {:?}", t_client_create_pk);

    // Server: Online.
    // Homomorphically evaluate the quadratic function over every combination of (x_i, y_j).
    let a_diag_pt = Q_A_diag
        .iter()
        .map(|row| {
            Plaintext::try_encode(
                &[row.clone(), vec![0; n]].concat(),
                Encoding::simd_at_level(0),
                &params,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let (results, t_server_ct): (Vec<Vec<Ciphertext>>, _) = timeit_return_t!(
        "server cipher text op",
        cfg_iter!(x_cts)
            .map(|x_i| y_cts
                .iter()
                .map(|y_i| { fhe_quadratic(n, x_i, y_i.clone(), &a_diag_pt, &rk, &ek).unwrap() })
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );

    // Compute proof.
    let t_server_start = std::time::Instant::now();
    let proof = vec_matrix_vec_pairing(&c_x_batch, &c_y_batch, &A);

    // Compute the cross products from input hints.
    let p_xAr = cfg_iter!(p_x)
        .map(|p_x_i| {
            vec_matrix_vec_pairing(
                &cfg_iter!(p_x_i)
                    .map(|p_x_i_j| G1::generator().mul(p_x_i_j))
                    .collect::<Vec<G1>>(),
                &r,
                &A,
            )
        })
        .collect::<Vec<GT>>();
    let c_xAr = cfg_iter!(c_x)
        .map(|c_x_i| vec_matrix_vec_pairing(&c_x_i, &r, &A))
        .collect::<Vec<GT>>();
    let hAp_y = cfg_iter!(p_y)
        .map(|p_y_i| {
            vec_matrix_vec_pairing(
                &h,
                &cfg_iter!(p_y_i)
                    .map(|p_y_i_j| G2::generator().mul(p_y_i_j))
                    .collect::<Vec<G2>>(),
                &A,
            )
        })
        .collect::<Vec<GT>>();
    let hAc_y = cfg_iter!(c_y)
        .map(|c_y_i| vec_matrix_vec_pairing(&h, &c_y_i, &A))
        .collect::<Vec<GT>>();

    let t_server_create_pf = t_server_start.elapsed();
    println!("server create pf: {:?}", t_server_create_pf);

    // Client: Download.
    // Decrypt the ciphertexts.
    let (xAy, t_client_dec): (Vec<Vec<u64>>, _) = timeit_return_t!(
        "client dec cipher text",
        cfg_iter!(results)
            .map(|row| cfg_iter!(row)
                .map(|ct| {
                    Vec::<u64>::try_decode(&sk.try_decrypt(&ct).unwrap(), Encoding::simd()).unwrap()
                        [0]
                })
                .collect::<Vec<u64>>())
            .collect::<Vec<Vec<u64>>>()
    );

    let t_client_verify_pf_start = std::time::Instant::now();

    // Recover cross-product term sums.
    let xAr_sum: GT = cfg_iter!(c_xAr)
        .enumerate()
        .zip_eq(p_xAr)
        .map(|((i, c_xAr_i), mut p_xAr_i)| {
            p_xAr_i = p_xAr_i * &s_x;
            let mut v_ij = c_xAr_i.sub(p_xAr_i);
            v_ij *= &(alpha.get(i + 1).unwrap() * &beta.get(0).unwrap());
            v_ij
        })
        .sum();
    let hAy_sum: GT = cfg_iter!(hAc_y)
        .enumerate()
        .zip_eq(hAp_y)
        .map(|((i, hAc_y_i), mut hAp_y_i)| {
            hAp_y_i = hAp_y_i * &s_y;
            let mut v_ij = hAc_y_i.sub(hAp_y_i);
            v_ij = v_ij * &(alpha.get(0).unwrap() * &beta.get(i + 1).unwrap());
            v_ij
        })
        .sum();

    // Sum up all the xAy.
    let xAy_sum_scalar: Scalar = cfg_iter!(xAy)
        .enumerate()
        .map(|(i, row)| {
            cfg_iter!(row)
                .enumerate()
                .map(|(j, xAy_ij)| {
                    alpha.get(i + 1).unwrap()
                        * &beta.get(j + 1).unwrap()
                        * Scalar::from(xAy_ij.clone())
                    // let mut base = GT::generator();
                    // base = base * &v_ij;
                    // base
                })
                .sum::<Scalar>()
        })
        .sum::<Scalar>(); // TODO: Can we sum here or better to do fold w/ parrallel?

    let xAy_sum_gt = GT::generator() * &xAy_sum_scalar;

    // Verify the proof.
    cm = cm * &(alpha.get(0).unwrap() * beta.get(0).unwrap());
    assert_eq!(proof, xAy_sum_gt + xAr_sum + hAy_sum + cm);

    let t_client_verify = t_client_verify_pf_start.elapsed();
    println!("client verify pf: {:?}", t_client_verify);

    // Verify the results.
    let local_results = cfg_iter!(x_int)
        .map(
            |x_i| {
                cfg_iter!(y_int)
                    .map(|y_i| local_vec_matrix_vec(x_i.clone(), y_i.clone(), Q_A.clone()))
                    .collect::<Vec<u64>>()
            }, // Collect the inner iterator into a Vec<u64>
        )
        .collect::<Vec<Vec<u64>>>(); // Collect the outer iterator into a Vec<Vec<u64>>

    assert_eq!(local_results, xAy);

    let ratio_client_upload: f64 = t_client_ct_create.as_micros() as f64
        / ((t_client_ct_create + t_client_create_pk).as_micros() as f64);
    println!("ratio_client_upload: {:?}", ratio_client_upload);
    let ratio_server_online: f64 =
        t_server_ct.as_micros() as f64 / ((t_server_ct + t_server_create_pf).as_micros() as f64);
    println!("ratio_server_online: {:?}", ratio_server_online);
    let ratio_client_download =
        t_client_dec.as_micros() as f64 / ((t_client_dec + t_client_verify).as_micros() as f64);
    println!("ratio_client_download: {:?}", ratio_client_download);

    Ok(())
}

pub fn bench_inner_product_server<T: 'static>(
    n: usize,
    d: usize,
    p: u64,
    q_sizes: [usize; 3],
) -> Result<(), Box<dyn Error>>
where
    u64: From<T>,
    Standard: Distribution<T>,
{
    println!(
        "Inner product server bench: n: {:?}, d: {:?}, p: {:?}, q_sizes: {:?}",
        n, d, p, q_sizes
    );

    assert_eq!(n * 2, d);

    // Setup.
    // Create random number generator.
    let mut rng = thread_rng();

    // Sample common reference string.
    let h = random_vector::<G1, G1, _>(n, &mut rng);
    let r = random_vector::<G2, G2, _>(n, &mut rng);

    // Client: Upload.
    // Initialize BFV instance.
    let params = BfvParametersBuilder::new()
        .set_degree(d)
        .set_plaintext_modulus(p)
        .set_moduli_sizes(&q_sizes)
        .build_arc()
        .unwrap();

    // Create private key.
    let sk = SecretKey::random(&params, &mut rng);

    // Create relinearization key.
    let rk = RelinearizationKey::new(&sk, &mut rng).unwrap();

    // Create the evaluation key.
    let log2_length: usize = n.trailing_zeros() as usize;
    let ek = (0..log2_length)
        .map(|i| {
            EvaluationKeyBuilder::new_leveled(&sk, 0, 0)
                .unwrap()
                .enable_column_rotation(1 << i)
                .unwrap()
                .build(&mut rng)
                .unwrap()
        })
        .collect::<Vec<EvaluationKey>>();

    // Sample randomness.
    let alpha = random_vector::<Scalar, Scalar, _>(2, &mut rng);
    let beta = random_vector::<Scalar, Scalar, _>(2, &mut rng);

    // Initialize the input.
    let x_int: Vec<u64> = random_vector::<T, u64, _>(n, &mut rng);
    let y_int: Vec<u64> = random_vector::<T, u64, _>(n, &mut rng);

    // Input randomizer.
    let p_x = random_vector::<Scalar, Scalar, _>(n, &mut thread_rng());
    let p_y = random_vector::<Scalar, Scalar, _>(n, &mut thread_rng());

    // Create input component of public key.
    let x: Vec<Scalar> = x_int
        .iter()
        .map(|&x_i_field| Scalar::from(x_i_field))
        .collect();
    let y: Vec<Scalar> = y_int
        .iter()
        .map(|&y_i_field| Scalar::from(y_i_field))
        .collect();

    // Create ciphertexts.
    let x_padded = [x_int.clone(), vec![0; d - n]].concat();
    let x_pt = Plaintext::try_encode(&x_padded, Encoding::simd(), &params).unwrap();
    let x_ct = sk.try_encrypt(&x_pt, &mut thread_rng()).unwrap();
    let y_padded = [y_int.clone(), vec![0; d - n]].concat();
    let y_pt = Plaintext::try_encode(&y_padded, Encoding::simd(), &params).unwrap();
    let y_ct = sk.try_encrypt(&y_pt, &mut thread_rng()).unwrap();

    // Compute proving keys.
    let alpha_0 = alpha.get(0).unwrap();
    let beta_0 = beta.get(0).unwrap();
    let alpha_1 = alpha.get(1).unwrap();
    let beta_1 = beta.get(1).unwrap();
    let x_batch = x.iter().map(|x_i| x_i * alpha_1).collect::<Vec<_>>();
    let y_batch = y.iter().map(|y_i| y_i * beta_1).collect::<Vec<_>>();
    let c_x_batch: Vec<G1> = h
        .iter()
        .zip_eq(x_batch)
        .map(|(h_i, x_i)| (h_i.mul(alpha_0) + G1::generator().mul(x_i)))
        .collect::<Vec<_>>();
    let c_y_batch: Vec<G2> = r
        .iter()
        .zip_eq(y_batch)
        .map(|(r_i, y_i)| (r_i.mul(beta_0) + G2::generator().mul(y_i)))
        .collect::<Vec<_>>();

    // Create cross term verifying keys.
    let s_x = Scalar::rand(&mut rng);
    let s_y = Scalar::rand(&mut rng);
    let c_x = cfg_iter!(x)
        .zip_eq(p_x.clone())
        .map(|(x_i_j, p_x_i_j)| G1::generator().mul(s_x * &p_x_i_j + x_i_j))
        .collect::<Vec<_>>();
    let c_y = cfg_iter!(y)
        .zip_eq(p_y.clone())
        .map(|(y_i_j, p_y_i_j)| G2::generator().mul(s_y * &p_y_i_j + y_i_j))
        .collect::<Vec<_>>();

    // Server: Online.
    // Homomorphically evaluate the inner product.
    let (_result, t_server_ct): (Ciphertext, _) = timeit_return_t!(
        "server cipher text op",
        fhe_inner_product(n, &x_ct, y_ct, &rk, &ek).unwrap()
    );

    // Compute proof.
    let t_server_start = std::time::Instant::now();
    let _proof: GT = cfg_iter!(c_x_batch)
        .zip_eq(c_y_batch)
        .map(|(x_i, Ay_i)| Bls12_381::pairing(x_i, Ay_i))
        .sum();

    // Compute the cross products from input hints.
    let _c_xr: GT = cfg_iter!(c_x)
        .zip_eq(r)
        .map(|(c_x_i, r_i)| Bls12_381::pairing(c_x_i, r_i))
        .sum();
    let _hc_y: GT = cfg_iter!(h)
        .zip_eq(c_y)
        .map(|(h_i, c_y_i)| Bls12_381::pairing(h_i, c_y_i))
        .sum();

    let t_server_create_pf = t_server_start.elapsed();
    println!("server create pf: {:?}", t_server_create_pf);

    let ratio_server_online: f64 =
        t_server_ct.as_micros() as f64 / ((t_server_ct + t_server_create_pf).as_micros() as f64);
    println!("ratio_server_online: {:?}", ratio_server_online);
    Ok(())
}

pub fn bench_inner_product_server_poly<T: 'static>(
    n: usize,
    d: usize,
    p: u64,
    q_sizes: [usize; 3],
) -> Result<(), Box<dyn Error>>
where
    u64: From<T>,
    Standard: Distribution<T>,
{
    println!(
        "Inner product server bench: n: {:?}, d: {:?}, p: {:?}, q_sizes: {:?}",
        n, d, p, q_sizes
    );

    assert_eq!(n * 2, d);

    // Setup.
    // Create random number generator.
    let mut rng = thread_rng();

    // Sample common reference string.
    let h = random_vector::<G1, G1, _>(n, &mut rng);
    let r = random_vector::<G2, G2, _>(n, &mut rng);

    // Client: Upload.
    // Initialize BFV instance.
    let params = BfvParametersBuilder::new()
        .set_degree(d)
        .set_plaintext_modulus(p)
        .set_moduli_sizes(&q_sizes)
        .build_arc()
        .unwrap();

    // Create private key.
    let sk = SecretKey::random(&params, &mut rng);

    // Sample randomness.
    let alpha = random_vector::<Scalar, Scalar, _>(2, &mut rng);
    let beta = random_vector::<Scalar, Scalar, _>(2, &mut rng);

    // Initialize the input.
    let x_int: Vec<u64> = random_vector::<T, u64, _>(n, &mut rng);
    let y_int: Vec<u64> = random_vector::<T, u64, _>(n, &mut rng);

    // Input randomizer.
    let p_x = random_vector::<Scalar, Scalar, _>(n, &mut thread_rng());
    let p_y = random_vector::<Scalar, Scalar, _>(n, &mut thread_rng());

    // Create input component of public key.
    let x: Vec<Scalar> = x_int
        .iter()
        .map(|&x_i_field| Scalar::from(x_i_field))
        .collect();
    let y: Vec<Scalar> = y_int
        .iter()
        .map(|&y_i_field| Scalar::from(y_i_field))
        .collect();

    // Create ciphertexts.
    let x_padded = [x_int.clone(), vec![0; d - n]].concat();
    let x_pt = Plaintext::try_encode(&x_padded, Encoding::poly(), &params).unwrap();
    let x_ct: Ciphertext = sk.try_encrypt(&x_pt, &mut thread_rng()).unwrap();
    let mut y_flipped = y_int.clone();
    y_flipped.reverse();
    let y_padded = [vec![0; d - n], y_flipped.clone()].concat();
    let y_pt = Plaintext::try_encode(&y_padded, Encoding::poly(), &params).unwrap();
    let y_ct: Ciphertext = sk.try_encrypt(&y_pt, &mut thread_rng()).unwrap();

    // Compute proving keys.
    let alpha_0 = alpha.get(0).unwrap();
    let beta_0 = beta.get(0).unwrap();
    let alpha_1 = alpha.get(1).unwrap();
    let beta_1 = beta.get(1).unwrap();
    let x_batch = x.iter().map(|x_i| x_i * alpha_1).collect::<Vec<_>>();
    let y_batch = y.iter().map(|y_i| y_i * beta_1).collect::<Vec<_>>();
    let c_x_batch: Vec<G1> = h
        .iter()
        .zip_eq(x_batch)
        .map(|(h_i, x_i)| (h_i.mul(alpha_0) + G1::generator().mul(x_i)))
        .collect::<Vec<_>>();
    let c_y_batch: Vec<G2> = r
        .iter()
        .zip_eq(y_batch)
        .map(|(r_i, y_i)| (r_i.mul(beta_0) + G2::generator().mul(y_i)))
        .collect::<Vec<_>>();

    // Create cross term verifying keys.
    let s_x = Scalar::rand(&mut rng);
    let s_y = Scalar::rand(&mut rng);
    let c_x = cfg_iter!(x)
        .zip_eq(p_x.clone())
        .map(|(x_i_j, p_x_i_j)| G1::generator().mul(s_x * &p_x_i_j + x_i_j))
        .collect::<Vec<_>>();
    let c_y = cfg_iter!(y)
        .zip_eq(p_y.clone())
        .map(|(y_i_j, p_y_i_j)| G2::generator().mul(s_y * &p_y_i_j + y_i_j))
        .collect::<Vec<_>>();

    // Server: Online.
    // Homomorphically evaluate the inner product.
    let (_result, t_server_ct): (Ciphertext, _) =
        timeit_return_t!("server cipher text op", &x_ct * &y_ct);

    // Compute proof.
    let t_server_start = std::time::Instant::now();
    let _proof: GT = cfg_iter!(c_x_batch)
        .zip_eq(c_y_batch)
        .map(|(x_i, Ay_i)| Bls12_381::pairing(x_i, Ay_i))
        .sum();

    // Compute the cross products from input hints.
    let _c_xr: GT = cfg_iter!(c_x)
        .zip_eq(r)
        .map(|(c_x_i, r_i)| Bls12_381::pairing(c_x_i, r_i))
        .sum();
    let _hc_y: GT = cfg_iter!(h)
        .zip_eq(c_y)
        .map(|(h_i, c_y_i)| Bls12_381::pairing(h_i, c_y_i))
        .sum();

    let t_server_create_pf = t_server_start.elapsed();
    println!("server create pf: {:?}", t_server_create_pf);

    let ratio_server_online: f64 =
        t_server_ct.as_micros() as f64 / ((t_server_ct + t_server_create_pf).as_micros() as f64);
    println!("ratio_server_online: {:?}", ratio_server_online);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::quadratic::{fhe_quadratic, vQHE_batch};
    use crate::timeit;
    use crate::utils::{
        diagonal_encode_square_matrix, dot_product, print_matrix, print_vector,
        random_square_matrix, random_vector_in_range,
    };
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ec::pairing::{Pairing, PairingOutput};
    use ark_ec::{Group};
    use ark_ff::{UniformRand};
    use ark_std::{Zero};
    use fhe::bfv::{
        BfvParametersBuilder, Ciphertext, Encoding, EvaluationKey, EvaluationKeyBuilder, Plaintext,
        RelinearizationKey, SecretKey,
    };
    use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};
    use itertools::{Itertools};
    
    
    use rand::thread_rng;
    use std::any::TypeId;
    use std::ops::Mul;

    type G1 = <Bls12_381 as Pairing>::G1;
    type G2 = <Bls12_381 as Pairing>::G2;
    type Scalar = <Bls12_381 as Pairing>::ScalarField;
    type GT = PairingOutput<Bls12_381>;

    #[test]
    fn test_qhe() {
        vQHE_batch::<u8>(8, 8, 16, 8589952001, [62; 5]).unwrap();
    }

    #[ignore]
    #[test]
    fn test_basic_pairing_fixed() {
        let _rng = thread_rng();
        let a: <Bls12_381 as Pairing>::G1 = G1::generator().mul(Fr::from(2u64));
        let b: <Bls12_381 as Pairing>::G2 = G2::generator().mul(Fr::from(3u64));
        let s: <Bls12_381 as Pairing>::ScalarField =
            <Bls12_381 as Pairing>::ScalarField::from(2u64);
        let s2: <Bls12_381 as Pairing>::ScalarField =
            <Bls12_381 as Pairing>::ScalarField::from(12u64);
        let expected = <Bls12_381 as Pairing>::pairing(G1::generator(), G2::generator()) * &s2;
        let expected2 = GT::generator() * &s2;

        assert_eq!(expected, expected2);

        let sa = a * s;
        let sb = b * s;

        let ans1 = <Bls12_381>::pairing(sa, b);
        let ans2 = <Bls12_381>::pairing(a, sb);
        let ans3 = <Bls12_381>::pairing(a, b) * s;

        assert_eq!(ans1, expected);

        assert_eq!(ans1, ans2);
        assert_eq!(ans2, ans3);

        assert_ne!(ans1, PairingOutput::zero());
        assert_ne!(ans2, PairingOutput::zero());
        assert_ne!(ans3, PairingOutput::zero());
    }
    #[ignore]
    #[test]
    fn test_basic_pairing_random() {
        let mut rng = thread_rng();
        let a: <Bls12_381 as Pairing>::G1 = UniformRand::rand(&mut rng);
        let b: <Bls12_381 as Pairing>::G2 = UniformRand::rand(&mut rng);
        let s: <Bls12_381 as Pairing>::ScalarField = UniformRand::rand(&mut rng);

        let sa = a * s;
        let sb = b * s;

        let ans1 = <Bls12_381>::pairing(sa, b);
        let ans2 = <Bls12_381>::pairing(a, sb);
        let ans3 = <Bls12_381>::pairing(a, b) * s;

        assert_eq!(ans1, ans2);
        assert_eq!(ans2, ans3);

        assert_ne!(ans1, PairingOutput::zero());
        assert_ne!(ans2, PairingOutput::zero());
        assert_ne!(ans3, PairingOutput::zero());
    }

    #[ignore]
    #[test]
    fn test_fhe_vec_matrix_vec() {
        let n = 8;
        let degree = 16;
        let plaintext_modulus = 8589952001;
        let moduli_sizes = [62; 10];
        assert_eq!(n * 2, degree);

        // Server: Create commitment to linear function.
        // Create random number generator.
        let mut rng = thread_rng();

        let max = if TypeId::of::<u8>() == TypeId::of::<u8>() {
            1 << 8
        } else {
            // Trim 1 bit each, since fhe.rs limits the plaintext modulus to 62 bits.
            1 << 15
        };

        let x_raw: Vec<u64> = random_vector_in_range(n, max, &mut rng);
        let y_raw: Vec<u64> = random_vector_in_range(n, max, &mut rng);

        // Extend by zeros to make length even. Make it a vec.
        let x = [x_raw.clone(), vec![0; degree - n]].concat();
        let y = [y_raw.clone(), vec![0; degree - n]].concat();

        // Initialize BFV instance.
        let params = BfvParametersBuilder::new()
            .set_degree(degree)
            .set_plaintext_modulus(plaintext_modulus)
            .set_moduli_sizes(&moduli_sizes)
            .build_arc()
            .unwrap();

        // Create private key.
        let sk = SecretKey::random(&params, &mut rng);

        // Create relinearization key.
        let rk = RelinearizationKey::new(&sk, &mut rng).unwrap();

        // Initialize the operands.
        let Q_A_raw = random_square_matrix::<u8, u64, _>(n, &mut rng);
        let a_diag_raw = diagonal_encode_square_matrix(n, Q_A_raw.clone());
        let a_diag_pt = a_diag_raw
            .iter()
            .map(|row| {
                Plaintext::try_encode(
                    &[row.clone(), vec![0; n]].concat(),
                    Encoding::simd_at_level(0),
                    &params,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // Create the evaluation key.
        let log2_length: usize = n.trailing_zeros() as usize;
        let ek = (0..log2_length)
            .map(|i| {
                EvaluationKeyBuilder::new_leveled(&sk, 0, 0)
                    .unwrap()
                    .enable_column_rotation(1 << i)
                    .unwrap()
                    .build(&mut rng)
                    .unwrap()
            })
            .collect::<Vec<EvaluationKey>>();

        // Create the plaintexts and ciphertexts.
        let y_pt = timeit!(
            "create y_pt",
            Plaintext::try_encode(&y, Encoding::simd_at_level(0), &params).unwrap()
        );
        let x_pt = timeit!(
            "create y_pt",
            Plaintext::try_encode(&x, Encoding::simd_at_level(0), &params).unwrap()
        );
        let x_ct: Ciphertext = timeit!("create ct", sk.try_encrypt(&x_pt, &mut rng).unwrap());
        let y_ct: Ciphertext = timeit!("create ct", sk.try_encrypt(&y_pt, &mut rng).unwrap());

        // Compute the result.
        let result = fhe_quadratic(n, &x_ct, y_ct, &a_diag_pt, &rk, &ek).unwrap();

        // Decrypt result.
        let result_pt = timeit!("decrypt ct", sk.try_decrypt(&result).unwrap());
        let res_vec = timeit!(
            "decode pt",
            Vec::<u64>::try_decode(&result_pt, Encoding::simd_at_level(0)).unwrap()
        );

        // A * y.
        let expected_result_ay: Vec<u64> = Q_A_raw
            .iter()
            .map(|row| dot_product(&y_raw.clone(), row))
            .collect();
        // <x, A_y>
        let expected_result: u64 =
            dot_product::<u64, u64, u64>(&x_raw.clone(), &expected_result_ay);

        assert_eq!(res_vec[0], expected_result);

        dbg!(res_vec.clone());

        print_vector(x_raw.clone());
        println!("-------------");
        print_vector(y_raw.clone());
        println!("-------------");
        print_matrix(&Q_A_raw.clone());
    }
}
