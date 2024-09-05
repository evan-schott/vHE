use crate::utils::{
    diagonal_encode_square_matrix, dot_product, flip_matrix, linear_combination_gen,
    random_vector,
};
use crate::{timeit, timeit_return_t};
use ark_bls12_381::{Fr, G1Affine as G1, G1Projective};
use ark_ec::VariableBaseMSM;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{UniformRand, Zero};
use ark_std::rand::distributions::{Distribution};
use ark_std::{cfg_into_iter, cfg_iter};
use fhe::bfv::{
    BfvParametersBuilder, Ciphertext, Encoding, EvaluationKey, EvaluationKeyBuilder, Plaintext,
    SecretKey,
};
use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};
use itertools::Itertools;
use rand::distributions::Standard;
use rand::{thread_rng};
use rayon::prelude::*;
use std::error::Error;
use std::ops::{Mul};

/// Batch FHE linear function evaluations.
/// n: Input length
/// d: Degree of ciphertext
/// p: Plaintext modulus
/// q_sizes: Moduli sizes
pub fn linear_poly<T>(n: usize, d: usize, p: u64, q_sizes: [usize; 3]) -> Result<(), Box<dyn Error>>
where
    u64: From<T>,
    Standard: Distribution<T>,
{
    println!("vLHE n: {}, d: {}, p: {}, q_sizes: {:?}", n, d, p, q_sizes);
    assert_eq!(n * 2, d);

    // Server: Create commitment to linear function.
    // Create random number generator.
    let mut rng = thread_rng();

    // Initialize the linear function.
    let a_int = random_vector::<T, u64, _>(n, &mut rng);
    let mut a_fp: Vec<Fr> = a_int.iter().map(|a_i| Fr::from(*a_i)).collect();
    // Sample 1 element from F_p.
    let y = Fr::rand(&mut rng);
    a_fp.push(y);

    // Sample n + 1 elements from G1.
    let r = random_vector::<G1, G1, _>(n + 1, &mut rng);

    // Pedersen vector commitment to linear function.
    let cm = timeit!(
        "dot prod cm",
        <G1Projective as VariableBaseMSM>::msm(&r, &a_fp).unwrap()
    );

    // Client: Send public key and ciphertext.
    // Initialize BFV instance.
    let params = BfvParametersBuilder::new()
        .set_degree(d)
        .set_plaintext_modulus(p)
        .set_moduli_sizes(&q_sizes)
        .build_arc()
        .unwrap();

    // Create private key.
    let sk = SecretKey::random(&params, &mut rng);

    // Initialize the input.
    let x: Vec<u64> = random_vector::<T, u64, _>(n, &mut rng);
    // Create input component of public key.
    let mut x_field: Vec<Fr> = x.iter().map(|&x| Fr::from(x)).collect();
    // Append 0's to make it the same size as the linear function.
    x_field.push(Fr::zero());
    // Convert to G1.
    let x_g1: Vec<G1> = x_field
        .iter()
        .map(|val| G1::generator().mul(val).into_affine())
        .collect();

    // Create the ciphertext.
    let x_padded = [x.clone(), vec![0; d - n]].concat();
    let x_pt = timeit!(
        "create pt",
        Plaintext::try_encode(&x_padded, Encoding::poly(), &params).unwrap()
    );
    let mut ct: Ciphertext = timeit!("create ct", sk.try_encrypt(&x_pt, &mut rng).unwrap());

    // Create the verifying key (a, b).
    let vk: [Fr; 2] = random_vector::<Fr, Fr, _>(2, &mut rng).try_into().unwrap();

    // Create the proving key.
    let pairs = r
        .iter()
        .zip_eq(x_g1)
        .map(|(a, b)| ((*a).into(), b.into()))
        .collect::<Vec<_>>();
    let proving_key: Vec<G1> = timeit!(
        "create proving key",
        cfg_iter!(pairs)
            .map(|(r, x): &(G1, G1)| (r.mul(vk[0]) + x.mul(vk[1])).into_affine())
            .collect()
    );

    // Server: Perform delegated computation, and generate proof.
    // Generate proof.
    let pf = timeit!(
        "create proof",
        <G1Projective as VariableBaseMSM>::msm(&proving_key, &a_fp).unwrap()
    );
    let mut a_flipped = a_int.clone();
    a_flipped.reverse();
    let a_padded = [vec![0; d - n], a_flipped].concat();

    // Apply linear function to ciphertext.
    let a_pt = Plaintext::try_encode(&a_padded, Encoding::poly(), &params).unwrap();
    timeit!("server ct", ct = &ct * &a_pt);

    // Client: Verify proof.
    // Decrypt result.
    let y_pt = timeit!("decrypt ct", sk.try_decrypt(&ct).unwrap());
    let result = timeit!(
        "decode pt",
        Vec::<u64>::try_decode(&y_pt, Encoding::poly()).unwrap()
    )[d - 1];

    let y_g1: G1 = G1::generator().mul(Fr::from(result)).into_affine();

    // Verify proof.
    let client_proof: G1 = timeit!("verify pf", (cm.mul(vk[0]) + y_g1.mul(vk[1])).into_affine());
    assert_eq!(client_proof, pf);

    // Check the result.
    assert_eq!(result, dot_product::<u64, u64, u64>(&x, &a_int));

    Ok(())
}

/// Batch FHE linear function evaluations.
/// n: Input length
/// d: Degree of ciphertext
/// p: Plaintext modulus
/// q_sizes: Moduli sizes
pub fn linear_simd<T>(n: usize, d: usize, p: u64, q_sizes: [usize; 3]) -> Result<(), Box<dyn Error>>
where
    u64: From<T>,
    Standard: Distribution<T>,
{
    println!("vLHE n: {}, d: {}, p: {}, q_sizes: {:?}", n, d, p, q_sizes);
    assert_eq!(n * 2, d);

    // Server: Create commitment to linear function.
    // Create random number generator.
    let mut rng = thread_rng();

    // Initialize the linear function.
    let a_int = random_vector::<T, u64, _>(n, &mut rng);
    let mut a_fp: Vec<Fr> = a_int.iter().map(|a_i| Fr::from(*a_i)).collect();
    // Sample 1 element from F_p.
    let y = Fr::rand(&mut rng);
    a_fp.push(y);

    // Sample n + 1 elements from G1.
    let r = random_vector::<G1, G1, _>(n + 1, &mut rng);

    // Pedersen vector commitment to linear function.
    let cm = timeit!(
        "dot prod cm",
        <G1Projective as VariableBaseMSM>::msm(&r, &a_fp).unwrap()
    );

    // Client: Send public key and ciphertext.
    // Initialize BFV instance.
    let params = BfvParametersBuilder::new()
        .set_degree(d)
        .set_plaintext_modulus(p)
        .set_moduli_sizes(&q_sizes)
        .build_arc()
        .unwrap();

    // Create private key.
    let sk = SecretKey::random(&params, &mut rng);

    // Initialize the input.
    let x: Vec<u64> = random_vector::<T, u64, _>(n, &mut rng);
    // Create input component of public key.
    let mut x_field: Vec<Fr> = x.iter().map(|&x| Fr::from(x)).collect();
    // Append 0's to make it the same size as the linear function.
    x_field.push(Fr::zero());
    // Convert to G1.
    let x_g1: Vec<G1> = x_field
        .iter()
        .map(|val| G1::generator().mul(val).into_affine())
        .collect();

    // Create the ciphertext.
    let x_padded = [x, vec![0; d - n]].concat();
    let x_pt = timeit!(
        "create pt",
        Plaintext::try_encode(&x_padded, Encoding::simd(), &params).unwrap()
    );
    let mut ct: Ciphertext = timeit!("create ct", sk.try_encrypt(&x_pt, &mut rng).unwrap());

    // Create the verifying key (a, b).
    let vk: [Fr; 2] = random_vector::<Fr, Fr, _>(2, &mut rng).try_into().unwrap();

    // Create the proving key.
    let pairs = r
        .iter()
        .zip_eq(x_g1)
        .map(|(a, b)| ((*a).into(), b.into()))
        .collect::<Vec<_>>();
    let proving_key: Vec<G1> = timeit!(
        "create proving key",
        cfg_iter!(pairs)
            .map(|(r, x): &(G1, G1)| (r.mul(vk[0]) + x.mul(vk[1])).into_affine())
            .collect()
    );

    // Server: Perform delegated computation, and generate proof.
    // Generate proof.
    let pf = timeit!(
        "create proof",
        <G1Projective as VariableBaseMSM>::msm(&proving_key, &a_fp).unwrap()
    );

    let a_padded = [a_int, vec![0; d - n]].concat();

    // Apply linear function to ciphertext.
    let a_pt = Plaintext::try_encode(&a_padded, Encoding::simd(), &params).unwrap();
    let t_server_ct_start = std::time::Instant::now();
    ct = &ct * &a_pt;

    // Make sure we have a power of 2 as length.
    assert_eq!(n.count_ones(), 1);

    // Calculate log2 of length.
    let log2_length = n.trailing_zeros();

    // Do first rotation & sum.
    let mut rot: Ciphertext;
    // Do the rest of the rotations & sums.
    for shift in (0..log2_length).map(|i| 1 << i) {
        // Create the evaluation key.
        let ek_i = EvaluationKeyBuilder::new_leveled(&sk, 0, 0)?
            .enable_column_rotation(shift)?
            .build(&mut rng)?;
        rot = ek_i.rotates_columns_by(&ct, shift)?;
        ct = &ct + &rot;
    }

    let t_server_ct = t_server_ct_start.elapsed();
    println!("server ct: {:?}", t_server_ct);

    // Client: Verify proof.
    // Decrypt result.
    let y_pt = timeit!("decrypt ct", sk.try_decrypt(&ct).unwrap());
    let y = timeit!(
        "decode pt",
        Vec::<u64>::try_decode(&y_pt, Encoding::simd()).unwrap()
    );

    // Dot product will be collected in all terms.
    let y_sum = y[0];
    let y_g1: G1 = G1::generator().mul(Fr::from(y_sum)).into_affine();

    // Verify proof.
    let client_proof: G1 = timeit!("verify pf", (cm.mul(vk[0]) + y_g1.mul(vk[1])).into_affine());

    assert_eq!(client_proof, pf);

    Ok(())
}

/// Batch FHE linear function evaluations.
/// n: Input length
/// m: Number of linear functions
/// l: Number of inputs
/// d: Degree of ciphertext
/// p: Plaintext modulus
/// q_sizes: Moduli sizes
pub fn batch_linear<T>(
    m: usize,
    l: usize,
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
        "vLHE_batch m: {}, l: {}, n: {}, d: {}, p: {}, q_sizes: {:?}",
        m, l, n, d, p, q_sizes
    );

    assert_eq!(n * 2, d);
    assert_eq!(n, m);

    // Server: Create commitments to linear functions.
    // Create random number generator.
    let mut rng = thread_rng();

    // Sample n + 1 elements from G1.
    let r = random_vector::<G1, G1, _>(n + 1, &mut rng);

    // Initialize the linear functions.
    let mut linear_functions: Vec<Vec<u64>> = Vec::with_capacity(m); // m x n-length
    let mut linear_functions_fp: Vec<Vec<Fr>> = Vec::with_capacity(m); // m x (n + 1)-length
    (0..m).for_each(|_| {
        let a_int = random_vector::<T, u64, _>(n, &mut thread_rng());
        let mut a_fp: Vec<Fr> = a_int.iter().map(|a_i| Fr::from(*a_i)).collect();
        // Sample 1 element from F_p.
        let y = Fr::rand(&mut thread_rng());
        a_fp.push(y);

        // Pedersen vector commitment to linear function.
        linear_functions.push(a_int);
        linear_functions_fp.push(a_fp);
    });

    // Calculate the commitments.
    let cm_vals: Vec<G1> = timeit!(
        "create cm vals",
        cfg_iter!(linear_functions_fp)
            .map(|a_fp| <G1Projective as VariableBaseMSM>::msm(&r, &a_fp)
                .unwrap()
                .into_affine())
            .collect()
    );

    // Client: Send public key and ciphertext
    // Initialize BFV instance.
    let params = BfvParametersBuilder::new()
        .set_degree(d)
        .set_plaintext_modulus(p)
        .set_moduli_sizes(&q_sizes)
        .build_arc()
        .unwrap();

    // Create private key.
    let sk = SecretKey::random(&params, &mut rng);

    // Initialize the inputs.
    let mut inputs: Vec<Vec<Fr>> = Vec::with_capacity(l);
    let mut inputs_int: Vec<Vec<u64>> = Vec::with_capacity(l);
    (0..l).for_each(|_| {
        // Initialize the input.
        let x: Vec<u64> = random_vector::<T, u64, _>(n, &mut thread_rng());
        inputs_int.push(x.clone());
        // Create input component of public key.
        let x_field: Vec<Fr> = x.iter().map(|&x| Fr::from(x)).collect();
        // // Convert to G1.
        // let x_g1: Vec<G1> = x_field.iter().map(|val| G1::generator().mul(val).into_affine()).collect();
        inputs.push(x_field);
    });

    let (cts, t_client_create_ct): (Vec<Ciphertext>, _) = timeit_return_t!(
        "create l ct",
        cfg_iter!(inputs_int)
            .map(|x| {
                // Create the ciphertext.
                let x_padded: Vec<u64> = [x.clone(), vec![0; d - n]].concat();
                let x_pt = Plaintext::try_encode(&x_padded, Encoding::simd(), &params).unwrap();
                let ct: Ciphertext = sk.try_encrypt(&x_pt, &mut thread_rng()).unwrap();

                // Append to the list.
                ct
            })
            .collect::<Vec<_>>()
    );

    // Create the verifying key (\alpha_1, ... \alpha_n).
    let vk: Vec<Fr> = random_vector::<Fr, Fr, _>(l, &mut rng);
    let x_batch = linear_combination_gen(&vk, &inputs);
    // \alpha_0
    let vk_0 = Fr::rand(&mut rng);

    // Create the proving key.
    let pairs: Vec<(G1, Fr)> = r[..n]
        .iter()
        .zip_eq(x_batch)
        .map(|(a, b)| ((*a).into(), b))
        .collect();
    let (proving_key, t_client_create_pk) = timeit_return_t!(
        "create proving key",
        vec![
            cfg_iter!(pairs)
                .map(|(r, x): &(G1, Fr)| (r.mul(vk_0) + G1::generator().mul(*x)).into_affine())
                .collect(),
            vec![(r.get(n).unwrap()).mul(vk_0).into_affine()]
        ]
        .concat()
    );

    // Server: Perform delegated computation, and generate proof.
    // Generate proofs.
    let (proofs, t_server_pk) = timeit_return_t!(
        "server proving key",
        cfg_iter!(linear_functions_fp)
            .map(|a_fp| { <G1Projective as VariableBaseMSM>::msm(&proving_key, &a_fp).unwrap() })
            .collect::<Vec<_>>()
    );

    // Apply linear functions to ciphertexts.
    // Make sure we have a power of 2 as length.
    assert_eq!(n.count_ones(), 1);

    // Calculate log2 of length.
    let log2_length = n.trailing_zeros();

    // Make the evaluation keys in advance. (Can be passed from client).
    let ekeys: Vec<EvaluationKey> = cfg_into_iter!(0..log2_length)
        .map(|i| {
            EvaluationKeyBuilder::new_leveled(&sk, 0, 0)
                .unwrap()
                .enable_column_rotation(1 << i)
                .unwrap()
                .build(&mut thread_rng())
                .unwrap()
        })
        .collect();

    // Encode the linear functions diagonally into a matrix.
    let linear_functions_diagonal_matrix =
        diagonal_encode_square_matrix(m, linear_functions.clone());
    let (a_pt, t_server_ct_mat_mat) = timeit_return_t!(
        "server matrix-matrix",
        cfg_iter!(linear_functions_diagonal_matrix)
            .map(|a| {
                let zeros: Vec<u64> = vec![0; d - n];
                let a_padded: Vec<u64> = [&a[..], &zeros[..]].concat();
                Plaintext::try_encode(&a_padded, Encoding::simd(), &params).unwrap()
            })
            .collect::<Vec<_>>()
    );

    // Apply linear functions to ciphertexts.
    let (sum_cts, t_server_inner_sum) = timeit_return_t!(
        "server inner product",
        cfg_iter!(cts)
            .enumerate()
            .map(|(_proc, ct_xi)| {
                let mut ct_xi_shifted = ct_xi.clone();
                let mut a_xi = ct_xi * &a_pt[0];
                for i in 1..m {
                    ct_xi_shifted = ekeys[0].rotates_columns_by(&ct_xi_shifted, 1).unwrap();

                    a_xi = &a_xi + &(&ct_xi_shifted * &a_pt[i]);
                }
                a_xi
            })
            .collect::<Vec<_>>()
    );

    // Client: Verify proof.
    // Decrypt results.
    let (ys_flipped, t_client_decrypt) = timeit_return_t!(
        "decrypt ct",
        cfg_iter!(sum_cts)
            .map(|ct_from_xi| {
                let a_xj_pt = sk.try_decrypt(&ct_from_xi).unwrap();
                let y_dec = Vec::<u64>::try_decode(&a_xj_pt, Encoding::simd()).unwrap();
                let (first, _) = y_dec.split_at(n);
                first.to_vec()
            })
            .collect::<Vec<Vec<u64>>>()
    );

    // Flip the matrix along the diagonal.
    let ys = flip_matrix(ys_flipped);

    // convert to Fr.
    let ys_fr: Vec<Vec<Fr>> = ys
        .iter()
        .map(|ys| ys.iter().map(|y| Fr::from(*y)).collect())
        .collect();

    // Verify proofs.
    let (_, t_client_verify) = timeit_return_t!(
        "verify proofs",
        cfg_iter!(proofs)
            .zip_eq(cm_vals)
            .zip_eq(ys_fr.clone())
            .for_each(|((pf, cm), ys_from_li)| {
                assert_eq!(
                    (pf.into_affine()),
                    (cm.mul(vk_0)
                        + G1::generator().mul(dot_product::<Fr, Fr, Fr>(&vk, &ys_from_li)))
                    .into_affine()
                );
            })
    );

    // Benchmark local computation.
    let (ys_local, t_local): (Vec<Vec<u64>>, _) = timeit_return_t!(
        "local cost",
        cfg_iter!(linear_functions)
            .map(|l_a| {
                cfg_iter!(inputs_int)
                    .map(|x| {
                        // Collect the result of dot_product into y
                        dot_product::<u64, u64, u64>(&x, &l_a)
                    })
                    .collect() // Collects into Vec<u64>
            })
            .collect()
    );

    // Comparison check.
    assert_eq!(ys, ys_local);

    println!(
        "total: {:?}",
        t_client_create_ct + t_client_create_pk + t_client_decrypt + t_client_verify
    );

    let ratio = ((t_client_create_ct + t_client_create_pk + t_client_decrypt + t_client_verify)
        .as_micros() as f64)
        / (t_local.as_micros() as f64);

    println!(
        "total server {:?}",
        t_server_pk + t_server_ct_mat_mat + t_server_inner_sum
    );

    let ratio_server = t_server_pk.as_micros() as f64
        / (t_server_pk + t_server_ct_mat_mat + t_server_inner_sum).as_micros() as f64;

    println!("ratio_server: {:?}", ratio_server);

    let client_upload_ratio = (t_client_create_pk.as_micros() as f64)
        / (t_client_create_ct + t_client_create_pk).as_micros() as f64;
    let client_download_ratio = (t_client_verify.as_micros() as f64)
        / (t_client_decrypt + t_client_verify).as_micros() as f64;

    println!("client_upload_ratio: {:?}", client_upload_ratio);
    println!("client_download_ratio: {:?}", client_download_ratio);

    println!("ratio: {:?}", ratio);

    Ok(())
}
