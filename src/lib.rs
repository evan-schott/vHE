pub mod linear;
pub mod quadratic;
mod utils;

#[cfg(test)]
mod tests {
    use crate::linear::{batch_linear, linear_poly, linear_simd};
    use crate::quadratic::{bench_inner_product_server_poly, vQHE_batch};
    use crate::timeit;
    use crate::utils::{
        diagonal_encode_square_matrix, dot_product, linear_combination_gen, print_matrix,
        print_vector, random_square_matrix, random_vector, random_vector_in_range,
    };
    use ark_bls12_381::{Fr, G1Affine, G1Projective as G1, G1Projective};
    use ark_ec::{CurveGroup, VariableBaseMSM};
    use ark_ff::UniformRand;
    
    use fhe::bfv::{BfvParametersBuilder, Ciphertext, Encoding, Plaintext, SecretKey};
    use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};
    use itertools::Itertools;
    use rand::distributions::Uniform;
    use rand::prelude::ThreadRng;
    use rand::thread_rng;
    use std::ops::Mul;

    #[test]
    fn test_all() {
        vQHE_batch::<u16>(2, 8, 16, 2305843009213704193, [62; 5]).unwrap();
        linear_simd::<u16>(8, 16, 70368744210433, [62; 3]).unwrap();
        linear_poly::<u16>(8, 16, 70368744210433, [62; 3]).unwrap();
        batch_linear::<u16>(8, 8, 8, 16, 70368744210433, [62; 3]).unwrap();
    }

    #[ignore]
    #[test]
    fn test_inner_product() {
        for i in 3..14 {
            bench_inner_product_server_poly::<u16>(1 << i, 1 << (i + 1), 70368744210433, [62; 3])
                .unwrap();
        }
    }

    #[ignore]
    #[test]
    fn test_linear() {
        for i in 3..14 {
            linear_poly::<u16>(1 << i, 1 << (i + 1), 70368744210433, [62; 3]).unwrap();
        }
    }

    #[ignore]
    #[test]
    fn test_quadratic_batch() {
        for i in 1..10 {
            vQHE_batch::<u16>(20 * i, 256, 512, 2305843009213704193, [62; 5]).unwrap();
        }
    }

    #[ignore]
    #[test]
    fn test_linear_batch() {
        for i in 3..14 {
            batch_linear::<u16>(
                1 << i,
                1 << i,
                1 << i,
                1 << (i + 1),
                70368744210433,
                [62; 3],
            )
            .unwrap();
        }
    }

    #[test]
    fn test_msm() {
        let n = 800;
        let mut rng = thread_rng();
        let g_n = random_vector::<G1, G1, _>(n, &mut rng);
        let g_n_affine: Vec<G1Affine> = g_n.iter().map(|g| g.into_affine()).collect_vec();
        let f_n = random_vector::<u128, Fr, _>(n, &mut rng);

        let _res_1 = timeit!(
            "MSM specialized",
            <G1Projective as VariableBaseMSM>::msm(&g_n_affine, &f_n).unwrap()
        );
    }

    #[test]
    fn test_bls_12_381() {
        // Initialize the random number generator
        let mut rng = thread_rng();

        timeit!("random mul", {
            for _ in 0..800 {
                // Generate a random group element in G1
                let random_g1_element = G1::rand(&mut rng);

                // Generate a random scalar (Fr is the field of scalars for BLS12-381)
                let random_scalar = Fr::rand(&mut rng);

                // Multiply the group element by the scalar
                let _result = random_g1_element.mul(&random_scalar);
            }
        });
    }

    #[test]
    fn test_dot_product() {
        // Initialize parameters.
        let degree = 4096;
        // Want enough for 8-bit integer multiplication.
        let plaintext_modulus: u64 = 2 << 25;
        // Required for 128-bit security / sufficient noise budget given degree.
        let moduli_sizes = [38, 38, 38];

        // Initialize BFV instance.
        // Need (p is prime) AND (p = 1 mod d) for simd encoding.
        let params = BfvParametersBuilder::new()
            .set_degree(degree)
            .set_plaintext_modulus(plaintext_modulus)
            .set_moduli_sizes(&moduli_sizes)
            .build_arc()
            .unwrap();

        for _ in 0..100 {
            // Create private key.
            let mut rng = thread_rng();
            let sk = SecretKey::random(&params, &mut rng);

            // Create vector.
            let n = 400;
            let _distribution: Uniform<u64> = Uniform::new(0, u8::MAX as u64);

            let short_a = random_vector::<u8, u64, _>(n, &mut rng);
            let mut short_b = random_vector::<u8, u64, _>(n, &mut rng);

            let dot_prod = dot_product(&short_a.clone(), &short_b.clone());

            // Pad with 0's.
            let mut a: Vec<u64> = vec![0; degree - n];
            a = [short_a.clone(), a].concat();

            // Flip the second vector.
            short_b.reverse();
            let mut b: Vec<u64> = vec![0; degree - n];
            b = [b, short_b.clone()].concat();

            let a_pt = Plaintext::try_encode(&a, Encoding::poly(), &params).unwrap();
            let b_pt = Plaintext::try_encode(&b, Encoding::poly(), &params).unwrap();
            let mut ct: Ciphertext = sk.try_encrypt(&a_pt, &mut rng).unwrap();

            // Multiply.
            ct = &ct * &b_pt;

            // Recover dot product.
            let result_pt = sk.try_decrypt(&ct).unwrap();
            let result = Vec::<u64>::try_decode(&result_pt, Encoding::poly()).unwrap();

            short_b.reverse();
            // println!("Result {:?} * {:?}: {}", short_a, short_b, dot_prod);
            assert_eq!(result[degree - 1], dot_prod);
        }
    }

    #[test]
    fn test_diagonal_encode_matrix() {
        let matrix = random_square_matrix::<u8, u8, _>(9, &mut thread_rng());
        print_matrix(&matrix);

        let result = diagonal_encode_square_matrix(9, matrix);

        println!("------------");

        print_matrix(&result);
    }

    #[test]
    fn test_linear_combination() {
        let n = 2;
        let m = 3;
        let rng = &mut thread_rng();
        let alpha = random_vector_in_range(m, 3, rng);
        print_vector(alpha.clone());
        println!("------------");
        let mut x = Vec::new();
        for _ in 0..m {
            x.push(random_vector_in_range(n, 3, rng));
        }
        print_matrix(&x);
        println!("------------");

        let result = linear_combination_gen::<u64>(&alpha, &x);
        print_vector(result);
    }

    #[test]
    fn test_pairing() {
        use ark_bls12_381::{Bls12_381, G1Projective, G2Projective};
        use ark_ec::pairing::{Pairing};
        use ark_std::UniformRand;

        // Create a random number generator
        let mut rng = ark_std::test_rng();

        // Generate random points on G1 and G2
        let g1_point = G1Projective::rand(&mut rng).into_affine();
        let g2_point = G2Projective::rand(&mut rng).into_affine();

        // Perform the pairing
        let pairing_result = Bls12_381::pairing(g1_point, g2_point);
        let _a = pairing_result.0;

        // Output the result
        println!("Pairing result: {:?}", pairing_result);
    }
}
