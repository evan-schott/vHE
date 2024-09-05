## vHE: Verifiable Homomorphic Encryption for Linear and Quadratic Functions

### Overview
- Client sends server encrypted data and proving key, server homomorphically invokes function on encrypted data and returns verifiable proof.
- Uses BFV scheme implemented in [fhe](https://github.com/tlepoint/fhe.rs) crate and BLS 12-381 implemented in [arkworks](https://github.com/arkworks-rs) for elliptic curve operations.

### Examples
`cargo test test_all --features parallel -- --nocapture` 

