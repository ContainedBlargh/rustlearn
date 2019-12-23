extern crate arrayfire;
extern crate rand;
extern crate rand_distr;

use arrayfire::*;
use rand::thread_rng;
use rand_distr::*;

pub fn d_row(rows: u64) -> Dim4 {
    return Dim4::new(&[1, rows, 1, 1]);
}

pub fn d_column(columns: u64) -> Dim4 {
    return Dim4::new(&[columns, 1, 1, 1]);
}

pub fn d_matrix(rows: u64, columns:u64) -> Dim4 {
    return Dim4::new(&[rows, columns, 1, 1]);
}

pub fn t(vector: &Array<f32>) -> Array<f32> {
    return transpose(vector, false);
}

pub fn array(data: &Vec<f32>, dims: Dim4) -> Array<f32> {
    return Array::new(data, dims);
}

/**
 * Multiplies two ArrayFire arrays.
 * 
 * ArrayFire matrix multiplication rules.
 * | Size of Input Matrix A | Size of Input Matrix B | Output Matrix Size |
 * |------------------------|------------------------|--------------------|
 * | {M,K,1,1}              | {K,N,1,1}              | {M,N,1,1}          |
 * | {M,K,b2,b3}            | {K,N,b2,b3}            | {M,N,b2,b3}        |
 * |                        | {K,N,b2,b3}            | {M,N,b2,b3}        |
 * | {M,K,b2,b3}            | {K,N,1,1}              | {M,N,b2,b3}        |
 * 
 */
pub fn mdot(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    if lhs.dims()[1] != rhs.dims()[0] {
        println!("lhs dims: {}, rhs dims: {}", lhs.dims(), rhs.dims());
        panic!("mdot failed: dimension mismatch");
    }
    return matmul(lhs, rhs, MatProp::NONE, MatProp::NONE);
}

pub fn relu(x: &Array<f32>) -> Array<f32> {
    return (abs(x) + x) / 2_f32;
}

pub fn relu_deriv(x: &Array<f32>) -> Array<f32> {
    let zeroes = constant(0.0_f32, x.dims());
    let ones = constant(1.0_f32, x.dims());
    let comparison = gt(x, &zeroes, false);
    return select(&ones, &comparison, &zeroes);
}

pub fn he_weights(width: u64, height: u64) -> Array<f32> {
    let dist = Normal::new(0_f64, 0.001_f64).unwrap();
    let values: Vec<f32> = dist.sample_iter(thread_rng())
                               .take((width * height) as usize)
                               .map(|f| f as f32)
                               .collect();
    return Array::new(&values, d_matrix(width, height));
}

pub fn zero_biases(width: u64) -> Array<f32> {
    return constant(0_f32, d_row(width));
}
