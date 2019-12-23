extern crate arrayfire;
extern crate rand;
extern crate rand_distr;

mod utils;
mod layer;
mod fc;

use arrayfire::*;
use utils::*;
use layer::*;
use fc::*;

fn arrayfire_test() {
    println!("Backends: {}, Device. {}", get_backend_count(), get_device());

    set_backend(Backend::OPENCL);

    let random_vector = randu(d_row(5));

    af_print!("Random vector:", &random_vector);

    let relu_ed = relu(&random_vector);
    
    af_print!("Relu'ed vector:", &relu_ed);

    let minus_1 = relu(&(&random_vector - (constant(0.5_f32, d_row(5)))));

    af_print!("Relu'ed random vector minus 0.5", &minus_1);

    let d5x10 = d_matrix(5, 10);
    let random_matrix: Array<f32> = randu(d5x10);

    af_print!("Random matrix:", &random_matrix);

    let tanh_matrix = tanh(&(random_matrix - (constant(0.5_f32, d5x10))));

    af_print!("Random matrix relued:", &tanh_matrix);

    println!("Vector dims: {}, Matrix dims: {}", minus_1.dims(), tanh_matrix.dims());

    af_print!("Product:", mdot(&minus_1, &tanh_matrix));

    af_print!("Transposed vector:", t(&minus_1));
}

fn fc_test() {
    let input = Array::new(&[1.0, 0.0, 2.0, 0.0, 3.0], d_column(5));
    af_print!("Input:", input);
    let mut fc = init_identity_fc(5, 5);
    let output = fc.apply(&input);
    af_print!("Output:", output);

    let target = Array::new(&[1.0, 0.0, 1.0, 0.0, 1.0], d_column(5));
    let gt = fc.gradient(&target);
    af_print!("Gradient:", gt);
}

fn main() {
    arrayfire_test();
    fc_test();
}

//DOWNLOAD ARRAYFIRE: https://arrayfire.com/download/
