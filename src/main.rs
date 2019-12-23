extern crate arrayfire;
extern crate rand;
extern crate rand_distr;
extern crate csv;
extern crate statistical;

mod utils;
mod layer;
mod fc;

use arrayfire::*;
use utils::*;
use layer::*;
use fc::*;

use csv::*;
use statistical::standard_deviation as stddev;

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

fn train_test() {
    let mut rdr = Reader::from_path("./data/train.csv").unwrap();
    let records = rdr.records()
                     .map(|r| r.unwrap())
                     .map(|r: StringRecord| (r[8].parse::<i32>().unwrap(), r[1].parse::<i32>().unwrap()));
    let (age, survived): (Vec<_>, Vec<_>) = records.unzip();
    println!("age, survived:\n{:?}\n{:?}", age, survived);
    let n = age.len();
    let mean_age = (age.iter().sum::<i32>() as f32) / n as f32;
    let std_age = stddev(&(age.iter().map(|i| *i as f32).collect::<Vec<f32>>()), None);
    let norm_age: Vec<f32> = age.iter().map(|age| ((*age as f32) - mean_age) / std_age).collect();
    let input = array(&norm_age, d_column(n as u64));
    let output = array(&survived.iter().map(|s| *s as f32).collect(), d_column(n as u64));

    let mut layer1 = FCLayer::new(he_weights(n as u64, (n as u64) * 2), zero_biases(5), relu, relu_deriv);
    let mut layer2 = FCLayer::new(he_weights((n as u64) * 2, n as u64), zero_biases(1), relu, relu_deriv);

    let epochs = 500;
    for i in 0..epochs {
        let l1o = layer1.apply(&input);
        let l2o = layer2.apply(&l1o);
        af_print!("Input, Output, Prediction:", join_many(1, vec!(&input, &l2o, &output)));
    }
}

fn main() {
    // arrayfire_test();
    // fc_test();
    train_test();
}

//DOWNLOAD ARRAYFIRE: https://arrayfire.com/download/
