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
    let gt = fc.error_delta(&target);
    af_print!("Gradient:", gt);
}

fn train_test() {
    let mut rdr = Reader::from_path("./data/train.csv").unwrap();
    let records = rdr.records()
                    .map(Result::unwrap)
                    .enumerate()
                    .filter_map(|(i, r)| {
                        let age = r[5].parse::<i32>();
                        let survived = r[1].parse::<i32>();
                        return age.map(|a| survived.map(|s| (a, s))).ok();
                    })
                    .map(|r| r.unwrap())
                    .take(20);
    let (age, survived): (Vec<_>, Vec<_>) = records.unzip();
    let n = age.len() as u64;
    let mean_age = (age.iter().sum::<i32>() as f32) / n as f32;
    let std_age = stddev(&(age.iter().map(|i| *i as f32).collect::<Vec<f32>>()), None);
    let norm_age: Vec<f32> = age.iter().map(|age| ((*age as f32) - mean_age) / std_age).collect();
    let input = array(&norm_age, d_column(n));
    let output = array(&survived.iter().map(|s| *s as f32).collect(), d_column(n));

    let mut layer1 = FCLayer::new(he_weights(n, (n) * 2), zero_biases((n) * 2), relu, relu_deriv);
    let mut layer2 = FCLayer::new(he_weights((n) * 2, n), zero_biases(n), relu, relu_deriv);

    /**
     * This test example's analogue to a loss function;
     * the distance between the network output and the 
     */
    fn mean_square_error(Y: &Array<f32>, y: &Array<f32>) -> f32 {
        let delta = pow(&(y - Y), &2.0_f32, false);
        let (mean, _) = mean_all(&delta);
        return mean as f32;
    };

    println!("Beginning training.");
    let epochs = 500;
    for i in 0..epochs {
        let l1o = layer1.apply(&input);
        let l2o = layer2.apply(&l1o);
        af_print!("Input, Output, Prediction:", join_many(1, vec!(&input, &l2o, &output)));
        let l2d = layer2.error_delta(&output);
        layer2.train(&l1o, &output);
        layer1.train(&input, &l2d);

        let mse = mean_square_error(&output, &l2o);
        println!("MSE: {}", mse);
    }
}

fn main() {
    // arrayfire_test();
    // fc_test();
    train_test();
}

//DOWNLOAD ARRAYFIRE: https://arrayfire.com/download/
