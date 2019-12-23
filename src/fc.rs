extern crate arrayfire;

use arrayfire::*;

use utils::*;

pub struct FCLayer {
    w: Array<f32>,
    b: Array<f32>,
    a: fn(&Array<f32>) -> Array<f32>,
    d: fn(&Array<f32>) -> Array<f32>,
    recent_x: Option<Array<f32>>,
    recent_y: Option<Array<f32>>
}

use layer::Layer;

impl FCLayer {
    /**
     * Creates a new FCLayer using the given parameters.
     */
    pub fn new(weights: Array<f32>, biases: Array<f32>, activation: fn(&Array<f32>) -> Array<f32>, derivation: fn(&Array<f32>) -> Array<f32>) -> FCLayer {
        return FCLayer{
            w: weights, 
            b: biases, 
            a: activation, 
            d: derivation,
            recent_x: None,
            recent_y: None
        };
    }
}

impl Layer for FCLayer {

    fn apply(&mut self, x: &Array<f32>) -> Array<f32> {
        //TODO: rewrite the underneath as a helper function for easy array copy.
        self.recent_x = Some(t(x));
        let dotted = mdot(&t(x), &self.w);
        let added = dotted + &self.b;
        let activated = (self.a)(&added);
        self.recent_y = Some((self.a)(&added));
        return t(&activated);
    }

    fn gradient(&mut self, target: &Array<f32>) -> Array<f32> {
        return match (&self.recent_x, &self.recent_y) {
            (Some(x), Some(y)) => {
                let error = t(y) - target;
                af_print!("Error:", error);
                let delta = mdot(&error, &((self.d)(y)));
                af_print!("Delta:", error);
                let gt = mdot(&x, &delta);
                return t(&gt);
            },
            _ => constant(0_f32, target.dims())
        };
    }

    fn update(&mut self, modifier: &Array<f32>) {
        self.w -= modifier + constant(0, modifier.dims());
    }
}

pub fn init_random_fc(input_size: u64, output_size: u64) -> FCLayer {
    let w = he_weights(input_size, output_size);
    let b = zero_biases(output_size);
    return FCLayer::new(w, b, relu, relu_deriv);
}

pub fn init_identity_fc(input_size: u64, output_size: u64) -> FCLayer {
    let w = identity(d_matrix(input_size, output_size));
    let b = zero_biases(output_size);
    return FCLayer::new(w, b, relu, relu_deriv);
}
