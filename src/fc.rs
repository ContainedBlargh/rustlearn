extern crate arrayfire;

use arrayfire::*;

use utils::*;

pub struct FCLayer {
    w: Array<f32>,
    b: Array<f32>,
    a: fn(&Array<f32>) -> Array<f32>,
    d: fn(&Array<f32>) -> Array<f32>,
    recent_x: Option<Array<f32>>,
    recent_y: Option<Array<f32>>,
    lr: f32,
    dr: f32,
    sdr: f32,
    e: f32,
    t: u32,
    m: Array<f32>,
    v: Array<f32>, //squared_movement, but we adhere to the paper's naming convention.
}

use layer::Layer;

impl FCLayer {
    /**
     * Creates a new FCLayer using the given parameters.
     */
    pub fn new(
        weights: Array<f32>,
        biases: Array<f32>,
        activation: fn(&Array<f32>) -> Array<f32>,
        derivation: fn(&Array<f32>) -> Array<f32>,
    ) -> FCLayer {
        let dims = biases.dims();
        return FCLayer {
            w: weights,
            b: biases,
            a: activation,
            d: derivation,
            recent_x: None,
            recent_y: None,
            lr: 0.001,
            dr: 0.9,
            sdr: 0.999,
            e: 1E-8,
            t: 0,
            m: constant(0.0_f32, dims),
            v: constant(0.0_f32, dims)
        };
    }
}

impl Layer for FCLayer {
    fn learning_rate(self) -> f32 {
        return self.lr;
    }
    fn decay_rate(self) -> f32 {
        return self.dr;
    }
    fn squared_decay_rate(self) -> f32 {
        return self.sdr;
    }
    fn epsilon(self) -> f32 {
        return self.e;
    }
    fn timestep(self) -> u32 {
        return self.t;
    }
    fn movement(self) -> Array<f32> {
        return self.m;
    }
    fn squared_movement(self) -> Array<f32> {
        return self.v;
    }

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
            }
            _ => constant(0_f32, target.dims()),
        };
    }

    /**
     * Trains the layer using the Adam Opitmizer strategy.
     * 
     * See Kingma & Ba, Adam, Algorithm 1.
     */
    fn train(&mut self, target: &Array<f32>) {
        let gt = self.gradient(target);
        self.t += 1;
        self.m = &gt * (&self.m * self.dr) + (1.0_f32 - self.dr);
        self.v = pow(&gt, &2.0_f32, false) * (&self.v * self.sdr) + (1.0_f32 - self.sdr);
        let bcm = &self.m / (1.0_f32 - self.dr.powf(self.t as f32));
        let bcv = &self.v / (1.0_f32 - self.sdr.powf(self.t as f32));
        let delta = (bcm / (sqrt(&bcv) + self.e)) * self.lr;
        self.w = &self.w - delta;
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
