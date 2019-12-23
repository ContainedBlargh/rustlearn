extern crate arrayfire;
use arrayfire::*;

/**
 * Layer struct.
 * 
 * Makes sure that the datastructure can be treated as a Layer in a Neural Network.
 * This means that it needs to implement a training strategy.
 * 
 * In this case, it will be training using backpropagation and the adam optimizer.
 */
pub trait Layer {

    // Ensure that the following fields exist by enforcing getters to be implemented.
    fn learning_rate(self) -> f32;

    fn decay_rate(self) -> f32;

    fn squared_decay_rate(self) -> f32;

    fn epsilon(self) -> f32;

    fn timestep(self) -> u32;

    fn movement(self) -> Array<f32>;

    fn squared_movement(self) -> Array<f32>;

    /**
     * Apply the layer to a set of values and output the result.
     */
    fn apply(&mut self, x: &Array<f32>) -> Array<f32>;
        
    /**
     * Compute the gradients for the layer, using backpropagation.
     */
    fn error_delta(&mut self, target: &Array<f32>) -> Array<f32>;

    /**
     * Fit the layer using target values.
     * Returns the delta applied to layer (for use as previous layer targets).
     */
    fn train(&mut self, input: &Array<f32>, target: &Array<f32>);
}