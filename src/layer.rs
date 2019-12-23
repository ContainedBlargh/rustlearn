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
    fn gradient(&mut self, target: &Array<f32>) -> Array<f32>;

    /**
     * Fit the layer using loss.
     */
    fn train(&mut self, target: &Array<f32>);
}