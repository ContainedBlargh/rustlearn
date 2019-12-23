extern crate arrayfire;
use arrayfire::*;

pub trait Layer {
    fn apply(&mut self, x: &Array<f32>) -> Array<f32>;
    fn gradient(&mut self, target: &Array<f32>) -> Array<f32>;
    fn update(&mut self, modifier: &Array<f32>);
}