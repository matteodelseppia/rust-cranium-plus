use rand::Rng;
use crate::matrix::*;
use crate::prelude::*;

pub type Activation = fn(&mut Matrix);

pub fn sigmoid_func(input: f32) -> f32 {
    1.0 / (1.0 + (-1.0 * input).exp())
}

pub fn sigmoid_deriv(sigmoid_input: f32) -> f32 {
    sigmoid_input * (1.0 - sigmoid_input)
}

pub fn relu_func(input: f32) -> f32 {
    input.max(0.0)
}

pub fn relu_deriv(relu_input: f32) -> f32 {
    if relu_input > 0.0 { 1.0 } else { 0.0 }
}

pub fn sigmoid(input: &mut Matrix) {
    input.transform(|x| sigmoid_func(x));
}

pub fn tanh_func(input: f32) -> f32 {
    input.tanh()
}

pub fn tanh_deriv(tanh_input: f32) -> f32 {
    1.0 - (tanh_input * tanh_input)
}

pub fn relu(input: &mut Matrix) {
    input.transform(|x| relu_func(x));
}

pub fn tanh(input: &mut Matrix) {
    input.transform(|x| tanh_func(x));
}

pub fn softmax(input: &mut Matrix) {
    for i in 0..input.rows() {
        let mut summed = 0.0;
        for j in 0..input.cols() {
            summed += input.get(i, j).exp();
        }
        for j in 0..input.cols() {
            input.set(i, j, input.get(i, j).exp() / summed);
        }
    }
}

pub fn linear(input: &mut Matrix) {}

pub fn linear_deriv(_linear_input: f32) -> f32 {
    1.0
}

pub fn box_muller(x: f32) -> f32 {
    const EPSILON: f32 = f32::MIN;
    const TWO_PI: f32 = 2.0 * std::f32::consts::PI;
    static mut Z0: f32 = 0.0;
    static mut Z1: f32 = 0.0;
    static mut GENERATE: bool = true;
    
    unsafe {
        GENERATE = !GENERATE;
        if !GENERATE {
            return Z1;
        }
        let mut u1;
        let mut u2;
        loop {
            u1 = rand::random::<f32>() * (1.0 / f32::MAX);
            u2 = rand::random::<f32>() * (1.0 / f32::MAX);
            if u1 > EPSILON {
                break;
            }
        }
        let z0 = (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos();
        Z1 = (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).sin();
        Z0 = z0;
        z0
    }
}

pub fn get_function_name(func: Activation) -> &'static str {
    match func {
        sigmoid => "sigmoid",
        relu => "relu",
        tanh => "tanh",
        softmax => "softmax",
        _ => "linear",
    }
}

pub fn get_function_by_name(name: &str) -> Activation {
    match name {
        "sigmoid" => sigmoid,
        "relu" => relu,
        "tanh" => tanh,
        "softmax" => softmax,
        _ => linear,
    }
}

pub fn activation_derivative(func: Option<Activation>) -> fn(f32) -> f32 {
    match func.unwrap() {
        sigmoid => sigmoid_deriv,
        relu => relu_deriv,
        tanh => tanh_deriv,
        _ => linear_deriv,
    }
}
