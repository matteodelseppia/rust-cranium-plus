use crate::matrix::*;
use crate::prelude::*;
use crate::function::*;

#[derive(Clone)]
pub enum LayerType {
    INPUT,
    HIDDEN,
    OUTPUT
}

#[derive(Clone)]
pub struct Layer {
    pub layer_type: LayerType,
    pub size: usize,
    pub activation: Option<Activation>,
    pub input: Rc<RefCell<Matrix>>
}

pub struct Connection {
    pub from: Layer,
    pub to: Layer,
    pub weights: Matrix,
    pub bias: Matrix
}

pub fn create_layer(layer_type: LayerType, size: usize, activation: Option<Activation>) -> Layer {
    let row = vec![vec![0.0 as f32; size]; 1];
    let input: Matrix = Matrix::create_matrix(1, size, row);
    Layer {layer_type, size, activation, input: Rc::new(RefCell::new(input))}
}

pub fn create_connection(from: &Layer, to: &Layer) -> Connection {
    let to_size = to.size;
    let from_size = from.size;
    let weights_data = vec![vec![0.0 as f32; to_size]; from_size];
    let bias_data = vec![vec![0.0 as f32; to_size]; 1];
    let weights = Matrix::create_matrix(from_size, to_size, weights_data);
    let bias = Matrix::create_matrix(1, to_size, bias_data);
    Connection {from: from.clone(), to: to.clone(), weights, bias}
}

impl Layer {
    pub fn activate(&mut self) {
        match self.activation {
            None => {},
            Some(activation) => activation(self.input.clone())
        }
    }

    pub fn set_input(&mut self, input: Rc<RefCell<Matrix>>) {
        self.input = input;
    }
}

impl Connection {
    pub fn init(&mut self) {
        self.bias.to_zero();

        let neurons_in = (self.weights.rows as f32).sqrt();
        self.weights.transform(|x| box_muller(x) / neurons_in);
    }
}