use crate::matrix::*;
use crate::prelude::*;
use crate::function::*;

pub enum LayerType {
    INPUT,
    HIDDEN,
    OUTPUT
}

pub struct Layer {
    pub layer_type: LayerType,
    pub size: usize,
    pub activation: Option<Activation>,
    pub input: Matrix
}

pub struct Connection {
    pub from: Rc<RefCell<Layer>>,
    pub to: Rc<RefCell<Layer>>,
    pub weights: Matrix,
    pub bias: Matrix
}

pub fn create_layer(layer_type: LayerType, size: usize, activation: Option<Activation>) -> Layer {
    let row: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(vec![0.0; size]));
    let input = create_matrix(1, size, row);
    Layer {layer_type, size, activation, input}
}

pub fn create_connection(from: Rc<RefCell<Layer>>, to: Rc<RefCell<Layer>>) -> Connection {
    let to_size = to.borrow().size;
    let from_size = from.borrow().size;
    let weights_data: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(vec![0.0; to_size*from_size]));
    let bias_data: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(vec![0.0; to_size]));
    let weights = create_matrix(from_size, to_size, weights_data);
    let bias = create_matrix(1, to_size, bias_data);
    Connection {from, to, weights, bias}
}

impl Layer {
    pub fn activate(&mut self) {
        match self.activation {
            None => {},
            Some(activation) => (activation)(&mut self.input)
        }
    }

    pub fn input_cols(&self) -> usize {
        self.input.cols()
    }

    pub fn input_rows(&self) -> usize {
        self.input.rows()
    }

    pub fn set_input(&mut self, input: Matrix) {
        self.input = input;
    }
}

impl Connection {
    pub fn init(&mut self) {
        self.bias.to_zero();

        let neurons_in = (self.weights.rows() as f32).sqrt();
        self.weights.transform(|x| box_muller(x) / neurons_in);
    }
}