use crate::matrix::*;
use crate::network;
use crate::prelude::*;
use crate::function::*;
use crate::layer::*;
use std::f32::MIN;

pub struct Network {
    num_layers: usize,
    layers: Vec<Rc<RefCell<Layer>>>,
    num_connections: usize,
    connections: Vec<Connection>
}

pub enum LossFunction {
    CrossEntropy,
    MeanSquaredError,
}

pub struct ParameterSet {
    dataset: Rc<RefCell<DataSet>>,
    classes: Rc<RefCell<DataSet>>,
    loss: LossFunction,
    batch_size: usize,
    learning_rate: f32,
    search_time: f32,
    regularization: f32,
    momentum: f32,
    max_iters: usize,
    shuffle: bool,
    verbose: bool
}

pub fn create_network(
    num_features: usize, 
    num_hidden_layers: usize, 
    hidden_sizes: Vec<usize>,
    hidden_activations: Vec<Option<Activation>>,
    num_outputs: usize,
    output_activation: Option<Activation>) -> Network {

    assert!(num_features > 0 && num_outputs > 0);
    let num_layers = num_hidden_layers + 2;
    let mut layers: Vec<Rc<RefCell<Layer>>> = Vec::new();
    for i in 0..num_layers {
        if (i == 0) {
            layers.push(Rc::new(RefCell::new(create_layer(LayerType::INPUT, num_features, None))));
        } else if (i == num_layers - 1)  {
            layers.push(Rc::new(RefCell::new(create_layer(LayerType::OUTPUT, num_outputs, output_activation))));
        } else {
            layers.push(Rc::new(RefCell::new(create_layer(LayerType::HIDDEN, hidden_sizes[i-1], hidden_activations[i-1]))));
        }
    }

    let num_connections = num_layers - 1;
    let mut connections: Vec<Connection> = Vec::new();
    for i in 0..num_connections {
        connections.push(create_connection(layers[i].clone(), layers[i+1].clone()));
        connections[i].init();
    }

    Network {num_layers, layers, num_connections, connections}
}

impl Network {
    pub fn forward_pass(&mut self, input: &Matrix) {
        {
            let mut input_layer = self.layers[0].borrow_mut();
            assert!(input.cols() == input_layer.input_cols());
            input_layer.input = input.clone();
        }
        let mut tmp;
        let mut tmp2;
        for i in 0..self.num_connections {
            tmp = self.layers[i].borrow().input.multiply(&self.connections[i].weights);
            tmp2 = tmp.add_to_each_row(&self.connections[i].bias);

            let mut layer = self.connections[i].to.borrow_mut();
            layer.input = tmp2;
            layer.activate();
        }
    }

    pub fn forward_pass_dataset(&mut self, input: Rc<RefCell<DataSet>>) {
        let matrix_data = input.borrow().to_matrix();
        self.forward_pass(&matrix_data);
    }

    pub fn cross_entropy_loss(&self, prediction: &Matrix, actual: Rc<RefCell<DataSet>>, regularization: f32) -> f32 {
        assert!(prediction.rows() == actual.borrow().rows());
        assert!(prediction.cols() == actual.borrow().cols());
        let mut total_err: f32 = 0.0;
        for i in 0..prediction.rows() {
            let mut cur_err: f32 = 0.0;
            for j in 0..prediction.cols() {
                cur_err += actual.borrow().data().borrow()[i][j] * f32::max(MIN, prediction.get(i, j)).ln();
            }

            total_err += cur_err;
        }

        let mut reg_err: f32 = 0.0;
        for i in 0..self.num_connections {
            let weights = &self.connections[i].weights;
            for j in 0..weights.rows() {
                for k in 0..weights.cols() {
                    reg_err += weights.get(j,k) * weights.get(j, k);
                }
            }
        }

        ((-1.0 / (actual.borrow().rows() as f32)) * total_err) + (regularization * 0.5 * reg_err)
    }

    pub fn mean_squared_error(&self, prediction: &Matrix, actual: Rc<RefCell<DataSet>>, regularization: f32) -> f32 {
        assert!(prediction.rows() == actual.borrow().rows());
        assert!(prediction.cols() == actual.borrow().cols());
        let mut total_err: f32 = 0.0;
        for i in 0..prediction.rows() {
            let mut cur_err: f32 = 0.0;
            for j in 0..prediction.cols() {
                let tmp = actual.borrow().data().borrow()[i][j] - prediction.get(i, j);
                cur_err += tmp;
            }

            total_err += cur_err;
        }

        let mut reg_err: f32 = 0.0;
        for i in 0..self.num_connections {
            let weights = &self.connections[i].weights;
            for j in 0..weights.rows() {
                for k in 0..weights.cols() {
                    reg_err += weights.get(j,k) * weights.get(j, k);
                }
            }
        }

        ((0.5 / (actual.borrow().rows() as f32)) * total_err) + (regularization * 0.5 * reg_err)
    }
    
    pub fn get_output(&self) -> Matrix {
        self.layers[self.num_layers-1].borrow().input.clone()
    }

    pub fn predict(&self) -> Vec<i32> {
        let mut max = 0;
        let output_layer = self.layers[self.num_layers-1].borrow();
        let mut predictions: Vec<i32> = Vec::new();
        for i in 0..output_layer.input.rows() {
            max = 0;
            for j in 1..output_layer.size {
                if (output_layer.input.get(i, j) > output_layer.input.get(i, max)) {
                    max = j;
                }
            }
            predictions.push(max as i32)
        }

        predictions
    }

    pub fn accuracy(&mut self, dataset: Rc<RefCell<DataSet>>, classes: Rc<RefCell<DataSet>>) -> f32 {
        assert!(dataset.borrow().rows() == classes.borrow().rows());
        assert!(classes.borrow().cols() == self.layers[self.num_layers-1].borrow().size);
        self.forward_pass_dataset(dataset.clone());
        let predictions = self.predict();
        let mut num_correct: f32 = 0.0;
        for i in 0..dataset.borrow().rows() {
            if (classes.borrow().data().borrow()[i][predictions[i] as usize] == 1.0) {
                num_correct = num_correct + 1.0;
            }
        }

        num_correct / (classes.borrow().rows() as f32)
    }

    pub fn batch_gradient_descent(&mut self, params: &mut ParameterSet) {
        assert!(self.layers[0].borrow().size == params.dataset.borrow().cols());
        assert!(params.dataset.borrow().rows() == params.classes.borrow().rows());
        assert!(self.layers[self.num_layers-1].borrow().size == params.classes.borrow().cols());
        assert!(params.batch_size <= params.dataset.borrow().rows());
        assert!(params.max_iters >= 1);

        let mut errori: Vec<Matrix> = Vec::new();
        let mut dWi: Vec<Matrix> = Vec::new();
        let mut dbi: Vec<Matrix> = Vec::new();
        let mut regi: Vec<Matrix> = Vec::new();
        let mut before_output_t = create_matrix_zeroes(self.layers[self.num_layers-2].borrow().size, 1);

        for i in 0..self.num_connections {
            errori.push(create_matrix_zeroes(1, self.layers[i].borrow().size));
            dWi.push(create_matrix_zeroes(self.connections[i].weights.rows(), self.connections[i].weights.cols()));
            dbi.push(create_matrix_zeroes(1, self.connections[i].bias.cols()));
            regi.push(create_matrix_zeroes(self.connections[i].weights.rows(), self.connections[i].weights.cols()));
        }

        errori.push(create_matrix_zeroes(1, self.layers[self.num_connections].borrow().size));

        let num_hidden = self.num_layers - 2;
        let mut WTi: Vec<Matrix> = Vec::new();
        let mut error_last_ti: Vec<Matrix> = Vec::new();
        let mut fprimei: Vec<Matrix> = Vec::new();
        let mut input_ti: Vec<Matrix> = Vec::new();
        if (num_hidden > 0) {
            for k in 0..num_hidden {
                WTi.push(create_matrix_zeroes(self.connections[k+1].weights.cols(), self.connections[k+1].weights.rows()));
                error_last_ti.push(create_matrix_zeroes(1, WTi[k].cols()));
                fprimei.push(create_matrix_zeroes(1, self.connections[k].to.borrow().size));
                input_ti.push(create_matrix_zeroes(self.connections[k].from.borrow().size, 1));
            }
        }

        let mut dWi_avg: Vec<Matrix> = Vec::new();
        let mut dbi_avg: Vec<Matrix> = Vec::new();
        let mut dWi_last: Vec<Matrix> = Vec::new();
        let mut dbi_last: Vec<Matrix> = Vec::new();
        for i in 0..self.num_connections {
            dWi_avg.push(create_matrix_zeroes(self.connections[i].weights.rows(), self.connections[i].weights.cols()));
            dbi_avg.push(create_matrix_zeroes(1, self.connections[i].bias.cols()));
            dWi_last.push(create_matrix_zeroes(self.connections[i].weights.rows(), self.connections[i].weights.cols()));
            dbi_last.push(create_matrix_zeroes(1, self.connections[i].bias.cols()));
        }

        let num_batches = params.dataset.borrow().rows() / params.batch_size + if (params.dataset.borrow().rows() % params.batch_size != 0) {1} else {0};
        let (mut training, mut batch, mut epoch, mut layer): (usize, usize, usize, usize) ;
        let mut batches: Vec<Batch> = Vec::new();
        let mut classBatches: Vec<DataSet> = Vec::new();
        epoch = 1;
        while epoch <= params.max_iters {
            if (params.shuffle) {
                shuffle_together(params.dataset.clone(), params.classes.clone());
            }

            let data_batches = create_batches(params.dataset.clone(), num_batches);
            let class_batches = create_batches(params.classes.clone(), num_batches);
            for batch in 0..num_batches {
                if (epoch > params.max_iters) { break; }  
                let cur_batch_size = if batch == num_batches - 1 {
                    if params.dataset.borrow().rows() % params.batch_size != 0 {
                        params.dataset.borrow().rows() % params.batch_size
                    } else {
                        params.batch_size
                    }
                } else {
                    params.batch_size
                };

                let batch_training = &data_batches[batch];
                let batch_classes = &class_batches[batch];
                let split_training = split_rows(batch_training);
                let split_classes = split_rows(batch_classes);

                for training in 0..cur_batch_size {
                    let row_idx = split_training[training].row_idx;
                    let row_example = params.dataset.borrow().data().borrow()[row_idx].clone();
                    let example = create_matrix(1, params.dataset.borrow().cols(), Rc::new(RefCell::new(row_example)));
                    let row_idx_target = split_classes[training].row_idx;
                    let row_target = params.classes.borrow().data().borrow()[row_idx_target].clone();
                    let target = create_matrix(1, params.dataset.borrow().cols(), Rc::new(RefCell::new(row_target)));
                    self.forward_pass(&example);

                    for layer in (self.num_layers-1)..0 {
                        let to = self.layers[layer].borrow();
                        let con = &self.connections[layer-1];
                        if (layer == self.num_layers - 1) {
                            to.input.copy_values_into(&mut errori[layer]);
                            match params.loss {
                                LossFunction::CrossEntropy => {
                                    for j in 0..errori[layer].cols() {
                                        errori[layer].data.borrow_mut()[j] -= target.data.borrow()[j];
                                    }
                                },
                                _ => {
                                    for j in 0..errori[layer].cols() {
                                        errori[layer].data.borrow_mut()[j] -= target.data.borrow()[j];
                                    }
                                }
                            }

                            con.from.borrow().input.transpose_into(&mut before_output_t);
                            before_output_t.multiply_into(&errori[layer], &mut dWi[layer-1]);
                            errori[layer].copy_values_into(&mut dbi[layer-1]);
                        }

                        else {
                            let hidden_layer = layer-1;
                            self.connections[layer].weights.transpose_into(&mut WTi[hidden_layer]);
                            errori[layer + 1].multiply_into(&WTi[hidden_layer], &mut error_last_ti[hidden_layer]);
                            con.to.borrow().input.copy_values_into(&mut fprimei[hidden_layer]);
                            let derivative = activation_derivative(con.to.borrow().activation);
                            for j in 0..fprimei[hidden_layer].cols() {
                                fprimei[hidden_layer].data.borrow_mut()[j] = derivative(fprimei[hidden_layer].data.borrow()[j]);
                            }
                            error_last_ti[hidden_layer].hadamard_into(&fprimei[hidden_layer], &mut errori[layer]);
                            con.from.borrow().input.transpose_into(&mut input_ti[hidden_layer]);
                            input_ti[hidden_layer].multiply_into(&errori[layer], &mut dWi[layer - 1]);
                            errori[layer].copy_values_into(&mut dbi[layer-1]);
                        }
                    }

                    for i in 0..self.num_connections {
                        dWi[i].add_to(&mut dWi_avg[i]);
                        dbi[i].add_to(&mut dbi_avg[i]);
                    }

                    before_output_t.to_zero();
                    for i in 0..self.num_connections {
                        errori[i].to_zero();
                        dWi[i].to_zero();
                        dbi[i].to_zero();
                    }

                    errori[self.num_connections].to_zero();
                    if (num_hidden > 0) {
                        for i in 0..num_hidden {
                            WTi[i].to_zero();
                            error_last_ti[i].to_zero();
                            fprimei[i].to_zero();
                            input_ti[i].to_zero();
                        }
                    }
                }

                let current_lr = if params.search_time == 0.0 { params.learning_rate } else { params.learning_rate / (1.0 + (epoch as f32 / params.search_time))};
                for i in 0..self.num_connections {
                    let lambda = current_lr / params.dataset.borrow().rows() as f32;
                    dWi_avg[i].scalar_multiply(lambda);
                    dbi_avg[i].scalar_multiply(lambda);
                }

                for i in 0..self.num_connections {
                    dWi_last[i].scalar_multiply(params.momentum);
                    dbi_last[i].scalar_multiply(params.momentum);
                    dWi_last[i].add_to(&mut dWi_avg[i]);
                    dbi_last[i].add_to(&mut dbi_avg[i]);
                }

                for i in 0..self.num_connections {
                    dWi_avg[i].scalar_multiply(-1.0);
                    dbi_avg[i].scalar_multiply(-1.0);
                    dWi_avg[i].add_to(&mut self.connections[i].weights);
                    dbi_avg[i].add_to(&mut self.connections[i].bias);
                }

                for i in 0..self.num_connections {
                    dWi_avg[i].copy_values_into(&mut dWi_last[i]);
                    dbi_avg[i].copy_values_into(&mut dbi_last[i]);
                    dWi_last[i].scalar_multiply(-1.0);
                    dbi_last[i].scalar_multiply(-1.0);
                }

                for i in 0..self.num_connections {
                    dWi_avg[i].to_zero();
                    dbi_avg[i].to_zero();
                    regi[i].to_zero();
                }

                if (params.verbose) {
                    if (epoch % 10 == 0 || epoch == 1) {
                        self.forward_pass_dataset(params.dataset.clone());
                        match params.loss {
                            LossFunction::CrossEntropy => println!("{:?}, {:?}", epoch, self.cross_entropy_loss(&self.get_output(), params.classes.clone(), params.regularization)),
                            _ => println!("{:?}, {:?}", epoch, self.mean_squared_error(&self.get_output(), params.classes.clone(), params.regularization))
                        }
                    }
                }
            }
        }

    }
    
}
