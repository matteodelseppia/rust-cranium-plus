use crate::matrix::*;
use crate::dataset::*;
use crate::prelude::*;
use crate::function::*;
use crate::layer::*;
use std::f32::MIN;

pub struct Network {
    num_layers: usize,
    layers: Vec<Layer>,
    num_connections: usize,
    connections: Vec<Connection>
}

pub enum LossFunction {
    CrossEntropy,
    MeanSquaredError,
}

pub struct ParameterSet {
    dataset: DataSet,
    classes: DataSet,
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
    let mut layers = Vec::new();
    for i in 0..num_layers {
        if (i == 0) {
            layers.push(create_layer(LayerType::INPUT, num_features, None));
        } else if (i == num_layers - 1)  {
            layers.push(create_layer(LayerType::OUTPUT, num_outputs, output_activation));
        } else {
            layers.push(create_layer(LayerType::HIDDEN, hidden_sizes[i-1], hidden_activations[i-1]));
        }
    }

    let num_connections = num_layers - 1;
    let mut connections: Vec<Connection> = Vec::new();
    for i in 0..num_connections {
        connections.push(create_connection(&layers[i], &layers[i+1]));
        connections[i].init();
    }

    Network {num_layers, layers, num_connections, connections}
}

impl Network {
    pub fn forward_pass(&mut self, input: Rc<RefCell<Matrix>>) {
        {
            let mut input_layer = &mut self.layers[0];
            assert!(input.borrow().cols == input_layer.input.borrow().cols);
            input_layer.input = input;
        }

        let mut tmp;
        let mut tmp2;
        for i in 0..self.num_connections {
            tmp = self.layers[i].input.borrow().multiply(&self.connections[i].weights);
            tmp2 = tmp.add_to_each_row(&self.connections[i].bias);

            let mut layer = &mut self.connections[i].to;
            layer.input = Rc::new(RefCell::new(tmp2));
            layer.activate();
        }
    }

    pub fn cross_entropy_loss(&self, prediction: &Matrix, actual: Rc<RefCell<DataSet>>, regularization: f32) -> f32 {
        assert!(prediction.rows == actual.borrow().rows);
        assert!(prediction.cols == actual.borrow().cols);
        let mut total_err: f32 = 0.0;
        for i in 0..prediction.rows {
            let mut cur_err: f32 = 0.0;
            for j in 0..prediction.cols {
                cur_err += actual.borrow().data[i][j] * f32::max(MIN, prediction.get(i, j)).ln();
            }

            total_err += cur_err;
        }

        let mut reg_err: f32 = 0.0;
        for i in 0..self.num_connections {
            let weights = &self.connections[i].weights;
            for j in 0..weights.rows {
                for k in 0..weights.cols {
                    reg_err += weights.get(j,k) * weights.get(j, k);
                }
            }
        }

        ((-1.0 / (actual.borrow().rows as f32)) * total_err) + (regularization * 0.5 * reg_err)
    }

    pub fn mean_squared_error(&self, prediction: &Matrix, actual: Rc<RefCell<DataSet>>, regularization: f32) -> f32 {
        assert!(prediction.rows == actual.borrow().rows);
        assert!(prediction.cols == actual.borrow().cols);
        let mut total_err: f32 = 0.0;
        for i in 0..prediction.rows {
            let mut cur_err: f32 = 0.0;
            for j in 0..prediction.cols {
                let tmp = actual.borrow().data[i][j] - prediction.get(i, j);
                cur_err += tmp;
            }

            total_err += cur_err;
        }

        let mut reg_err: f32 = 0.0;
        for i in 0..self.num_connections {
            let weights = &self.connections[i].weights;
            for j in 0..weights.rows {
                for k in 0..weights.cols {
                    reg_err += weights.get(j,k) * weights.get(j, k);
                }
            }
        }

        ((0.5 / (actual.borrow().rows as f32)) * total_err) + (regularization * 0.5 * reg_err)
    }
    
    pub fn get_output(&self) -> Rc<RefCell<Matrix>> {
        self.layers[self.num_layers-1].input.clone()
    }

    pub fn predict(&mut self) -> Vec<i32> {
        let mut max = 0;
        let output_layer = &mut self.layers[self.num_layers-1];
        let mut predictions: Vec<i32> = Vec::new();
        for i in 0..output_layer.input.borrow().rows {
            max = 0;
            for j in 1..output_layer.size {
                if (output_layer.input.borrow().get(i, j) > output_layer.input.borrow().get(i, max)) {
                    max = j;
                }
            }
            predictions.push(max as i32)
        }

        predictions
    }

    pub fn accuracy(&mut self, dataset: Rc<RefCell<Matrix>>, classes: Rc<RefCell<Matrix>>) -> f32 {
        assert!(dataset.borrow().rows == classes.borrow().rows);
        assert!(classes.borrow().cols == self.layers[self.num_layers-1].size);
        self.forward_pass(dataset.clone());
        let predictions = self.predict();
        let mut num_correct: f32 = 0.0;
        for i in 0..dataset.borrow().rows {
            if (classes.borrow().data[i][predictions[i] as usize] == 1.0) {
                num_correct = num_correct + 1.0;
            }
        }

        num_correct / (classes.borrow().rows as f32)
    }

    pub fn batch_gradient_descent(&mut self, params: &mut ParameterSet) {
        assert!(self.layers[0].size == params.dataset.cols);
        assert!(params.dataset.rows == params.classes.rows);
        assert!(self.layers[self.num_layers-1].size == params.classes.cols);
        assert!(params.batch_size <= params.dataset.rows);
        assert!(params.max_iters >= 1);

        let mut errori: Vec<Matrix> = Vec::new();
        let mut dWi: Vec<Matrix> = Vec::new();
        let mut dbi: Vec<Matrix> = Vec::new();
        let mut regi: Vec<Matrix> = Vec::new();
        let mut before_output_t = Matrix::create_zero_matrix(self.layers[self.num_layers-2].size, 1);

        for i in 0..self.num_connections {
            errori.push(Matrix::create_zero_matrix(1, self.layers[i].size));
            dWi.push(Matrix::create_zero_matrix(self.connections[i].weights.rows, self.connections[i].weights.cols));
            dbi.push(Matrix::create_zero_matrix(1, self.connections[i].bias.cols));
            regi.push(Matrix::create_zero_matrix(self.connections[i].weights.rows, self.connections[i].weights.cols));
        }

        errori.push(Matrix::create_zero_matrix(1, self.layers[self.num_connections].size));

        let num_hidden = self.num_layers - 2;
        let mut WTi: Vec<Matrix> = Vec::new();
        let mut error_last_ti: Vec<Matrix> = Vec::new();
        let mut fprimei: Vec<Matrix> = Vec::new();
        let mut input_ti: Vec<Matrix> = Vec::new();
        if (num_hidden > 0) {
            for k in 0..num_hidden {
                WTi.push(Matrix::create_zero_matrix(self.connections[k+1].weights.cols, self.connections[k+1].weights.rows));
                error_last_ti.push(Matrix::create_zero_matrix(1, WTi[k].cols));
                fprimei.push(Matrix::create_zero_matrix(1, self.connections[k].to.size));
                input_ti.push(Matrix::create_zero_matrix(self.connections[k].from.size, 1));
            }
        }

        let mut dWi_avg: Vec<Matrix> = Vec::new();
        let mut dbi_avg: Vec<Matrix> = Vec::new();
        let mut dWi_last: Vec<Matrix> = Vec::new();
        let mut dbi_last: Vec<Matrix> = Vec::new();
        for i in 0..self.num_connections {
            dWi_avg.push(Matrix::create_zero_matrix(self.connections[i].weights.rows, self.connections[i].weights.cols));
            dbi_avg.push(Matrix::create_zero_matrix(1, self.connections[i].bias.cols));
            dWi_last.push(Matrix::create_zero_matrix(self.connections[i].weights.rows, self.connections[i].weights.cols));
            dbi_last.push(Matrix::create_zero_matrix(1, self.connections[i].bias.cols));
        }

        let num_batches = params.dataset.rows / params.batch_size + if (params.dataset.rows % params.batch_size != 0) {1} else {0};
        let (mut training, mut batch, mut epoch, mut layer): (usize, usize, usize, usize) ;
        let mut batches: Vec<Batch> = Vec::new();
        let mut classBatches: Vec<DataSet> = Vec::new();
        epoch = 1;
        while epoch <= params.max_iters {
            if (params.shuffle) {
                shuffle_together(&mut params.dataset, &mut params.classes);
            }

            let data_batches = create_batches(&params.dataset, num_batches);
            let class_batches = create_batches(&params.classes, num_batches);
            for batch in 0..num_batches {
                if (epoch > params.max_iters) { break; }  
                let cur_batch_size = if batch == num_batches - 1 {
                    if params.dataset.rows % params.batch_size != 0 {
                        params.dataset.rows % params.batch_size
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
                    let row_example = vec![params.dataset.data[row_idx].clone(); 1];
                    let example: Matrix = Matrix::create_matrix(1, params.dataset.cols, row_example);
                    let row_idx_target = split_classes[training].row_idx;
                    let row_target = vec![params.classes.data[row_idx_target].clone(); 1];
                    let target = Matrix::create_matrix(1, params.dataset.cols, row_target);
                    self.forward_pass(Rc::new(RefCell::new(example)));

                    for layer in (self.num_layers-1)..0 {
                        let to = &mut self.layers[layer];
                        let con = &mut self.connections;
                        if (layer == self.num_layers - 1) {
                            to.input.borrow().copy_into(&mut errori[layer]);
                            match params.loss {
                                LossFunction::CrossEntropy => {
                                    for i in 0..errori[layer].rows {
                                        for j in 0..errori[layer].cols {
                                            errori[layer].data[i][j] -= target.data[i][j];
                                        }
                                    }
                                },
                                _ => {
                                        for i in 0..errori[layer].rows {
                                            for j in 0..errori[layer].cols {
                                                errori[layer].data[i][j] -= target.data[i][j];
                                            }
                                        }
                                }
                            }

                            con[layer-1].from.input.borrow().transpose_into(&mut before_output_t);
                            before_output_t.multiply_into(&errori[layer], &mut dWi[layer-1]);
                            errori[layer].copy_into(&mut dbi[layer-1]);
                        }

                        else {
                            let hidden_layer = layer-1;
                            con[layer].weights.transpose_into(&mut WTi[hidden_layer]);
                            errori[layer + 1].multiply_into(&WTi[hidden_layer], &mut error_last_ti[hidden_layer]);
                            con[layer-1].to.input.borrow().copy_into(&mut fprimei[hidden_layer]);
                            let derivative = activation_derivative(con[layer-1].to.activation);
                            for i in 0..fprimei[hidden_layer].rows {
                                for j in 0..fprimei[hidden_layer].cols {
                                    fprimei[hidden_layer].data[i][j] = derivative(fprimei[hidden_layer].data[i][j]);
                                }
                            }
                            error_last_ti[hidden_layer].hadamard_into(&fprimei[hidden_layer], &mut errori[layer]);
                            con[layer-1].from.input.borrow().transpose_into(&mut input_ti[hidden_layer]);
                            input_ti[hidden_layer].multiply_into(&errori[layer], &mut dWi[layer - 1]);
                            errori[layer].copy_into(&mut dbi[layer-1]);
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
                    let lambda = current_lr / params.dataset.rows as f32;
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
                    dWi_avg[i].copy_into(&mut dWi_last[i]);
                    dbi_avg[i].copy_into(&mut dbi_last[i]);
                    dWi_last[i].scalar_multiply(-1.0);
                    dbi_last[i].scalar_multiply(-1.0);
                }

                for i in 0..self.num_connections {
                    dWi_avg[i].to_zero();
                    dbi_avg[i].to_zero();
                    regi[i].to_zero();
                }
            }
        }

    }
    
}
