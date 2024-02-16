use crate::prelude::*;
use crate::dataset::*;

pub type Matrix = DataSet;

impl Matrix {

    pub fn create_matrix(rows: usize, cols: usize, data: Rc<RefCell<Vec<Vec<f32>>>>) -> Matrix {
        assert!(rows > 0 && cols > 0);
        Matrix {rows, cols, data}
    }
    
    pub fn create_matrix_zeroes(rows: usize, cols: usize) -> Matrix {
        assert!(rows > 0 && cols > 0);
        Matrix {rows, cols, data: Rc::new(RefCell::new(vec![vec![0.0; cols]; rows]))}
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data.borrow()[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data.borrow_mut()[row * self.cols + col] = val;
    }

    pub fn to_zero(&mut self) {
        self.data.borrow_mut().fill(0.0);
    }

    pub fn transform<F>(&mut self, mut func: F)
    where F: FnMut(f32) -> f32 {
            for x in self.data.borrow_mut().iter_mut() {
                *x = func(x.clone());
        }
    }

    pub fn copy_values_into(&self, to: &mut Matrix) {
        assert!(self.rows == to.rows && self.cols == to.cols);
        to.data.borrow_mut().copy_from_slice(&self.data.borrow());
    }

    pub fn transpose(&self) -> Matrix {
        let data: Vec<f32> = vec![0.0; self.cols*self.rows];
        let mut result: Matrix = create_matrix(self.cols, self.rows, Rc::clone(&self.data));
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set( i, j, self.get( i, j));
            }
        }
    
        result
    }

    pub fn transpose_into(&self, into: &mut Matrix) {
        assert!(self.rows == into.cols && self.cols == into.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                into.set(j, i, self.get(i, j));
            }
        }
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert!(self.rows == other.rows && self.cols == other.cols);
        let data: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(vec![0.0; self.cols*self.rows]));
        let mut result: Matrix = create_matrix(self.cols, self.rows, data);
        for i in 0..self.rows {
            for j in 0..self.rows {
                let val = self.get(i, j) + other.get(i, j);
                result.set(i, j, val);
            }
        }
    
        result
    }

    pub fn add_to(&self, to: &mut Matrix) {
        assert!(self.rows == to.rows && self.cols == to.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = self.get(i, j) + to.get(i, j);
                to.set(i, j, val);
            }
        }
    }

    pub fn add_to_each_row(&self, other: &Matrix) -> Matrix {
        assert!(self.cols == other.cols && other.rows == 1);
        let data: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(vec![0.0; self.cols*self.rows]));
        let mut result: Matrix = create_matrix(self.cols, self.rows, data);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = self.get(i, j) + other.get(0, j);
                result.set(i, j, val);
            }
        }
    
        result
    }

    pub fn scalar_multiply(&mut self, k: f32) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = self.get(i, j) * k;
                self.set(i, j, val);
            }
        }
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        assert!(self.cols == other.rows);
        let data: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(vec![0.0; other.cols*self.rows]));
        let mut result: Matrix = create_matrix(self.rows, other.cols, data);
    
        result.to_zero();
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum: f32 = 0.0;
                for k in 0..other.rows {
                    sum += self.get(i, k) * other.get(k, j);
                }
    
                result.set(i, j, sum);
            }
        }
    
        result
    }

    pub fn multiply_into(&self, other: &Matrix, into: &mut Matrix) {
        assert!(self.cols == other.rows);
        assert!(self.rows == into.rows && other.cols == into.cols);

        into.to_zero();
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum:f32 = 0.0;
                for k in 0..other.rows {
                    sum += self.get(i, j) * other.get(k, j);
                }

                into.set(i, j, sum);
            }
        }
    }

    pub fn hadamard(&self, other: &Matrix) -> Matrix {
        assert!(self.rows == other.rows && self.cols == other.cols);
        let data: Vec<f32> = vec![0.0; self.cols*self.rows];
        let mut result: Matrix = create_matrix(self.cols, self.rows, Rc::clone(&self.data));
        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = self.get(i, j) * other.get(i, j);
                result.set(i, j, val);
            }
        }
        result
    }

    pub fn hadamard_into(&self, other: &Matrix, into: &mut Matrix) {
        assert!(self.rows == other.rows && self.cols == other.cols);
        assert!(self.rows == into.rows && self.cols == into.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = self.get(i, j) * other.get(i, j);
                into.set(i, j, val);
            }
        }
    }

    pub fn copy(&self) -> Matrix {
        let mut data: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(vec![0.0; self.cols*self.rows]));
        data.borrow_mut().copy_from_slice(&self.data.borrow());
        Matrix {rows: self.rows, cols: self.cols, data}
    }

    pub fn equals(&self, other: &Matrix) -> bool {
        if (self.rows != other.rows) {
            return false;
        }
        
        if (self.cols != other.cols) {
            return false;
        }

        for i in 0..self.rows {
            for j in 0..other.cols {
                if (self.get(i, j) != other.get(i, j)) {
                    return false;
                }
            }
        }

        true
    }

}