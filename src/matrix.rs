use crate::prelude::*;
use crate::dataset::*;

pub type Matrix = DataSet;

impl Matrix {

    pub fn create_matrix(rows: usize, cols: usize, data: Vec<Vec<f32>>) -> Matrix {
        assert!(rows > 0 && cols > 0);
        Matrix {rows, cols, data}
    }

    pub fn create_zero_matrix(rows: usize, cols: usize) -> Matrix {
        assert!(rows > 0 && cols > 0);
        Matrix {rows, cols, data: vec![vec![0.0; cols]; rows]}
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row][col] = val;
    }

    pub fn to_zero(&mut self) {
        for z in self.data.iter_mut() {
            z.fill(0.0);
        }
    }

    pub fn transform<F>(&mut self, mut func: F)
    where F: FnMut(f32) -> f32 {
            for row in self.data.iter_mut() {
                for x in row.iter_mut() {
                    *x = func(x.clone());
                }
        }
    }

    pub fn copy(&self) -> Matrix {
        let mut data = vec![vec![0.0 as f32; self.cols]; self.rows];
        for (idx, row) in data.iter_mut().enumerate() {
            row.copy_from_slice(&self.data[idx]);
        }

        Matrix {rows: self.rows, cols: self.cols, data}
    }

    pub fn copy_into(&self, to: &mut Matrix) {
        assert!(self.rows == to.rows && self.cols == to.cols);
        for (idx, row) in to.data.iter_mut().enumerate() {
            row.copy_from_slice(&self.data[idx]);
        }
    }

    pub fn transpose(&self) -> Matrix {
        let data: Vec<f32> = vec![0.0; self.cols*self.rows];
        let mut result: Matrix = Matrix::create_zero_matrix(self.cols, self.rows);
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
        let mut result: Matrix = Matrix::create_zero_matrix(self.cols, self.rows);
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
        let mut result: Matrix = Matrix::create_zero_matrix(self.cols, self.rows);
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
        let mut result: Matrix = Matrix::create_zero_matrix(self.rows, other.cols);
    
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
        let mut result: Matrix = Matrix::create_zero_matrix(self.cols, self.rows);
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