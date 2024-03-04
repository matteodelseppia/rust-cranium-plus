#[cfg(test)]
mod matrix_tests {
    use cranium_rs::matrix::*;

    #[test]
    fn test_create_matrix() {
        let rows = 3;
        let cols = 2;
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix = Matrix::create_matrix(rows, cols, data.clone());

        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);
        assert_eq!(matrix.data, data);
    }

    #[test]
    fn test_create_zero_matrix() {
        let rows = 3;
        let cols = 2;
        let data = vec![vec![0.0; cols]; rows];
        let matrix = Matrix::create_matrix(rows, cols, data.clone());

        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);
        assert_eq!(matrix.data, data);
    }

    #[test]
    fn test_get() {
        let rows = 3;
        let cols = 2;
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix = Matrix::create_matrix(rows, cols, data);

        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(0, 1), 2.0);
        assert_eq!(matrix.get(1, 0), 3.0);
        assert_eq!(matrix.get(1, 1), 4.0);
        assert_eq!(matrix.get(2, 0), 5.0);
        assert_eq!(matrix.get(2, 1), 6.0);

    }

    #[test]
    fn test_set() {
        let rows = 3;
        let cols = 2;
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut matrix = Matrix::create_matrix(rows, cols, data);

        matrix.set(0, 1, 10.0);
        matrix.set(1, 1, 11.0);
        matrix.set(2, 1, 12.0);
        
        assert_eq!(matrix.get(0, 1), 10.0);
        assert_eq!(matrix.get(1, 1), 11.0);
        assert_eq!(matrix.get(2, 1), 12.0);

    }

    #[test]
    fn test_to_zero() {
        let rows = 3;
        let cols = 2;
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut matrix = Matrix::create_matrix(rows, cols, data);

        matrix.to_zero();

        let data_zero = vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]];
        
        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);
        assert_eq!(matrix.data, data_zero);

    }

    #[test]
    fn test_copy() {
        let rows = 3;
        let cols = 2;
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix = Matrix::create_matrix(rows, cols, data);

        let copied_matrix = matrix.copy();

        assert_eq!(matrix.rows, copied_matrix.rows);
        assert_eq!(matrix.cols, copied_matrix.cols);
        assert_eq!(matrix.data, copied_matrix.data);

    }

    #[test]
    fn test_copy_into() {
        let rows = 3;
        let cols = 2;
        let data_from = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let from = Matrix::create_matrix(rows, cols, data_from.clone());

        let data_to = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let mut to = Matrix::create_matrix(rows, cols, data_to);
        
        from.copy_into(&mut to);

        assert_eq!(to.data, data_from);
    }

    #[test]
    fn test_transpose() {
        let rows = 3;
        let cols = 2;
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix = Matrix::create_matrix(rows, cols, data);

        let transpoded_matrix = matrix.transpose();

        let transpoded_data = vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]];

        assert_eq!(transpoded_matrix.data, transpoded_data);

    }

    #[test]
    fn test_transpose_into() {
        let rows = 3;
        let cols = 2;
        let data1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let from = Matrix::create_matrix(rows, cols, data1);

        let data2 = vec![vec![7.0, 9.0, 11.0], vec![8.0, 10.0, 12.0]];
        let mut to = Matrix::create_matrix(cols, rows, data2);

        from.transpose_into(&mut to);

        let transpoded_data = vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]];

        assert_eq!(to.data, transpoded_data);

    }

    #[test]
    fn test_add() {
        let rows = 3;
        let cols = 2;
        let data1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix1 = Matrix::create_matrix(rows, cols, data1);

        let data2 = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let matrix2 = Matrix::create_matrix(rows, cols, data2);

        let matrix_sum = matrix1.add(&matrix2);

        assert_eq!(matrix_sum.get(0, 0), 8.0);
        assert_eq!(matrix_sum.get(0, 1), 10.0);
        assert_eq!(matrix_sum.get(1, 0), 12.0);
        assert_eq!(matrix_sum.get(1, 1), 14.0);
        assert_eq!(matrix_sum.get(2, 0), 16.0);
        assert_eq!(matrix_sum.get(2, 1), 18.0);

    }

    #[test]
    fn test_add_to() {
        let rows = 3;
        let cols = 2;
        let data1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix1 = Matrix::create_matrix(rows, cols, data1);

        let data2 = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let mut matrix2 = Matrix::create_matrix(rows, cols, data2);

        matrix1.add_to(&mut matrix2);

        assert_eq!(matrix2.get(0, 0), 8.0);
        assert_eq!(matrix2.get(0, 1), 10.0);
        assert_eq!(matrix2.get(1, 0), 12.0);
        assert_eq!(matrix2.get(1, 1), 14.0);
        assert_eq!(matrix2.get(2, 0), 16.0);
        assert_eq!(matrix2.get(2, 1), 18.0);

    }

    #[test]
    fn test_add_to_each_row() {
        let rows = 3;
        let cols = 2;
        let data1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix1 = Matrix::create_matrix(rows, cols, data1);

        let data2 = vec![vec![1.0, 2.0]];
        let matrix2 = Matrix::create_matrix(1, cols, data2);

        let result = matrix1.add_to_each_row(&matrix2);

        assert_eq!(result.get(0, 0), 2.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 0), 4.0);
        assert_eq!(result.get(1, 1), 6.0);
        assert_eq!(result.get(2, 0), 6.0);
        assert_eq!(result.get(2, 1), 8.0);

    }

    #[test]
    fn test_scalar_multiply() {
        let rows = 3;
        let cols = 2;
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut matrix = Matrix::create_matrix(rows, cols, data);

        matrix.scalar_multiply(2.0);

        assert_eq!(matrix.get(0, 0), 2.0);
        assert_eq!(matrix.get(0, 1), 4.0);
        assert_eq!(matrix.get(1, 0), 6.0);
        assert_eq!(matrix.get(1, 1), 8.0);
        assert_eq!(matrix.get(2, 0), 10.0);
        assert_eq!(matrix.get(2, 1), 12.0);

    }

    #[test]
    fn test_multiply() {
        let rows1 = 3;
        let cols1 = 2;
        let data1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix1 = Matrix::create_matrix(rows1, cols1, data1);

        let rows2 = 2;
        let cols2 = 3;
        let data2 = vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]];
        let matrix2 = Matrix::create_matrix(rows2, cols2, data2);

        let result = matrix1.multiply(&matrix2);

        assert_eq!(result.get(0, 0), 27.0);
        assert_eq!(result.get(0, 1), 30.0);
        assert_eq!(result.get(0, 2), 33.0);
        assert_eq!(result.get(1, 0), 61.0);
        assert_eq!(result.get(1, 1), 68.0);
        assert_eq!(result.get(1, 2), 75.0);
        assert_eq!(result.get(2, 0), 95.0);
        assert_eq!(result.get(2, 1), 106.0);
        assert_eq!(result.get(2, 2), 117.0);

    }

    #[test]
    fn test_multiply_into() {
        let rows1 = 3;
        let cols1 = 2;
        let data1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix1 = Matrix::create_matrix(rows1, cols1, data1);

        let rows2 = 2;
        let cols2 = 3;
        let data2 = vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]];
        let matrix2 = Matrix::create_matrix(rows2, cols2, data2);

        let mut result = Matrix::create_zero_matrix(rows1, cols2);

        matrix1.multiply_into(&matrix2, &mut result);

        assert_eq!(result.get(0, 0), 27.0);
        assert_eq!(result.get(0, 1), 30.0);
        assert_eq!(result.get(0, 2), 33.0);
        assert_eq!(result.get(1, 0), 61.0);
        assert_eq!(result.get(1, 1), 68.0);
        assert_eq!(result.get(1, 2), 75.0);
        assert_eq!(result.get(2, 0), 95.0);
        assert_eq!(result.get(2, 1), 106.0);
        assert_eq!(result.get(2, 2), 117.0);

    }

    #[test]
    fn test_hadamard() {
        let rows = 3;
        let cols = 2;
        let data1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix1 = Matrix::create_matrix(rows, cols, data1);

        let data2 = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let matrix2 = Matrix::create_matrix(rows, cols, data2);

        let result = matrix1.hadamard(&matrix2);

        assert_eq!(result.get(0, 0), 7.0);
        assert_eq!(result.get(0, 1), 16.0);
        assert_eq!(result.get(1, 0), 27.0);
        assert_eq!(result.get(1, 1), 40.0);
        assert_eq!(result.get(2, 0), 55.0);
        assert_eq!(result.get(2, 1), 72.0);

    }

    #[test]
    fn test_hadamard_into() {
        let rows = 3;
        let cols = 2;
        let data1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix1 = Matrix::create_matrix(rows, cols, data1);

        let data2 = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let matrix2 = Matrix::create_matrix(rows, cols, data2);

        let mut result = Matrix::create_zero_matrix(rows, cols);

        matrix1.hadamard_into(&matrix2, &mut result);

        assert_eq!(result.get(0, 0), 7.0);
        assert_eq!(result.get(0, 1), 16.0);
        assert_eq!(result.get(1, 0), 27.0);
        assert_eq!(result.get(1, 1), 40.0);
        assert_eq!(result.get(2, 0), 55.0);
        assert_eq!(result.get(2, 1), 72.0);

    }

    #[test]
    fn test_equals() {
        let rows = 3;
        let cols = 2;
        let data1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix1 = Matrix::create_matrix(rows, cols, data1.clone());
        let matrix2 = Matrix::create_matrix(rows, cols, data1);

        let data2 = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let matrix3 = Matrix::create_matrix(rows, cols, data2);

        assert_eq!(matrix1.equals(&matrix2), true);
        assert_eq!(matrix1.equals(&matrix3), false);
        assert_eq!(matrix2.equals(&matrix3), false);

    }
    
}
