#[cfg(test)]
mod dataset_tests {
    use cranium_rs::dataset::*;

    #[test]
    fn test_create_dataset() {
        let rows = 3;
        let cols = 2;
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let dataset = create_dataset(rows, cols, data.clone());
        
        assert_eq!(dataset.rows, rows);
        assert_eq!(dataset.cols, cols);
        assert_eq!(*dataset.data, data);
    }

    #[test]
    fn test_create_batches() {
        let rows = 10;
        let cols = 2;
        let data = vec![vec![0.0; cols]; rows];
        let dataset = create_dataset(rows, cols, data.clone());

        let num_batches = 3;
        let batches = create_batches(&dataset, num_batches);

        assert_eq!(batches.len(), num_batches);
        assert_eq!(batches.iter().map(|b| b.size).sum::<usize>(), rows);
    }

    #[test]
    fn test_split_rows() {
        let rows = 5;
        let cols = 2;
        let data = vec![vec![0.0; cols]; rows];
        let dataset = create_dataset(rows, cols, data.clone());
        let batch = Batch { size: rows, offset:0, dataset: &dataset };

        let split = split_rows(&batch);
        assert_eq!(split.len(), rows);
    }

    #[test]
    fn test_shuffle_together() {
        let rows = 3;
        let cols = 2;
        let data_a = vec![vec![0.0, 3.0], vec![1.0, 4.0], vec![2.0, 5.0]];
        let data_b = vec![vec![6.0, 7.0], vec![8.0, 9.0], vec![10.0, 11.0]];

        let mut dataset_a = create_dataset(rows, cols, data_a.clone());
        let mut dataset_b = create_dataset(rows, cols, data_b.clone());

        shuffle_together(&mut dataset_a, &mut dataset_b);

        assert_ne!(dataset_a.data, data_a);
        assert_ne!(dataset_b.data, data_b);

        // Check if the elements in both datasets are shuffled in the same way
        let shuffled_data_b = dataset_b.data.clone();
        let shuffled_data_a = dataset_a.data.clone();

        let mut permutation: Vec<usize> = vec![0;rows];
        for i in 0..rows {
            permutation[i] = shuffled_data_a[i][0] as usize;
        }

        let mut reconstruction = vec![vec![0.0; cols]; rows];
        for z in 0..rows {
            for i in 0..rows {
                if permutation[i] == z{
                    reconstruction[z][0] = shuffled_data_b[i][0];
                    reconstruction[z][1] = shuffled_data_b[i][1];
                    break;
                }
            }
        }
        assert_eq!(reconstruction, data_b);
    }
}