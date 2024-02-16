#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use cranium_rs::dataset::*;

    #[test]
    fn test_create_dataset() {
        let rows = 3;
        let cols = 2;
        let data = Rc::new(RefCell::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]));
        let dataset = create_dataset(rows, cols, data.clone());
        
        assert_eq!(dataset.rows, rows);
        assert_eq!(dataset.cols, cols);
        assert_eq!(*dataset.data.borrow(), *data.borrow());
    }

    #[test]
    fn test_create_batches() {
        let rows = 10;
        let cols = 2;
        let data = Rc::new(RefCell::new(vec![vec![0.0; cols]; rows]));
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
        let data = Rc::new(RefCell::new(vec![vec![0.0; cols]; rows]));
        let dataset = create_dataset(rows, cols, data.clone());
        let batch = Batch { size: rows, dataset: &dataset };

        let split = split_rows(&batch);
        assert_eq!(split.len(), rows);
    }

    #[test]
    fn test_shuffle_together() {
        let rows = 5;
        let cols = 2;
        let data_a = Rc::new(RefCell::new(vec![vec![0.0; cols]; rows]));
        let data_b = Rc::new(RefCell::new(vec![vec![1.0; cols]; rows]));

        let original_a = data_a.clone();
        let original_b = data_b.clone();

        let dataset_a = create_dataset(rows, cols, data_a.clone());
        let dataset_b = create_dataset(rows, cols, data_b.clone());

        shuffle_together(&dataset_a, &dataset_b);

        // Check if the elements in both datasets are shuffled in the same way
        assert_ne!(*data_a.borrow(), *original_a.borrow());
        assert_ne!(*data_b.borrow(), *original_b.borrow());
    }
}
