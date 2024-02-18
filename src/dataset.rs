use rand::seq::SliceRandom;
use rand::Rng;
use crate::prelude::*;
//use crate::matrix::*;

#[derive(Debug)]
pub struct DataSet {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub struct Batch<'a> {
    pub size: usize,
    pub dataset: &'a DataSet,
}

#[derive(Debug)]
pub struct Row<'a> {
    pub row_idx: usize,
    pub batch: &'a Batch<'a>,
}

pub fn create_dataset(rows: usize, cols: usize, data: Vec<Vec<f32>>) -> DataSet {
    DataSet {rows, cols, data}
}

pub fn create_batches(dataset: &DataSet, num_batches: usize) -> Vec<Batch> {
    let rows = dataset.rows;
    let mut remainder = rows % num_batches;
    let mut current_row = 0;
    let mut batches: Vec<Batch> = Vec::new();

    for _ in 0..num_batches {
        let mut size = rows / num_batches;
        if (remainder > 0) {
            size += 1;
        }

        batches.push(Batch{size, dataset});
        current_row += size;
    
        if remainder > 0 {
            remainder -= 1;
        }
    }
    
    batches
}

pub fn split_rows<'a>(batch: &'a Batch<'a>) -> Vec<Row<'a>> {
    let mut rows: Vec<Row> = Vec::new();
    
    for i in 0..batch.dataset.rows {
        rows.push(Row{row_idx: i, batch});
    }

    rows
}

pub fn shuffle_together(data_a: &mut DataSet, data_b: &mut DataSet) {
    assert!(data_a.rows == data_b.rows);
    let mut rng = rand::thread_rng();
    let mut permutation: Vec<usize> = (0..data_a.rows).collect();
    permutation.shuffle(&mut rng);
    
    for i in 0..data_a.rows {
        let j = permutation[i];
        data_a.data.swap(i, j);
        data_b.data.swap(i, j);
    }
}