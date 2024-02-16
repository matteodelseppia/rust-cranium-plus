#![allow(unused)]

mod prelude;
pub mod error;
pub mod matrix;
pub mod function;
pub mod layer;
pub mod network;

#[cfg(test)]
mod tests {
    #[derive(Debug)]
struct Data {
    pub val: i32,
}

    use std::{vec};
    use crate::prelude::*;
    use crate::matrix::*;
    use crate::function::*;
    use crate::network::*;

    #[test]
    fn test() {
        let mut matrix = create_matrix(1, 3, Rc::new(RefCell::new(vec![1.2, 3.4, 5.6])));
        sigmoid(&mut matrix);
        println!("{:?}", &matrix);
    }
}