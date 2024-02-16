pub use crate::error::Error;
pub use std::usize;
pub use std::ptr::null;
pub use std::rc::Rc;
pub use std::cell::RefCell;

pub type Result<T> = core::result::Result<T, Error>;