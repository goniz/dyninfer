//! Dialect convenience helpers (func / arith / util / stream / linalg / tensor).

pub mod arith;
pub mod func;
pub mod linalg;
pub mod tensor;
pub mod util;

pub use arith::Arith;
pub use func::Func;
pub use linalg::Linalg;
pub use tensor::Tensor;
pub use util::Util;
