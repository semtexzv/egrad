pub mod hl;
pub mod ml;
pub mod ll;

use crate::hl::expr::{Eval, Expr, Value};
use crate::hl::module::Module;
use std::ops::{Deref, Mul};

fn main() {

    // let a: Expr<f32, ()> = param(shape![1]);
    // let module = model();
    // let b = module.forward(a.clone());
    //
    // println!("{:#?}", a.clone() * clone&b + &a);
}
