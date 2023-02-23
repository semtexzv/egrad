pub mod back;
pub mod expr;
pub mod module;
pub mod shape;

use crate::expr::param::param;
use crate::expr::{Eval, Expr, ExprImpl, Value};
use crate::module::Module;
use std::ops::{Deref, Mul};

fn model<T: Value, E: Eval>() -> impl Module<Expr<T, E>, Output = Expr<T, E>> {
    let bias: Expr<T, E> = param(shape![1]);
    let weight: Expr<T, E> = param(shape![1]);

    move |i| i * &weight + &bias
}

fn main() {
    let a: Expr<f32, ()> = param(shape![1]);
    let module = model();
    let b = module.forward(a.clone());

    println!("{:#?}", a.clone() * &b + &a);
}
