use crate::expr::{Eval, Expr, ExprData, ExprImpl, Ten, Value, Visitor};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use crate::shape::Shape;

pub struct Param<T, E> {
    _p: PhantomData<(T, E)>,
}

impl<T, E> Debug for Param<T, E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "param")
    }
}

impl<T: Value, E: Eval> ExprImpl<T, E> for Param<T, E> {
    fn accept(&self, v: &mut dyn Visitor<T, E>) {
        v.visit_param(self);
    }

    fn eval(&self, id: u64, e: &mut E) -> Ten<T> {
        todo!()
    }

    fn backward(&self, e: &mut E, grad: Expr<E::Grad, E>) {

        todo!()
    }
}

pub fn param<T: Value, E: Eval>(shape: Shape) -> Expr<T, E> {
    Expr(ExprData::new(Param {
        _p: Default::default(),
    }))
}
pub fn zero<T: Value, E: Eval>(shape: Shape) -> Expr<T, E> {
    Expr(ExprData::new(Param {
        _p: Default::default(),
    }))
}
