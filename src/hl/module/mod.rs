use crate::hl::expr::param::param;
use crate::hl::expr::{Eval, Expr, Ten, Value};
use crate::hl::shape::Shape;

pub trait ModuleInput {
    /// What shape each element in a batch has.
    /// Within an invocation, different inputs might have different batch size
    type Shapes;
    fn zero(s: &Self::Shapes) -> Self;
}

impl<T: Value, E: Eval> ModuleInput for Expr<T, E> {
    type Shapes = Shape;

    fn zero(s: &Self::Shapes) -> Self {
        param(s.clone())
    }
}

pub trait Module<Input: ModuleInput> {
    type Output;

    fn forward(&self, i: Input) -> Self::Output;
}

impl<Input: ModuleInput, T, O> Module<Input> for T
where
    T: Fn(Input) -> O,
{
    type Output = O;

    fn forward(&self, i: Input) -> Self::Output {
        (self)(i)
    }
}
