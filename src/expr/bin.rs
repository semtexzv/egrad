use crate::expr::{Eval, Expr, ExprData, ExprImpl, Ten, Value, Visitor};
use std::ops::{Add, Deref, Div, Mul, Sub};

#[derive(Debug)]
pub(crate) enum BinOp {
    Add,
    Mul,
}

#[derive(Debug)]
pub struct Bin<T: Value, E: Eval> {
    op: BinOp,
    l: Expr<T, E>,
    r: Expr<T, E>,
}

impl<T: Value, E: Eval> ExprImpl<T, E> for Bin<T, E> {
    fn accept(&self, v: &mut dyn Visitor<T, E>) {
        self.l.accept(v);
        self.r.accept(v);
    }

    fn eval(&self, id: u64, e: &mut E) -> Ten<T> {
        let l = self.l.eval(e);
        let r = self.r.eval(e);
        l
    }

    fn backward(&self, e: &mut E, grad: Expr<E::Grad, E>) {
        match self.op {
            BinOp::Add => {
                self.l.backward(e, || grad.clone());
                self.r.backward(e, || grad);
            }
            BinOp::Mul => {
                self.l.backward(e, || self.l.astype() * &grad);
                self.r.backward(e, || self.r.astype() * &grad);
            }
        }
    }
}

impl<T, E, RHS> Add<RHS> for Expr<T, E>
where
    T: Value,
    E: Eval,
    RHS: Into<Expr<T, E>>,
{
    type Output = Expr<T, E>;

    fn add(self, rhs: RHS) -> Self::Output {
        Expr(ExprData::new(Bin {
            op: BinOp::Add,
            l: self,
            r: rhs.into(),
        }))
    }
}

impl<T, E, RHS> Sub<RHS> for Expr<T, E>
where
    T: Value,
    E: Eval,
    RHS: Into<Expr<T, E>>,
{
    type Output = Expr<T, E>;

    fn sub(self, rhs: RHS) -> Self::Output {
        self + -rhs.into()
    }
}

impl<T, E, RHS> Mul<RHS> for Expr<T, E>
where
    T: Value,
    E: Eval,
    RHS: Into<Expr<T, E>>,
{
    type Output = Expr<T, E>;

    fn mul(self, rhs: RHS) -> Self::Output {
        Expr(ExprData::new(Bin {
            op: BinOp::Mul,
            l: self,
            r: rhs.into(),
        }))
    }
}

impl<T, E, RHS> Div<RHS> for Expr<T, E>
where
    T: Value,
    E: Eval,
    RHS: Into<Expr<T, E>>,
{
    type Output = Expr<T, E>;

    fn div(self, rhs: RHS) -> Self::Output {

        self * rhs.into().rec()
    }
}
