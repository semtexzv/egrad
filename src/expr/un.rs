use crate::expr::param::zero;
use crate::expr::{Eval, Expr, ExprData, ExprImpl, Ten, Value, Visitor};
use std::ops::Neg;

#[derive(Debug)]
/// Unary operation. Executed on each scalar, candidate for fusion.
enum UnOp {
    /// Negation y = -x
    Neg,
    /// Reciprocal y = 1/x
    Rec,
    /// Exponential y = e^x
    Exp,
    /// Logarithm, y = ln(x)
    Log,
    /// Greater-than-zero y = x > 0
    Gtz,
}

#[derive(Debug)]
struct Un<T: Value, E: Eval> {
    op: UnOp,
    x: Expr<T, E>,
}

impl<T: Value, E: Eval> ExprImpl<T, E> for Un<T, E> {
    fn accept(&self, v: &mut dyn Visitor<T, E>) {
        self.x.accept(v);
    }

    fn eval(&self, id: u64, e: &mut E) -> Ten<T> {
        // match self.op {
        //     UnOp::Neg => {
        //         self.x.forward()
        //     }
        //     UnOp::Rec => {}
        // }
        todo!()
    }

    fn backward(&self, e: &mut E, grad: Expr<E::Grad, E>) {
        match self.op {
            UnOp::Neg => self.x.backward(e, || -grad),
            UnOp::Rec => self.x.backward(e, || {
                let xt = self.x.astype();
                -grad * &xt * xt
            }),
            UnOp::Exp => self.x.backward(e, || grad * self.x.astype()),
            UnOp::Log => self.x.backward(e, || grad / self.x.astype()),
            UnOp::Gtz => self.x.backward(e, || zero::<T, E>(grad.shape())),
        }
    }
}

impl<T: Value, E: Eval> Neg for Expr<T, E> {
    type Output = Expr<T, E>;

    fn neg(self) -> Self::Output {
        Expr(ExprData::new(Un {
            op: UnOp::Neg,
            x: self,
        }))
    }
}

impl<T: Value, E: Eval> Expr<T, E> {
    pub fn rec(self) -> Expr<T, E> {
        Expr(ExprData::new(Un {
            op: UnOp::Rec,
            x: self,
        }))
    }
    pub fn exp(self) -> Expr<T, E> {
        Expr(ExprData::new(Un {
            op: UnOp::Exp,
            x: self,
        }))
    }
    pub fn log(self) -> Expr<T, E> {
        Expr(ExprData::new(Un {
            op: UnOp::Log,
            x: self,
        }))
    }
    fn gtz(self) -> Expr<T, E> {
        Expr(ExprData::new(Un {
            op: UnOp::Gtz,
            x: self,
        }))
    }
}
