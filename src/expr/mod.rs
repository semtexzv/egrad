pub mod bin;
pub mod param;
pub mod un;

use crate::expr::param::Param;
use crate::shape::Shape;
use std::any::{type_name, Any as StdAny, Any, TypeId};
use std::cell::{Cell, RefCell};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Debug)]
pub struct Ten<T> {
    _t: PhantomData<T>,
}

impl<T> Ten<T> {
    pub fn zeros(s: &Shape) -> Self {
        todo!()
    }
}

impl<T> Clone for Ten<T> {
    fn clone(&self) -> Self {
        todo!()
    }
}

pub trait Value: Debug + 'static {}
impl Value for f32 {}

pub trait Eval: Debug + 'static {
    type Grad: Value;

    fn on_forward<F: FnOnce(&mut Self) -> R, R>(&mut self, id: u64, f: F) -> R;

    fn grad(&self) -> bool;
    fn mkid(&mut self) -> u64;
    fn saved(&mut self, id: u64);
}

impl Eval for () {
    type Grad = f32;

    fn on_forward<F: FnOnce(&mut Self) -> R, R>(&mut self, id: u64, f: F) -> R {
        todo!()
    }

    fn grad(&self) -> bool {
        todo!()
    }

    fn mkid(&mut self) -> u64 {
        todo!()
    }

    fn saved(&mut self, id: u64) {
        todo!()
    }
}

impl<T: Value, E: Eval> Into<Expr<T, E>> for &Expr<T, E> {
    fn into(self) -> Expr<T, E> {
        Expr::clone(self)
    }
}

/// General visitior for working traversing the expression tree
pub trait Visitor<T: Value, E: Eval> {
    fn visit_param(&mut self, p: &Param<T, E>);
}

/// Expression implementation. In the forward pass, it should evaluate subexpressions
/// 
/// In backwards passes it should prepare the backwards graph from the forward one
pub trait ExprImpl<T: Value, E: Eval>: Any + Debug {
    fn accept(&self, v: &mut dyn Visitor<T, E>);
    /// Evaluates this expression, producing materializable resutl
    fn eval(&self, id: u64, e: &mut E) -> Ten<T>;
    /// Implements backwards pass for a graph
    fn backward(&self, e: &mut E, grad: Expr<E::Grad, E>);

    /// Augmented Any like functionality
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

impl<T: Value, E: Eval> Expr<T, E> {
    pub fn eval(&self, e: &mut E) -> Ten<T> {
        self.0.eval(e)
    }
    /// If this tensor requires grad
    pub fn backward<V: Value, F: FnOnce() -> Expr<V, E>>(&self, e: &mut E, v: F) {
        // DO NOT RECURSIVELY CALL INTERNAL BACKWARD
    }
    pub fn astype<V: Value>(&self) -> Expr<V, E> {
        todo!()
    }
}

#[derive(Debug)]
/// A DST that contains all tensor data along with the implementation of the tensor logic.
pub struct ExprData<T: Value, E: Eval, I: ?Sized = dyn ExprImpl<T, E> + 'static> {
    pub id: Cell<u64>,
    pub val: RefCell<Option<Ten<T>>>,
    pub grad: RefCell<Option<Ten<E::Grad>>>,
    pub _impl: I,
}

impl<T: Value, E: Eval, I: ExprImpl<T, E> + Sized> ExprData<T, E, I> {
    pub fn new(i: I) -> Rc<ExprData<T, E, I>> {
        Rc::new(ExprData {
            id: Cell::new(0),
            val: RefCell::new(None),
            grad: RefCell::new(None),
            _impl: i,
        })
    }

    pub(crate) fn is(this: Rc<ExprData<T, E, dyn ExprImpl<T, E> + 'static>>) -> bool {
        this._impl.type_id() == TypeId::of::<I>()
    }

    pub(crate) fn downcast(this: Rc<ExprData<T, E, dyn ExprImpl<T, E> + 'static>>) -> Rc<Self> {
        assert!(
            this._impl.type_id() == TypeId::of::<I>(),
            "Downcasting from: {}, to: {}",
            this._impl.type_name(),
            type_name::<I>()
        );
        unsafe { Rc::from_raw(Rc::into_raw(this) as *mut _) }
    }
}

impl<T: Value, E: Eval, I: ExprImpl<T, E> + ?Sized> ExprData<T, E, I> {
    fn eval(&self, e: &mut E) -> Ten<T> {
        let mut id = self.id.get();

        if id == 0 {
            id = e.mkid();
            self.id.set(id);
        }
        e.on_forward(id, |e| self._impl.eval(id, e))
    }
}

#[derive(Debug)]
/// Reference counted expression. Internaly stores it's value & gradient (if needed).
///
/// Uses DST coercion to get favorable memory layout something like:
///
/// Rc<(id, cached output, cached grad, ExprImpl data)>.
pub struct Expr<T: Value, E: Eval, I: ?Sized = dyn ExprImpl<T, E>>(pub Rc<ExprData<T, E, I>>);

impl<T: Value, E: Eval> Expr<T, E> {
    fn accept(&self, v: &mut dyn Visitor<T, E>) {
        self.0._impl.accept(v);
    }
}

impl<T: Value, E: Eval, I: Sized + ExprImpl<T, E>> Expr<T, E, I> {
    pub(crate) fn is(this: Expr<T, E, dyn ExprImpl<T, E> + 'static>) -> bool {
        this.0._impl.type_id() == TypeId::of::<I>()
    }

    pub(crate) fn downcast(this: Expr<T, E, dyn ExprImpl<T, E> + 'static>) -> Self {
        assert!(
            this.0._impl.type_id() == TypeId::of::<I>(),
            "Downcasting from: {}, to: {}",
            this.0._impl.type_name(),
            type_name::<I>()
        );
        unsafe { Expr(Rc::from_raw(Rc::into_raw(this.0) as *mut _)) }
    }
}

impl<T: Value, E: Eval> Expr<T, E> {
    fn shape(&self) -> Shape {
        todo!()
    }
}

impl<T: Value, E: Eval> Clone for Expr<T, E> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
