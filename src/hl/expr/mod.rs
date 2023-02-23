pub mod bin;
pub mod param;
pub mod un;

use crate::hl::expr::param::Param;
use crate::hl::shape::Shape;
use crate::ml::{BufId, MLBuilder};
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

    /// Make a new unique id for expression node
    fn mkid(&mut self) -> u64;
    /// Whether we're interested in tracking gradients.
    fn grad(&self) -> bool;

    fn enter(&self, id: u64);
    fn exit(&self, id: u64);

    /// MLOp emitter. Used in forward pass to creat the actual computation graph (with optimizations).
    fn emitter(&mut self) -> &mut MLBuilder;
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
    fn shape(&self) -> &Shape;
    fn accept(&self, v: &mut dyn Visitor<T, E>);
    /// Evaluates this expression, producing materializable resutl
    fn eval(&self, id: u64, e: &mut E) -> BufId;
    /// Implements backwards pass for a graph
    fn backward(&self, e: &mut E, grad: Expr<E::Grad, E>);

    /// Augmented Any like functionality
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

impl<T: Value, E: Eval> Expr<T, E> {
    pub fn eval(&self, e: &mut E) -> BufId {
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
    pub _p: PhantomData<T>,
    pub id: Cell<u64>,
    pub val: Cell<Option<BufId>>,
    pub grad: RefCell<Option<Ten<E::Grad>>>,
    pub _impl: I,
}

impl<T: Value, E: Eval, I: ExprImpl<T, E> + Sized> ExprData<T, E, I> {
    pub fn new(i: I) -> Rc<ExprData<T, E, I>> {
        Rc::new(ExprData {
            _p: Default::default(),
            id: Cell::new(0),
            val: Cell::new(None),
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
    fn eval(&self, e: &mut E) -> BufId {
        if let Some(bufid) = self.val.get() {
            return bufid;
        }

        let mut id = self.id.get();

        if id == 0 {
            id = e.mkid();
            self.id.set(id);
        }

        e.enter(id);
        let out = self._impl.eval(id, e);
        e.exit(id);
        out
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
    fn shape(&self) -> &Shape {
        self.0._impl.shape()
    }
}

impl<T: Value, E: Eval> Clone for Expr<T, E> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
