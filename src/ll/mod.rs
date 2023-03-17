use crate::hl::shape::Shape;
use ndarray::{ArcArray, IxDyn};

/// Every tensor implementation must be able to materialize the generated tensor
pub trait BufferT<E: Backend<Buffer = Self>>: Sized {
    fn upload<T>(&self, e: &mut E, n: ArcArray<T, IxDyn>);
    fn download<T>(&self, shape: &Shape, e: &mut E) -> ArcArray<T, IxDyn>;
}

pub trait Backend: Sized {
    type Buffer: BufferT<Self>;
    fn alloc(&mut self, shp: &Shape) -> Self::Buffer;
    // fn begin(&mut self) -> ShaderBuilder<Self>;
}
