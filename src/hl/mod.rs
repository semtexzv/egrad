pub mod expr;
pub mod module;
pub mod shape;

#[cfg(test)]
mod test {
    use crate::hl::expr::param::param;
    use crate::hl::expr::{Eval, Expr, Value};
    use crate::hl::module::Module;
    use crate::ml::MLBuilder;
    use crate::shape;

    #[test]
    fn test_expr_bldr() {
        fn model<T: Value, E: Eval>() -> impl Module<Expr<T, E>, Output = Expr<T, E>> {
            let bias: Expr<T, E> = param(shape![1]);
            let weight: Expr<T, E> = param(shape![1]);

            move |i| i * &weight + &bias
        }

        #[derive(Debug)]
        struct TestEv {
            id: u64,
            bldr: MLBuilder,
        }

        impl Eval for TestEv {
            type Grad = f32;

            fn mkid(&mut self) -> u64 {
                self.id += 1;
                self.id
            }

            fn grad(&self) -> bool {
                todo!()
            }

            fn enter(&self, id: u64) {}

            fn exit(&self, id: u64) {}

            fn emitter(&mut self) -> &mut MLBuilder {
                &mut self.bldr
            }
        }

        let mut e = TestEv {
            id: 0,
            bldr: MLBuilder::new(),
        };
        let module = model::<f32, TestEv>();
        let out = module.forward(param(shape![0]));
        let out = out.eval(&mut e);
        println!("{e:#?}")
    }
}
