use crate::hl::shape::Shape;
use indexmap::map::Entry;
use indexmap::{IndexMap, IndexSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::BuildHasher;

#[repr(u8)]
#[derive(Debug, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub enum PadKind {
    // Pad with zeroes
    Zero,
    // Pad with ones
    One,
    // Mirror from the buffer
    Mirror,
    // Keep the edge value
    Edge,
}

#[repr(u8)]
#[derive(Debug, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub enum OpType {
    // Binary ops
    Add,
    Mul,
    Pow,
    Eq,

    // Unary ops
    Neg,
    Rec,
    Exp,
    Log,
    Gtz,

    // Matrix ops
    MatMul,

    // Reduce ops
    Max,
    Sum,

    //Shape ops
    Broadcast {
        axis: usize,
        count: usize,
    },
    Cat {
        axis: usize,
    },
    Flip {
        axis: usize,
    },
    Permute {
        axes: Vec<usize>,
    },
    Pad {
        axis: usize,
        amt: usize,
        kind: PadKind,
    },
}

#[derive(Debug, Default, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct BufId(u64);

#[derive(Debug, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct MLOp {
    op: OpType,
    src1: BufId,
    src2: BufId,
}

#[derive(Debug)]
pub struct OpInfo {
    op: OpType,
    osh: Shape,

    src1: BufId,

    src2: BufId,
}

struct ZeroInit;

impl BuildHasher for ZeroInit {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> Self::Hasher {
        DefaultHasher::new()
    }
}

#[derive(Debug)]
pub struct MLBuilder {
    maxid: u64,

    buffs: IndexMap<BufId, Shape, ZeroInit>,
    nodes: IndexMap<MLOp, BufId, ZeroInit>,
    instr: IndexMap<BufId, OpInfo, ZeroInit>,

    shap: IndexMap<BufId, Shape, ZeroInit>,
    // What nodes depend on a given node
    // If this is low + computation for this node is easy
    // We should inline it.
    deps: IndexMap<BufId, IndexSet<BufId>, ZeroInit>,
}

impl MLBuilder {
    pub fn new() -> Self {
        Self {
            maxid: 0,
            buffs: IndexMap::with_hasher(ZeroInit),
            nodes: IndexMap::with_hasher(ZeroInit),
            instr: IndexMap::with_hasher(ZeroInit),
            shap: IndexMap::with_hasher(ZeroInit),
            deps: IndexMap::with_hasher(ZeroInit),
        }
    }

    fn newid(&mut self) -> BufId {
        self.maxid += 1;
        BufId(self.maxid)
    }

    pub fn buffer(&mut self, shape: Shape) -> BufId {
        let id = self.newid();
        self.buffs.insert(id, shape);
        id
    }

    pub fn emit(&mut self, op: OpType, osh: &Shape, src1: BufId, src2: BufId) -> BufId {
        let mlop = MLOp {
            op: op.clone(),
            src1,
            src2,
        };
        let outid = match self.nodes.entry(mlop) {
            Entry::Occupied(b) => *b.get(),
            Entry::Vacant(e) => {
                self.maxid += 1;
                let outid = BufId(self.maxid);

                e.insert(outid);
                self.shap.insert(outid, osh.clone());
                self.instr.insert(
                    outid,
                    OpInfo {
                        op: op.clone(),
                        osh: osh.clone(),
                        src1,
                        src2,
                    },
                );

                outid
            }
        };

        self.deps.entry(src1).or_default().insert(outid);
        self.deps.entry(src2).or_default().insert(outid);

        outid
    }
}

#[cfg(test)]
mod test {
    use crate::ml::{BufId, MLBuilder, OpType};
    use crate::shape;

    #[test]
    fn test_emit() {
        let mut bld = MLBuilder::new();
        let b1 = bld.buffer(shape![1, 1]);
        let b2 = bld.buffer(shape![1, 1]);

        let b3 = bld.emit(OpType::Add, &shape![], b1, b2);
        assert_eq!(b3, BufId(3));

        let b4 = bld.emit(OpType::Mul, &shape![], b1, b2);
        assert_eq!(b4, BufId(4));

        let b5 = bld.emit(OpType::Add, &shape![], b1, b2);
        assert_eq!(b5, BufId(3));

        let b6 = bld.emit(OpType::Add, &shape![], b3, b4);
        assert_eq!(b6, BufId(5));

        println!("{:#?}", bld);
    }
}
