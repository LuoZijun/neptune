use crate::error::Error;
use crate::gpu::GPUBatchHasher;
use crate::poseidon::SimplePoseidonBatchHasher;
use crate::BatchHasher;
use generic_array::{typenum, ArrayLength, GenericArray};
use paired::bls12_381::Fr;
use std::ops::Add;
use typenum::bit::B1;
use typenum::uint::{UInt, UTerm, Unsigned};
//use typenum::{UInt, UTerm, Unsigned, U11, U2, U8};

#[derive(Clone)]
pub enum BatcherType {
    GPU,
    CPU,
}

pub(crate) enum Batcher<'a, Arity>
where
    Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>> + ArrayLength<Fr>,
    <Arity as Add<B1>>::Output: ArrayLength<Fr>,
{
    GPU(GPUBatchHasher<Arity>),
    CPU(SimplePoseidonBatchHasher<'a, Arity>),
}

impl<Arity> Batcher<'_, Arity>
where
    Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>> + ArrayLength<Fr>,
    <Arity as Add<B1>>::Output: ArrayLength<Fr>,
{
    pub(crate) fn t(&self) -> BatcherType {
        match self {
            Batcher::GPU(_) => BatcherType::GPU,
            Batcher::CPU(_) => BatcherType::CPU,
        }
    }

    pub(crate) fn new(t: &BatcherType) -> Result<Self, Error> {
        match t {
            BatcherType::GPU => Ok(Batcher::GPU(GPUBatchHasher::<Arity>::new()?)),
            BatcherType::CPU => Ok(Batcher::CPU(SimplePoseidonBatchHasher::<Arity>::new()?)),
        }
    }
}

impl<Arity> BatchHasher<Arity> for Batcher<'_, Arity>
where
    Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>> + ArrayLength<Fr>,
    <Arity as Add<B1>>::Output: ArrayLength<Fr>,
{
    fn hash(&mut self, preimages: &[GenericArray<Fr, Arity>]) -> Vec<Fr> {
        match self {
            Batcher::GPU(batcher) => batcher.hash(preimages),
            Batcher::CPU(batcher) => batcher.hash(preimages),
        }
    }
}
