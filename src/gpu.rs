use crate::error::Error;
use crate::poseidon::{Poseidon, PoseidonConstants};
use ff::{Field, PrimeField, PrimeFieldDecodingError, ScalarEngine};
use generic_array::{sequence::GenericSequence, typenum, ArrayLength, GenericArray};
use paired::bls12_381::{Bls12, Fr, FrRepr};
use std::marker::PhantomData;
use std::ops::Add;
use triton::FutharkContext;
use triton::{Array_u64_1d, Array_u64_2d, Array_u64_3d};
use typenum::bit::B1;
use typenum::{UInt, UTerm, Unsigned, U11, U2, U8};

type P2State = triton::FutharkOpaqueP2State;
type P8State = triton::FutharkOpaqueP8State;
type P11State = triton::FutharkOpaqueP11State;

struct GPUConstants<Arity>(PoseidonConstants<Bls12, Arity>)
where
    Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>>;

impl<Arity> GPUConstants<Arity>
where
    Arity: Unsigned + Add<B1> + Add<UInt<UTerm, B1>>,
{
    fn arity_tag(&self, ctx: &FutharkContext) -> Result<Array_u64_1d, Error> {
        let arity_tag = self.0.arity_tag;
        array_u64_1d_from_fr(ctx, arity_tag)
    }

    fn round_keys(&self, ctx: &FutharkContext) -> Result<Array_u64_2d, Error> {
        let round_keys = &self.0.compressed_round_constants;
        array_u64_2d_from_frs(ctx, &round_keys)
    }

    fn mds_matrix(&self, ctx: &FutharkContext) -> Result<Array_u64_3d, Error> {
        let matrix = &self.0.mds_matrices.m;

        array_u64_3d_from_frs_2d(ctx, matrix)
    }

    fn pre_sparse_matrix(&self, ctx: &FutharkContext) -> Result<Array_u64_3d, Error> {
        let pre_sparse_matrix = &self.0.pre_sparse_matrix;

        array_u64_3d_from_frs_2d(ctx, pre_sparse_matrix)
    }

    fn sparse_matrixes(&self, ctx: &FutharkContext) -> Result<Array_u64_3d, Error> {
        let sparse_matrixes = &self.0.sparse_matrixes;

        let frs_2d: Vec<Vec<Fr>> = sparse_matrixes
            .iter()
            .map(|m| {
                let mut x = m.w_hat.clone();
                x.extend(m.v_rest.clone());
                x.into_iter().collect()
            })
            .collect();

        array_u64_3d_from_frs_2d(ctx, &frs_2d)
    }
}

fn frs_to_u64s(frs: &[Fr]) -> Vec<u64> {
    let mut res = vec![u64::default(); frs.len() * 4];
    for (src, dest) in frs.iter().zip(res.chunks_mut(4)) {
        dest.copy_from_slice(&src.into_repr().0);
    }
    res
}

fn frs_2d_to_u64s(frs_2d: &[Vec<Fr>]) -> Vec<u64> {
    frs_2d
        .iter()
        .flat_map(|row| frs_to_u64s(row).into_iter())
        .collect()
}

fn array_u64_1d_from_fr(ctx: &FutharkContext, fr: Fr) -> Result<Array_u64_1d, Error> {
    Array_u64_1d::from_vec(*ctx, &fr.into_repr().0, &[4, 1])
        .map_err(|e| Error::Other(format!("error converting Fr: {:?}", e).to_string()))
}

fn array_u64_1d_from_frs(ctx: &FutharkContext, frs: &[Fr]) -> Result<Array_u64_1d, Error> {
    let u64s = frs_to_u64s(frs);

    Array_u64_1d::from_vec(*ctx, u64s.as_slice(), &[(frs.len() * 4) as i64, 1])
        .map_err(|e| Error::Other(format!("error converting Fr: {:?}", e).to_string()))
}

fn array_u64_2d_from_frs(ctx: &FutharkContext, frs: &[Fr]) -> Result<Array_u64_2d, Error> {
    let u64s = frs_to_u64s(frs);

    let d2 = 4;
    let d1 = u64s.len() as i64 / d2;
    let dim = [d1, d2];

    Array_u64_2d::from_vec(*ctx, u64s.as_slice(), &dim)
        .map_err(|e| Error::Other(format!("error converting Frs: {:?}", e).to_string()))
}

fn array_u64_3d_from_frs_2d(
    ctx: &FutharkContext,
    frs_2d: &[Vec<Fr>],
) -> Result<Array_u64_3d, Error> {
    let u64s = frs_2d_to_u64s(frs_2d);

    let mut l = u64s.len() as i64;
    let d1 = 4; // One Fr is 4 x u64.
    l /= d1;

    let d2 = frs_2d[0].len() as i64;
    assert!(
        frs_2d.iter().all(|x| x.len() == d2 as usize),
        "Frs must be grouped uniformly"
    );
    l /= d2;

    let d3 = l as i64;
    let dim = [d3, d2, d1];

    Array_u64_3d::from_vec(*ctx, u64s.as_slice(), &dim)
        .map_err(|e| Error::Other(format!("error converting Frs 2d: {:?}", e).to_string()))
}

pub fn exercise_gpu() -> Result<(), triton::Error> {
    let mut ctx = FutharkContext::new();

    let res_arr = ctx.simple11(5)?;
    let (vec, shape) = &res_arr.to_vec();
    let n = shape[0];
    let chunk_size = shape[1] as usize;

    assert_eq!(2, shape.len());

    for (i, chunk) in vec.chunks(chunk_size).enumerate() {
        print!("res {} of {}: ", i, n);
        print!("[");
        for elt in chunk.iter() {
            print!("{}, ", elt);
        }
        println!("]");
    }

    Ok(())
}

pub fn u64s_into_fr(limbs: &[u64]) -> Result<Fr, PrimeFieldDecodingError> {
    assert_eq!(limbs.len(), 4);
    let mut limb_arr = [0; 4];
    limb_arr.copy_from_slice(&limbs[..]);
    let repr = FrRepr(limb_arr);
    let fr = Fr::from_repr(repr);

    fr
}

// pub fn monts_into_frs<'a>(u64s: &[u64]) -> Result<&'a [Fr], PrimeFieldDecodingError> {
//     let fr_size = 4;
//     let fr_count = u64s.len() / fr_size;
//     assert_eq!(0, fr_count % fr_size);
//     // let mut copied = vec![0; u64s.len()];
//     // copied.copy_from_slice(&u64s[..]);

//     let frs =
//         unsafe { std::slice::from_raw_parts(u64s.as_ptr() as *const () as *const Fr, fr_count) };

//     Ok(frs)
// }

fn unpack_fr_array(vec_shape: (Vec<u64>, &[i64])) -> Result<Vec<Fr>, Error> {
    let (vec, shape) = vec_shape;
    let chunk_size = shape[shape.len() - 1] as usize;

    vec.chunks(chunk_size)
        .map(|x| u64s_into_fr(x))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::DecodingError)
}

fn unpack_fr_array_from_monts<'a>(monts: &'a [u64]) -> Result<&'a [Fr], Error> {
    let fr_size = 4;
    let fr_count = monts.len() / fr_size;
    if monts.len() % fr_size != 0 {
        return Err(Error::Other(
            "wrong size monts to convert to Frs".to_string(),
        ));
    }
    // let mut copied = vec![0; monts.len()];
    // copied.copy_from_slice(&vec[..]);

    let frs =
        unsafe { std::slice::from_raw_parts(monts.as_ptr() as *const () as *const Fr, fr_count) };

    Ok(frs)
}

fn as_mont_u64s<'a, U: ArrayLength<Fr>>(vec: &'a [GenericArray<Fr, U>]) -> &'a [u64] {
    let fr_size = 4; // Number of limbs in Fr.
    assert_eq!(
        fr_size * std::mem::size_of::<u64>(),
        std::mem::size_of::<Fr>(),
        "fr size changed"
    );

    unsafe {
        std::slice::from_raw_parts(
            vec.as_ptr() as *const () as *const u64,
            vec.len() * fr_size * U::to_usize(),
        )
    }
}

fn as_u64s<U: ArrayLength<Fr>>(vec: &[GenericArray<Fr, U>]) -> Vec<u64> {
    if vec.len() == 0 {
        return Vec::new();
    }
    let fr_size = std::mem::size_of::<Fr>();
    let mut safely = Vec::with_capacity(vec.len() * U::to_usize() * fr_size);
    for i in 0..vec.len() {
        for j in 0..U::to_usize() {
            for k in 0..4 {
                safely.push(vec[i][j].into_repr().0[k]);
            }
        }
    }
    safely
}

fn simple11(n: i32) -> Result<Vec<Fr>, Error> {
    let mut ctx = FutharkContext::new();

    let res_arr = ctx
        .simple11(n)
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;
    let (vec, shape) = &res_arr.to_vec();
    unpack_fr_array((vec.to_vec(), shape.as_slice()))
}

pub fn hash_binary(
    ctx: &mut FutharkContext,
    state: P2State,
    preimage: &[Fr],
) -> Result<(Fr, P2State), Error>
where
{
    let preimage_u64s = frs_to_u64s(preimage);
    let (res, state) = ctx
        .hash2(
            state,
            Array_u64_1d::from_vec(*ctx, &preimage_u64s, &[8, 1])
                .map_err(|_| Error::Other("could not convert".to_string()))?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let (vec, shape) = res.to_vec();
    dbg!(&vec, &shape);

    Ok((
        unpack_fr_array((vec, shape.as_slice())).map(|frs| frs[0])?,
        state,
    ))
}

pub fn hash_oct(
    ctx: &mut FutharkContext,
    state: P8State,
    preimage: Vec<Fr>,
) -> Result<(Fr, P8State), Error>
where
{
    let preimage_u64s = frs_to_u64s(&preimage);
    dbg!(preimage_u64s.len());
    let (res, state) = ctx
        .hash8(
            state,
            Array_u64_1d::from_vec(*ctx, &preimage_u64s, &[32, 1])
                .map_err(|_| Error::Other("could not convert".to_string()))?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let (vec, shape) = res.to_vec();

    Ok((
        unpack_fr_array((vec, shape.as_slice())).map(|frs| frs[0])?,
        state,
    ))
}

pub fn hash_column(
    ctx: &mut FutharkContext,
    state: P11State,
    preimage: Vec<Fr>,
) -> Result<(Fr, P11State), Error>
where
{
    let preimage_u64s = frs_to_u64s(&preimage);
    dbg!(preimage_u64s.len());
    let (res, state) = ctx
        .hash11(
            state,
            Array_u64_1d::from_vec(*ctx, &preimage_u64s, &[44, 1])
                .map_err(|_| Error::Other("could not convert".to_string()))?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let (vec, shape) = res.to_vec();

    Ok((
        unpack_fr_array((vec, shape.as_slice())).map(|frs| frs[0])?,
        state,
    ))
}

fn init_hash2(ctx: &mut FutharkContext) -> Result<P2State, Error> {
    let constants = GPUConstants(PoseidonConstants::<Bls12, U2>::new());
    let state = ctx
        .init2(
            constants.arity_tag(&ctx)?,
            constants.round_keys(&ctx)?,
            constants.mds_matrix(&ctx)?,
            constants.pre_sparse_matrix(&ctx)?,
            constants.sparse_matrixes(&ctx)?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    Ok(state)
}

fn init_hash8(ctx: &mut FutharkContext) -> Result<P8State, Error> {
    let constants = GPUConstants(PoseidonConstants::<Bls12, U8>::new());
    let state = ctx
        .init8(
            constants.arity_tag(&ctx)?,
            constants.round_keys(&ctx)?,
            constants.mds_matrix(&ctx)?,
            constants.pre_sparse_matrix(&ctx)?,
            constants.sparse_matrixes(&ctx)?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    Ok(state)
}

fn init_hash11(ctx: &mut FutharkContext) -> Result<P11State, Error> {
    let constants = GPUConstants(PoseidonConstants::<Bls12, U11>::new());
    let state = ctx
        .init11(
            constants.arity_tag(&ctx)?,
            constants.round_keys(&ctx)?,
            constants.mds_matrix(&ctx)?,
            constants.pre_sparse_matrix(&ctx)?,
            constants.sparse_matrixes(&ctx)?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    Ok(state)
}

fn batch_hash2(
    ctx: &mut FutharkContext,
    state: P2State,
    preimages: &[GenericArray<Fr, U2>],
) -> Result<(Vec<Fr>, P2State), Error> {
    let flat_preimages = as_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .batch_hash2(state, input)
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let (vec, shape) = res.to_vec();
    let unpacked = unpack_fr_array((vec, shape.as_slice()))?;

    Ok((unpacked, state))
}

fn batch_hash8(
    ctx: &mut FutharkContext,
    state: P8State,
    preimages: &[GenericArray<Fr, U8>],
) -> Result<(Vec<Fr>, P8State), Error> {
    let flat_preimages = as_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .batch_hash8(state, input)
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let (vec, shape) = res.to_vec();
    let unpacked = unpack_fr_array((vec, shape.as_slice()))?;

    Ok((unpacked, state))
}
fn batch_hash11(
    ctx: &mut FutharkContext,
    state: P11State,
    preimages: &[GenericArray<Fr, U11>],
) -> Result<(Vec<Fr>, P11State), Error> {
    let flat_preimages = as_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .batch_hash11(state, input)
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let (vec, shape) = res.to_vec();
    let unpacked = unpack_fr_array((vec, shape.as_slice()))?;

    Ok((unpacked, state))
}

fn mbatch_hash2(
    ctx: &mut FutharkContext,
    state: P2State,
    preimages: &[GenericArray<Fr, U2>],
) -> Result<(Vec<Fr>, P2State), Error> {
    //) -> Result<(Vec<Fr>, P2State), Error> {
    let flat_preimages = as_mont_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .mbatch_hash2(state, input)
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let (vec, shape) = res.to_vec();
    //let unpacked = unpack_fr_array((vec, shape.as_slice()))?;
    dbg!(&shape, &vec[0..4], vec.len());
    // assert_eq!(15263440169939121167, vec[0]);
    let frs = unpack_fr_array_from_monts(vec.as_slice())?;
    Ok((frs.to_vec(), state))
}

fn mbatch_hash8(
    ctx: &mut FutharkContext,
    state: P8State,
    preimages: &[GenericArray<Fr, U8>],
) -> Result<(Vec<Fr>, P8State), Error> {
    let flat_preimages = as_mont_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .mbatch_hash8(state, input)
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let (vec, shape) = res.to_vec();
    let frs = unpack_fr_array_from_monts(vec.as_slice())?;

    Ok((frs.to_vec(), state))
}
fn mbatch_hash11(
    ctx: &mut FutharkContext,
    state: P11State,
    preimages: &[GenericArray<Fr, U11>],
) -> Result<(Vec<Fr>, P11State), Error> {
    let flat_preimages = as_mont_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .mbatch_hash11(state, input)
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let (vec, shape) = res.to_vec();
    let frs = unpack_fr_array_from_monts(vec.as_slice())?;

    Ok((frs.to_vec(), state))
}

// type CTB2KState = triton::FutharkOpaqueCtb2KState;
// type CTB4GState = triton::FutharkOpaqueCtb4GState;
// type CTB512MState = triton::FutharkOpaqueCtb512MState;

// pub struct ColumnTreeBuilder2k<U11, U8>
// where
// // ColumnArity: ArrayLength<Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
// // TreeArity: ArrayLength<Fr> + Add<B1> + Add<UInt<UTerm, B1>>,
// {
//     ctx: FutharkContext,
//     state: CTB2KState,
//     _c: PhantomData<U11>,
//     _t: PhantomData<U8>,
// }

// impl ColumnTreeBuilderTrait<Bls12, U11, U8> for CTB2KState {
//     fn new(leaf_count: usize) -> Self {
//         assert_eq!(64, leaf_count);
//         let mut ctx = FutharkContext::new();
//         state = init_column_tree_builder_2k(&mut ctx).unwrap()
//     }

//     fn add_columns(&mut self, columns: &[GenericArray<Fr, U11>]) -> Result<(), Error> {
//         self. = add_columns_2k(&mut self.ctx, self.state, columns)?;

//         Ok(())
//     }

// }

// impl ColumnTreeBuilderTrait<Bls12, U11, U8> for ColumnTreeBuilder2k<U11, U8> {
//     fn new(leaf_count: usize) -> Self {
//         assert_eq!(64, leaf_count);
//         let mut ctx = FutharkContext::new();
//         let state = init_column_tree_builder_2k(&mut ctx).unwrap();
//         dbg!(&state.ptr);
//         Self {
//             ctx,
//             state,
//             _c: PhantomData::<U11>,
//             _t: PhantomData::<U8>,
//         }
//     }

//     fn add_columns(&mut self, columns: &[GenericArray<Fr, U11>]) -> Result<(), Error> {
//         let mut ctx = self.ctx;
//         //self.state = add_columns_2k(&mut ctx, self.state, columns)?;
//         Ok(())
//     }

//     fn add_final_columns(&mut self, columns: &[GenericArray<Fr, U11>]) -> Result<Vec<Fr>, Error> {
//         unimplemented!();
//         // let mut ctx = self.ctx;
//         // let (res, state) = finalize_2k(&mut ctx, self.state.to_owned())?;
//         // self.state = state;
//         // Ok(res)
//     }

//     fn reset(&mut self) {
//         unimplemented!();
//         // FIXME: Add entry point.
//     }
// }

// fn init_column_tree_builder_2k(ctx: &mut FutharkContext) -> Result<CTB2KState, Error> {
//     let column_constants = GPUConstants(PoseidonConstants::<Bls12, U11>::new());
//     let tree_constants = GPUConstants(PoseidonConstants::<Bls12, U8>::new());

//     let state = ctx
//         .init_2k(
//             tree_constants.arity_tag(&ctx)?,
//             tree_constants.round_keys(&ctx)?,
//             tree_constants.mds_matrix(&ctx)?,
//             tree_constants.pre_sparse_matrix(&ctx)?,
//             tree_constants.sparse_matrixes(&ctx)?,
//             column_constants.arity_tag(&ctx)?,
//             column_constants.round_keys(&ctx)?,
//             column_constants.mds_matrix(&ctx)?,
//             column_constants.pre_sparse_matrix(&ctx)?,
//             column_constants.sparse_matrixes(&ctx)?,
//         )
//         .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

//     Ok(state)
// }

// fn init_column_tree_builder_512m(ctx: &mut FutharkContext) -> Result<CTB512MState, Error> {
//     let column_constants = GPUConstants(PoseidonConstants::<Bls12, U11>::new());
//     let tree_constants = GPUConstants(PoseidonConstants::<Bls12, U8>::new());

//     let state = ctx
//         .init_512m(
//             tree_constants.arity_tag(&ctx)?,
//             tree_constants.round_keys(&ctx)?,
//             tree_constants.mds_matrix(&ctx)?,
//             tree_constants.pre_sparse_matrix(&ctx)?,
//             tree_constants.sparse_matrixes(&ctx)?,
//             column_constants.arity_tag(&ctx)?,
//             column_constants.round_keys(&ctx)?,
//             column_constants.mds_matrix(&ctx)?,
//             column_constants.pre_sparse_matrix(&ctx)?,
//             column_constants.sparse_matrixes(&ctx)?,
//         )
//         .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

//     Ok(state)
// }

// fn init_column_tree_builder_4g(ctx: &mut FutharkContext) -> Result<CTB4GState, Error> {
//     let column_constants = GPUConstants(PoseidonConstants::<Bls12, U11>::new());
//     let tree_constants = GPUConstants(PoseidonConstants::<Bls12, U8>::new());

//     let state = ctx
//         .init_4g(
//             tree_constants.arity_tag(&ctx)?,
//             tree_constants.round_keys(&ctx)?,
//             tree_constants.mds_matrix(&ctx)?,
//             tree_constants.pre_sparse_matrix(&ctx)?,
//             tree_constants.sparse_matrixes(&ctx)?,
//             column_constants.arity_tag(&ctx)?,
//             column_constants.round_keys(&ctx)?,
//             column_constants.mds_matrix(&ctx)?,
//             column_constants.pre_sparse_matrix(&ctx)?,
//             column_constants.sparse_matrixes(&ctx)?,
//         )
//         .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

//     Ok(state)
// }

fn u64_vec<'a, U: ArrayLength<Fr>>(vec: &'a [GenericArray<Fr, U>]) -> Vec<u64> {
    vec![0; vec.len() * U::to_usize() * std::mem::size_of::<Fr>()]
}

// fn add_columns_2k(
//     ctx: &mut FutharkContext,
//     state: CTB2KState,
//     columns: &[GenericArray<Fr, U11>],
// ) -> Result<CTB2KState, Error> {
//     let flat_columns = as_u64s(columns);
//     dbg!(&flat_columns.len());
//     // let flat_columns = u64_vec(columns);

//     // let x = Array_u64_1d::from_vec(*ctx, &flat_columns, &[flat_columns.len() as i64, 1])
//     //     .map_err(|_| Error::Other("could not convert".to_string()))?;

//     assert_eq!(flat_columns.len(), 4 * columns.len() * 11);
//     ctx.add_columns_2k(
//         state,
//         Array_u64_1d::from_vec(*ctx, &flat_columns, &[flat_columns.len() as i64, 1])
//             .map_err(|_| Error::Other("could not convert".to_string()))?,
//     )
//     .map_err(|e| Error::GPUError(format!("{:?}", e)))
// }

// fn finalize_2k(
//     ctx: &mut FutharkContext,
//     state: CTB2KState,
// ) -> Result<(Vec<Fr>, CTB2KState), Error> {
//     let (res, state) = ctx
//         .finalize_2k(state)
//         .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

//     let (vec, shape) = res.to_vec();
//     let unpacked = unpack_fr_array((vec, shape.as_slice()))?;
//     Ok((unpacked, state))
// }

// fn add_columns_4g(
//     ctx: &mut FutharkContext,
//     state: CTB4GState,
//     columns: &[GenericArray<Fr, U11>],
// ) -> Result<CTB4GState, Error> {
//     let flat_columns = as_u64s(columns);
//     dbg!(&flat_columns.len());
//     // let flat_columns = u64_vec(columns);

//     // let x = Array_u64_1d::from_vec(*ctx, &flat_columns, &[flat_columns.len() as i64, 1])
//     //     .map_err(|_| Error::Other("could not convert".to_string()))?;

//     assert_eq!(flat_columns.len(), 4 * columns.len() * 11);
//     ctx.add_columns_4g(
//         state,
//         Array_u64_1d::from_vec(*ctx, &flat_columns, &[flat_columns.len() as i64, 1])
//             .map_err(|_| Error::Other("could not convert".to_string()))?,
//     )
//     .map_err(|e| Error::GPUError(format!("{:?}", e)))
// }

// fn finalize_4g(
//     ctx: &mut FutharkContext,
//     state: CTB4GState,
// ) -> Result<(Vec<Fr>, CTB4GState), Error> {
//     let (res, state) = ctx
//         .finalize_4g(state)
//         .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

//     let (vec, shape) = res.to_vec();
//     let unpacked = unpack_fr_array((vec, shape.as_slice()))?;
//     Ok((unpacked, state))
// }

// fn add_columns_512m(
//     ctx: &mut FutharkContext,
//     state: CTB512MState,
//     columns: &[GenericArray<Fr, U11>],
// ) -> Result<CTB512MState, Error> {
//     let flat_columns = as_u64s(columns);
//     dbg!(&flat_columns.len());
//     // let flat_columns = u64_vec(columns);

//     // let x = Array_u64_1d::from_vec(*ctx, &flat_columns, &[flat_columns.len() as i64, 1])
//     //     .map_err(|_| Error::Other("could not convert".to_string()))?;

//     assert_eq!(flat_columns.len(), 4 * columns.len() * 11);
//     ctx.add_columns_512m(
//         state,
//         Array_u64_1d::from_vec(*ctx, &flat_columns, &[flat_columns.len() as i64, 1])
//             .map_err(|_| Error::Other("could not convert".to_string()))?,
//     )
//     .map_err(|e| Error::GPUError(format!("{:?}", e)))
// }

// fn finalize_512m(
//     ctx: &mut FutharkContext,
//     state: CTB512MState,
// ) -> Result<(Vec<Fr>, CTB512MState), Error> {
//     let (res, state) = ctx
//         .finalize_512m(state)
//         .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

//     let (vec, shape) = res.to_vec();
//     let unpacked = unpack_fr_array((vec, shape.as_slice()))?;
//     Ok((unpacked, state))
// }

fn test_binary(ctx: &mut FutharkContext, preimage: [Fr; 3]) -> Result<Fr, Error>
where
{
    let preimage_u64s = frs_to_u64s(&preimage);

    let (vec, shape) = ctx
        .test2(
            Array_u64_1d::from_vec(*ctx, &preimage_u64s, &[preimage.len() as i64 * 4, 1])
                .map_err(|_| Error::Other("could not convert".to_string()))?,
        )
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?
        .to_vec();

    unpack_fr_array((vec, shape.as_slice())).map(|frs| frs[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon::poseidon;
    use ff::Field;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_simple11() {
        let res = simple11(5).unwrap();
        // This just tests that the expected number of values were returned, for now.
        // TODO: verify the results.
        assert_eq!(5, res.len());
    }

    #[test]
    fn test_hash_binary() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();

        let mut state = init_hash2(&mut ctx).unwrap();
        let poseidon_constants = PoseidonConstants::<Bls12, U2>::new();

        for i in 0..1000 {
            let a = Fr::random(&mut rng);
            let b = Fr::random(&mut rng);
            let preimage = [a, b];
            let cpu_res = Poseidon::new_with_preimage(&preimage, &poseidon_constants).hash();
            let (gpu_res, new_state) = hash_binary(&mut ctx, state, &preimage).unwrap();
            state = new_state;

            assert_eq!(
                cpu_res, gpu_res,
                "GPU result ({:?}) differed from CPU ({:?}) result).",
                gpu_res, cpu_res
            );
        }
    }

    #[test]
    fn test_hash_oct() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();

        let mut state = init_hash8(&mut ctx).unwrap();
        let poseidon_constants = PoseidonConstants::<Bls12, U8>::new();

        for i in 0..1000 {
            let preimage: Vec<Fr> = (0..8).map(|_| Fr::random(&mut rng)).collect();

            let cpu_res = Poseidon::new_with_preimage(&preimage, &poseidon_constants).hash();
            let (gpu_res, new_state) = hash_oct(&mut ctx, state, preimage).unwrap();
            state = new_state;

            assert_eq!(
                cpu_res, gpu_res,
                "GPU result ({:?}) differed from CPU ({:?}) result).",
                gpu_res, cpu_res
            );
        }
    }

    #[test]
    fn test_hash_column() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();

        let mut state = init_hash11(&mut ctx).unwrap();
        let poseidon_constants = PoseidonConstants::<Bls12, U11>::new();

        for i in 0..1000 {
            let preimage: Vec<Fr> = (0..11).map(|_| Fr::random(&mut rng)).collect();

            let cpu_res = Poseidon::new_with_preimage(&preimage, &poseidon_constants).hash();
            let (gpu_res, new_state) = hash_column(&mut ctx, state, preimage).unwrap();
            state = new_state;

            assert_eq!(
                cpu_res, gpu_res,
                "GPU result ({:?}) differed from CPU ({:?}) result).",
                gpu_res, cpu_res
            );
        }
    }

    #[test]
    fn test_test_binary() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);

        let mut ctx = FutharkContext::new();

        for i in 0..100 {
            let a = Fr::random(&mut rng);
            let b = Fr::random(&mut rng);
            let c = Fr::random(&mut rng);
            let input = [a, b, c];
            let gpu_res = test_binary(&mut ctx, input).unwrap();

            let mut cpu_res = a.clone();
            cpu_res.mul_assign(&b);
            let cpu_mul_res = cpu_res.clone();
            cpu_res.add_assign(&c);

            dbg!(&a, &b, &c, &cpu_mul_res, &cpu_res, &gpu_res);

            assert_eq!(
                cpu_res, gpu_res,
                "GPU result ({:?}) differed from CPU ({:?}) result).",
                gpu_res, cpu_res
            );
        }
    }

    // #[test]
    // fn test_column_tree_builder_2k() {
    //     let leaves = 64;
    //     let num_batches = 8;
    //     let batch_size = leaves / num_batches;

    //     let mut ctb = ColumnTreeBuilder2k::new(64);

    //     for i in 0..(num_batches - 1) {
    //         let columns: Vec<GenericArray<Fr, U11>> = (0..batch_size)
    //             .map(|_| GenericArray::<Fr, U11>::generate(|i| Fr::zero()))
    //             .collect();

    //         ctb.add_columns(&columns);
    //     }
    //     // let columns: Vec<GenericArray<Fr, U11>> = (0..batch_size)
    //     //     .map(|_| GenericArray::<Fr, U11>::generate(|i| Fr::zero()))
    //     //     .collect();

    //     // let rest = ctb.add_final_columns(&columns).unwrap();
    // }
    use crate::column_tree_builder::ColumnTreeBuilder;

    // #[test]
    // fn test_direct_column_tree_builder_2k() {
    //     let leaves = 64;
    //     let mut ctx = FutharkContext::new();
    //     println!("start");
    //     let mut cpu_builder = ColumnTreeBuilder::<Bls12, U11, U8>::new(leaves);
    //     println!("initialized CPU");
    //     let state = init_column_tree_builder_2k(&mut ctx).unwrap();
    //     println!("initialize GPU");

    //     let columns: Vec<GenericArray<Fr, U11>> = (0..leaves)
    //         .map(|_| GenericArray::<Fr, U11>::generate(|i| Fr::zero()))
    //         .collect();

    //     let state = add_columns_2k(&mut ctx, state, columns.as_slice()).unwrap();
    //     println!("added GPU columns");
    //     let (res, state) = finalize_2k(&mut ctx, state).unwrap();
    //     println!("finalized GPU");

    //     let cpu_res = cpu_builder.add_final_columns(columns.as_slice()).unwrap();
    //     assert_eq!(cpu_res.len(), res.len());
    //     assert_eq!(cpu_res, res);

    //     let computed_root = cpu_builder
    //         .compute_uniform_tree_root(GenericArray::<Fr, U11>::generate(|i| Fr::zero()))
    //         .unwrap();

    //     assert_eq!(computed_root, res[res.len() - 1]);
    // }

    // #[test]
    // fn test_direct_column_tree_builder_4g() {
    //     let leaves = 134217728;
    //     let num_batches = 128;
    //     let batch_size = leaves / num_batches;

    //     let mut ctx = FutharkContext::new();

    //     // let mut cpu_builder = ColumnTreeBuilder::<Bls12, U11, U8>::new(leaves);

    //     let state = init_column_tree_builder_4g(&mut ctx).unwrap();

    //     // for i in 0..batch_size {
    //     //     let columns: Vec<GenericArray<Fr, U11>> = (0..leaves)
    //     //         .map(|_| GenericArray::<Fr, U11>::generate(|i| Fr::zero()))
    //     //         .collect();

    //     //     dbg!(&i, &columns.len());
    //     //     state = add_columns_4g(&mut ctx, state, columns.as_slice()).unwrap();
    //     // }

    //     let (res, state) = finalize_4g(&mut ctx, state).unwrap();

    //     // let cpu_res = cpu_builder.add_final_columns(columns.as_slice()).unwrap();

    //     // assert_eq!(cpu_res.len(), res.len());
    //     // assert_eq!(cpu_res, res);
    // }
    // #[test]
    // // Poor-man's benchmark, for now.
    // #[test]
    // #[ignore]
    // fn test_direct_column_tree_builder_512m() {
    //     let leaves = 1 << 24; // 2^29 / 32 = 16777216;
    //     let _ = test_direct_column_tree_builder_512m_aux(leaves);
    // }

    // #[test]
    // fn test_direct_column_tree_builder_512m_verify() {
    //     let leaves = 1 << 24; // 2^29 / 32 = 16777216;
    //     let root = test_direct_column_tree_builder_512m_aux(leaves);

    //     let mut cpu_builder = ColumnTreeBuilder::<Bls12, U11, U8>::new(leaves);

    //     let computed_root = cpu_builder
    //         .compute_uniform_tree_root(GenericArray::<Fr, U11>::generate(|i| Fr::zero()))
    //         .unwrap();

    //     assert_eq!(computed_root, root);
    // }

    // fn test_direct_column_tree_builder_512m_aux(leaves: usize) -> Fr {
    //     let num_batches = 128;
    //     let batch_size = leaves / num_batches;

    //     let mut ctx = FutharkContext::new();

    //     // let mut cpu_builder = ColumnTreeBuilder::<Bls12, U11, U8>::new(leaves);

    //     let mut state = init_column_tree_builder_512m(&mut ctx).unwrap();

    //     for i in 0..num_batches {
    //         let columns: Vec<GenericArray<Fr, U11>> = (0..batch_size)
    //             .map(|_| GenericArray::<Fr, U11>::generate(|i| Fr::zero()))
    //             .collect();

    //         dbg!(&i, &columns.len());
    //         state = add_columns_512m(&mut ctx, state, columns.as_slice()).unwrap();
    //     }

    //     let (res, state) = finalize_512m(&mut ctx, state).unwrap();

    //     // let cpu_res = cpu_builder.add_final_columns(columns.as_slice()).unwrap();
    //     // assert_eq!(cpu_res.len(), res.len());
    //     // assert_eq!(cpu_res, res);

    //     res[res.len() - 1]
    // }

    #[test]
    fn test_batch_hash2() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();
        let state = init_hash2(&mut ctx).unwrap();
        let batch_size = 100;
        let arity = 2;

        let poseidon_constants = PoseidonConstants::<Bls12, U2>::new();

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U2>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, state) = batch_hash2(&mut ctx, state, preimages.as_slice()).unwrap();
        let expected_hashes: Vec<_> = preimages
            .iter()
            .map(|preimage| Poseidon::new_with_preimage(&preimage, &poseidon_constants).hash())
            .collect();

        let (gpu_binary_res, _new_state) = hash_binary(&mut ctx, state, &preimages[0]).unwrap();
        dbg!(gpu_binary_res);
        assert_eq!(expected_hashes, hashes);
    }

    #[test]
    fn test_mbatch_hash2() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();
        let state = init_hash2(&mut ctx).unwrap();
        let batch_size = 100;
        let arity = 2;

        let poseidon_constants = PoseidonConstants::<Bls12, U2>::new();

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U2>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, state) = mbatch_hash2(&mut ctx, state, preimages.as_slice()).unwrap();
        let expected_hashes: Vec<_> = preimages
            .iter()
            .map(|preimage| Poseidon::new_with_preimage(&preimage, &poseidon_constants).hash())
            .collect();

        let (gpu_binary_res, _new_state) = hash_binary(&mut ctx, state, &preimages[0]).unwrap();
        dbg!(gpu_binary_res);
        assert_eq!(expected_hashes.len(), hashes.len());
        assert_eq!(expected_hashes, hashes);
    }

    #[test]
    fn test_batch_hash8() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();
        let state = init_hash8(&mut ctx).unwrap();
        let batch_size = 100;
        let arity = 2;

        let poseidon_constants = PoseidonConstants::<Bls12, U8>::new();

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U8>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, state) = batch_hash8(&mut ctx, state, preimages.as_slice()).unwrap();
        let expected_hashes: Vec<_> = preimages
            .iter()
            .map(|preimage| Poseidon::new_with_preimage(&preimage, &poseidon_constants).hash())
            .collect();

        assert_eq!(expected_hashes, hashes);
    }

    #[test]
    fn test_mbatch_hash8() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();
        let state = init_hash8(&mut ctx).unwrap();
        let batch_size = 100;
        let arity = 2;

        let poseidon_constants = PoseidonConstants::<Bls12, U8>::new();

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U8>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, state) = mbatch_hash8(&mut ctx, state, preimages.as_slice()).unwrap();
        let expected_hashes: Vec<_> = preimages
            .iter()
            .map(|preimage| Poseidon::new_with_preimage(&preimage, &poseidon_constants).hash())
            .collect();

        assert_eq!(expected_hashes, hashes);
    }

    #[test]
    fn test_batch_hash11() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();
        let state = init_hash11(&mut ctx).unwrap();
        let batch_size = 100;
        let arity = 2;

        let poseidon_constants = PoseidonConstants::<Bls12, U11>::new();

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U11>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, state) = batch_hash11(&mut ctx, state, preimages.as_slice()).unwrap();
        let expected_hashes: Vec<_> = preimages
            .iter()
            .map(|preimage| Poseidon::new_with_preimage(&preimage, &poseidon_constants).hash())
            .collect();

        assert_eq!(expected_hashes, hashes);
    }

    #[test]
    fn test_mbatch_hash11() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();
        let state = init_hash11(&mut ctx).unwrap();
        let batch_size = 100;
        let arity = 2;

        let poseidon_constants = PoseidonConstants::<Bls12, U11>::new();

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U11>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, state) = mbatch_hash11(&mut ctx, state, preimages.as_slice()).unwrap();
        let expected_hashes: Vec<_> = preimages
            .iter()
            .map(|preimage| Poseidon::new_with_preimage(&preimage, &poseidon_constants).hash())
            .collect();

        assert_eq!(expected_hashes, hashes);
    }
}
