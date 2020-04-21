use crate::error::Error;
use crate::poseidon::PoseidonConstants;
use ff::{PrimeField, PrimeFieldDecodingError};
use generic_array::{typenum, ArrayLength, GenericArray};
use paired::bls12_381::{Bls12, Fr, FrRepr};
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

pub fn u64s_into_fr(limbs: &[u64]) -> Result<Fr, PrimeFieldDecodingError> {
    assert_eq!(limbs.len(), 4);
    let mut limb_arr = [0; 4];
    limb_arr.copy_from_slice(&limbs[..]);
    let repr = FrRepr(limb_arr);
    let fr = Fr::from_repr(repr);

    fr
}

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
    assert_eq!(
        std::mem::size_of::<Fr>(),
        fr_size * std::mem::size_of::<u64>()
    );

    if monts.len() % fr_size != 0 {
        return Err(Error::Other(
            "wrong size monts to convert to Frs".to_string(),
        ));
    }

    let frs = unsafe { std::slice::from_raw_parts(monts.as_ptr() as *const () as _, fr_count) };

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

fn mbatch_hash2(
    ctx: &mut FutharkContext,
    state: P2State,
    preimages: &[GenericArray<Fr, U2>],
) -> Result<(Vec<Fr>, P2State), Error> {
    let flat_preimages = as_mont_u64s(preimages);
    let input = Array_u64_1d::from_vec(*ctx, &flat_preimages, &[flat_preimages.len() as i64, 1])
        .map_err(|_| Error::Other("could not convert".to_string()))?;

    let (res, state) = ctx
        .mbatch_hash2(state, input)
        .map_err(|e| Error::GPUError(format!("{:?}", e)))?;

    let (vec, _shape) = res.to_vec();
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

    let (vec, _shape) = res.to_vec();
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

    let (vec, _shape) = res.to_vec();
    let frs = unpack_fr_array_from_monts(vec.as_slice())?;

    Ok((frs.to_vec(), state))
}

fn u64_vec<'a, U: ArrayLength<Fr>>(vec: &'a [GenericArray<Fr, U>]) -> Vec<u64> {
    vec![0; vec.len() * U::to_usize() * std::mem::size_of::<Fr>()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon::{poseidon, Poseidon, SimplePoseidonBatchHasher};
    use crate::BatchHasher;
    use ff::{Field, ScalarEngine};
    use generic_array::sequence::GenericSequence;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_mbatch_hash2() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();
        let state = init_hash2(&mut ctx).unwrap();
        let batch_size = 100;
        let arity = 2;

        let mut simple_hasher = SimplePoseidonBatchHasher::<Bls12, U2>::new();

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U2>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, state) = mbatch_hash2(&mut ctx, state, preimages.as_slice()).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages);

        assert_eq!(expected_hashes.len(), hashes.len());
        assert_eq!(expected_hashes, hashes);
    }

    #[test]
    fn test_mbatch_hash8() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();
        let state = init_hash8(&mut ctx).unwrap();
        let batch_size = 100;
        let arity = 2;

        let mut simple_hasher = SimplePoseidonBatchHasher::<Bls12, U8>::new();

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U8>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, state) = mbatch_hash8(&mut ctx, state, preimages.as_slice()).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages);

        assert_eq!(expected_hashes, hashes);
    }

    #[test]
    fn test_mbatch_hash11() {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut ctx = FutharkContext::new();
        let state = init_hash11(&mut ctx).unwrap();
        let batch_size = 100;
        let arity = 2;

        let mut simple_hasher = SimplePoseidonBatchHasher::<Bls12, U11>::new();

        let preimages = (0..batch_size)
            .map(|_| GenericArray::<Fr, U11>::generate(|_| Fr::random(&mut rng)))
            .collect::<Vec<_>>();

        let (hashes, state) = mbatch_hash11(&mut ctx, state, preimages.as_slice()).unwrap();
        let expected_hashes: Vec<_> = simple_hasher.hash(&preimages);

        assert_eq!(expected_hashes, hashes);
    }
}
