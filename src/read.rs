#[allow(clippy::wildcard_imports)] // read::errors
pub mod errors {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum ErrorKind {
        #[error("invalid type: 0x{0:X}")]
        InvalidType(u8),
        #[error("recursion limit exceeded")]
        RecursionLimitExceeded,
        #[error("digit is out of range: {0}")]
        DigitOutOfRange(u16),
        #[error("unnormalized long")]
        UnnormalizedLong,
        #[error("null object")]
        IsNull,
        #[error("encountered unhashable: {0:?}")]
        Unhashable(crate::Obj),
        #[error("type error: {0:?}")]
        TypeError(crate::Obj),
        #[error("invalid reference")]
        InvalidRef,

        #[error("IO error occurred: {0}")]
        Io(#[from] std::io::Error),
        #[error("Error occurred while creating a str: {0}")]
        Utf8(#[from] std::str::Utf8Error),
        #[error("Error occurred while creating a string from utf8: {0}")]
        FromUtf8(#[from] std::string::FromUtf8Error),
        #[error("Error occurred while parsing float: {0}")]
        ParseFloat(#[from] std::num::ParseFloatError),
    }
}

use self::errors::*;
use crate::{Code, CodeFlags, Depth, Obj, ObjHashable, Type, utils};
use bstr::BString;
use num_bigint::BigInt;
use num_complex::Complex;
use num_traits::{FromPrimitive, Zero};
use std::{
    collections::{HashMap, HashSet},
    io::Read,
    str::FromStr,
    sync::{Arc, RwLock},
};

pub type ParseResult<T> = Result<T, ErrorKind>;

/// Index of a slot in the [`RefTable`].
#[derive(Copy, Clone, Debug)]
struct RefIdx(usize);

/// The `FLAG_REF` reference table. Marshal lets later objects back-reference
/// earlier ones by index; recursive containers reserve their slot before their
/// contents are parsed so they can refer to themselves.
#[derive(Default)]
struct RefTable(Vec<Obj>);

impl RefTable {
    /// Reserve a slot (filled with a placeholder) to be backfilled via [`Self::set`].
    fn reserve(&mut self) -> RefIdx {
        let idx = RefIdx(self.0.len());
        self.0.push(Obj::None);
        idx
    }

    /// Append an already-built object, returning its slot.
    fn push(&mut self, obj: Obj) -> RefIdx {
        let idx = RefIdx(self.0.len());
        self.0.push(obj);
        idx
    }

    fn set(&mut self, idx: RefIdx, obj: Obj) {
        self.0[idx.0] = obj;
    }
}

/// The `TYPE_STRINGREF` table of interned strings, indexed by appearance order.
#[derive(Default)]
struct StringRefs(Vec<Obj>);

impl StringRefs {
    fn push(&mut self, obj: Obj) {
        self.0.push(obj);
    }

    fn get(&self, idx: usize) -> Option<&Obj> {
        self.0.get(idx)
    }
}

struct RFile<R: Read> {
    depth: Depth,
    readable: R,
    refs: RefTable,
    stringrefs: StringRefs,
}

macro_rules! define_r {
    ($ident:ident -> $ty:ty; $n:literal) => {
        fn $ident(p: &mut RFile<impl Read>) -> ParseResult<$ty> {
            let mut buf: [u8; $n] = [0; $n];
            p.readable.read_exact(&mut buf)?;
            Ok(<$ty>::from_le_bytes(buf))
        }
    };
}

define_r! { r_byte      -> u8 ; 1 }
define_r! { r_short     -> u16; 2 }
define_r! { r_long      -> u32; 4 }
define_r! { r_long64    -> u64; 8 }
define_r! { r_float_bin -> f64; 8 }

fn r_bytes(n: usize, p: &mut RFile<impl Read>) -> ParseResult<Vec<u8>> {
    let mut buf = vec![0u8; n];
    p.readable.read_exact(&mut buf)?;
    Ok(buf)
}

fn r_string(n: usize, p: &mut RFile<impl Read>) -> ParseResult<String> {
    let buf = r_bytes(n, p)?;
    Ok(String::from_utf8(buf)?)
}

fn r_bstring(n: usize, p: &mut RFile<impl Read>) -> ParseResult<BString> {
    let buf = r_bytes(n, p)?;
    Ok(BString::from(buf))
}

fn r_float_str(p: &mut RFile<impl Read>) -> ParseResult<f64> {
    let n = r_byte(p)?;
    let s = r_string(n as usize, p)?;
    Ok(f64::from_str(s.as_ref())?)
}

// TODO: test
/// May misbehave on 16-bit platforms.
fn r_pylong(p: &mut RFile<impl Read>) -> ParseResult<BigInt> {
    #[allow(clippy::cast_possible_wrap)]
    let n = r_long(p)? as i32;
    if n == 0 {
        return Ok(BigInt::zero());
    };
    #[allow(clippy::cast_sign_loss)]
    let size = n.wrapping_abs() as u32;
    let mut digits = Vec::<u16>::with_capacity(size as usize);
    for _ in 0..size {
        let d = r_short(p)?;
        if d > (1 << 15) {
            return Err(ErrorKind::DigitOutOfRange(d));
        }
        digits.push(d);
    }
    if digits[(size - 1) as usize] == 0 {
        return Err(ErrorKind::UnnormalizedLong);
    }
    Ok(BigInt::from_biguint(utils::sign_of(&n), utils::biguint_from_pylong_digits(&digits)))
}

fn r_vec(n: usize, p: &mut RFile<impl Read>) -> ParseResult<Vec<Obj>> {
    let mut vec = Vec::with_capacity(n);
    for _ in 0..n {
        vec.push(r_object_not_null(p)?);
    }
    Ok(vec)
}

fn r_hashmap(p: &mut RFile<impl Read>) -> ParseResult<HashMap<ObjHashable, Obj>> {
    let mut map = HashMap::new();
    loop {
        match r_object(p)? {
            None => break,
            Some(key) => match r_object(p)? {
                None => break,
                Some(value) => {
                    map.insert(ObjHashable::try_from(&key).map_err(ErrorKind::Unhashable)?, value);
                } // TODO
            },
        }
    }
    Ok(map)
}

fn r_hashset(n: usize, p: &mut RFile<impl Read>) -> ParseResult<HashSet<ObjHashable>> {
    let mut set = HashSet::new();
    r_hashset_into(&mut set, n, p)?;
    Ok(set)
}

fn r_hashset_into(set: &mut HashSet<ObjHashable>, n: usize, p: &mut RFile<impl Read>) -> ParseResult<()> {
    for _ in 0..n {
        set.insert(ObjHashable::try_from(&r_object_not_null(p)?).map_err(ErrorKind::Unhashable)?);
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn r_object(p: &mut RFile<impl Read>) -> ParseResult<Option<Obj>> {
    let code: u8 = r_byte(p)?;
    let _depth_handle = p.depth.try_clone().ok_or(ErrorKind::RecursionLimitExceeded)?;
    let (flag, type_) = {
        let flag: bool = (code & Type::FLAG_REF) != 0;
        let type_u8: u8 = code & !Type::FLAG_REF;
        let type_: Type = Type::from_u8(type_u8).ok_or(ErrorKind::InvalidType(type_u8))?;
        if let Type::Bytes = type_ {
            // this is a fake type
            return Err(ErrorKind::InvalidType(type_u8));
        }
        (flag, type_)
    };
    let mut slot: Option<RefIdx> = match type_ {
        // immutable collections
        Type::Tuple | Type::FrozenSet | Type::Code if flag => Some(p.refs.reserve()),
        _ => None,
    };
    #[allow(clippy::cast_possible_wrap)]
    let retval = match type_ {
        Type::Bytes => unreachable!(),
        Type::Null => None,
        Type::None => Some(Obj::None),
        Type::StopIter => Some(Obj::StopIteration),
        Type::Ellipsis => Some(Obj::Ellipsis),
        Type::False => Some(Obj::Bool(false)),
        Type::True => Some(Obj::Bool(true)),
        Type::Int => Some(Obj::Long(Arc::new(RwLock::new(BigInt::from(r_long(p)? as i32))))),
        Type::Int64 => Some(Obj::Long(Arc::new(RwLock::new(BigInt::from(r_long64(p)? as i64))))),
        Type::Long => Some(Obj::Long(Arc::new(RwLock::new(r_pylong(p)?)))),
        Type::Float => Some(Obj::Float(r_float_str(p)?)),
        Type::BinaryFloat => Some(Obj::Float(r_float_bin(p)?)),
        Type::Complex => Some(Obj::Complex(Complex { re: r_float_str(p)?, im: r_float_str(p)? })),
        Type::BinaryComplex => Some(Obj::Complex(Complex { re: r_float_bin(p)?, im: r_float_bin(p)? })),
        Type::String | Type::Unicode => {
            let obj = Obj::String(Arc::new(RwLock::new(r_bstring(r_long(p)? as usize, p)?)));
            Some(obj)
        }
        Type::StringRef => {
            let n = r_long(p)? as usize;
            let result = p.stringrefs.get(n).ok_or(ErrorKind::InvalidRef)?.clone();
            if result.is_none() {
                return Err(ErrorKind::InvalidRef);
            }
            Some(result)
        }
        Type::Interned => {
            let obj = Obj::String(Arc::new(RwLock::new(r_bstring(r_long(p)? as usize, p)?)));
            p.stringrefs.push(obj.clone());
            Some(obj)
        }
        Type::Tuple => Some(Obj::Tuple(Arc::new(RwLock::new(r_vec(r_long(p)? as usize, p)?)))),
        Type::List => Some(Obj::List(Arc::new(RwLock::new(r_vec(r_long(p)? as usize, p)?)))),
        Type::Set => {
            let set = Arc::new(RwLock::new(HashSet::new()));

            if flag {
                slot = Some(p.refs.push(Obj::Set(Arc::clone(&set))));
            }

            r_hashset_into(&mut set.write().unwrap(), r_long(p)? as usize, p)?;
            Some(Obj::Set(set))
        }
        Type::FrozenSet => Some(Obj::FrozenSet(Arc::new(RwLock::new(r_hashset(r_long(p)? as usize, p)?)))),
        Type::Dict => Some(Obj::Dict(Arc::new(RwLock::new(r_hashmap(p)?)))),
        Type::Code => Some(Obj::Code(Arc::new(RwLock::new(Code {
            argcount: r_long(p)?,
            nlocals: r_long(p)?,
            stacksize: r_long(p)?,
            flags: CodeFlags::from_bits_truncate(r_long(p)?),
            code: r_object_extract_bytes(p)?,
            consts: r_object_extract_tuple(p)?,
            names: r_object_extract_tuple_string(p)?,
            varnames: r_object_extract_tuple_string(p)?,
            freevars: r_object_extract_tuple_string(p)?,
            cellvars: r_object_extract_tuple_string(p)?,
            filename: r_object_extract_string(p)?,
            name: r_object_extract_string(p)?,
            firstlineno: r_long(p)?,
            lnotab: r_object_extract_bytes(p)?,
        })))),
        Type::Unknown => return Err(ErrorKind::InvalidType(Type::Unknown as u8)),
    };
    match (&retval, slot) {
        (None, _)
        | (Some(Obj::None), _)
        | (Some(Obj::StopIteration), _)
        | (Some(Obj::Ellipsis), _)
        | (Some(Obj::Bool(_)), _) => {}
        (Some(x), Some(slot)) if flag => {
            p.refs.set(slot, x.clone());
        }
        (Some(x), None) if flag => {
            p.refs.push(x.clone());
        }
        (Some(_), _) => {}
    };
    Ok(retval)
}

fn r_object_not_null(p: &mut RFile<impl Read>) -> ParseResult<Obj> {
    r_object(p)?.ok_or(ErrorKind::IsNull)
}
fn r_object_extract_string(p: &mut RFile<impl Read>) -> ParseResult<Arc<BString>> {
    let mutex_val = r_object_not_null(p)?.extract_string().map_err(ErrorKind::TypeError)?;
    let guard = mutex_val.read().unwrap();
    let result = Arc::new(guard.clone());
    drop(guard);
    Ok(result)
}
fn r_object_extract_bytes(p: &mut RFile<impl Read>) -> ParseResult<Arc<Vec<u8>>> {
    let mutex_val = r_object_not_null(p)?.extract_string().map_err(ErrorKind::TypeError)?;
    // this forces an allocation but makes some operations easier
    let guard = mutex_val.read().unwrap();
    let result = Arc::new(guard.to_vec());
    drop(guard);
    Ok(result)
}
fn r_object_extract_tuple(p: &mut RFile<impl Read>) -> ParseResult<Arc<Vec<Obj>>> {
    let mutex_val = r_object_not_null(p)?.extract_tuple().map_err(ErrorKind::TypeError)?;
    let guard = mutex_val.read().unwrap();
    let result = Arc::new(guard.clone());
    drop(guard);
    Ok(result)
}
fn r_object_extract_tuple_string(p: &mut RFile<impl Read>) -> ParseResult<Vec<Arc<BString>>> {
    r_object_extract_tuple(p)?
        .iter()
        .map(|x| {
            let mutex_val = x.clone().extract_string().map_err(ErrorKind::TypeError)?;
            let guard = mutex_val.read().unwrap();
            let result = Arc::new(guard.clone());
            drop(guard);
            Ok(result)
        })
        .collect::<ParseResult<Vec<Arc<BString>>>>()
}

fn read_object(p: &mut RFile<impl Read>) -> ParseResult<Obj> {
    r_object_not_null(p)
}

/// # Errors
/// See [`ErrorKind`].
pub fn marshal_load(readable: impl Read) -> ParseResult<Obj> {
    let mut rf = RFile { depth: Depth::new(), readable, refs: RefTable::default(), stringrefs: StringRefs::default() };
    read_object(&mut rf)
}

/// Allows coercion from array reference to slice.
/// # Errors
/// See [`ErrorKind`].
pub fn marshal_loads(bytes: &[u8]) -> ParseResult<Obj> {
    marshal_load(bytes)
}

// Ported from <https://github.com/python/cpython/blob/master/Lib/test/test_marshal.py>
#[cfg(test)]
mod test {
    use super::{Obj, errors, marshal_load, marshal_loads};
    use num_bigint::BigInt;
    use std::io::{self, Read};

    macro_rules! assert_match {
        ($expr:expr, $pat:pat) => {
            match $expr {
                $pat => {}
                _ => panic!(),
            }
        };
    }

    fn load_unwrap(r: impl Read) -> Obj {
        marshal_load(r).unwrap()
    }

    fn loads_unwrap(s: &[u8]) -> Obj {
        load_unwrap(s)
    }

    #[test]
    fn test_ints() {
        assert_eq!(
            BigInt::parse_bytes(b"85070591730234615847396907784232501249", 10).unwrap(),
            *loads_unwrap(b"l\t\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xf0\x7f\xff\x7f\xff\x7f\xff\x7f?\x00")
                .extract_long()
                .unwrap()
                .read()
                .unwrap()
        );
    }

    #[allow(clippy::unreadable_literal)]
    #[test]
    fn test_int64() {
        for mut base in [i64::MAX, i64::MIN, -i64::MAX, -(i64::MIN >> 1)].iter().copied() {
            while base != 0 {
                let mut s = Vec::<u8>::new();
                s.push(b'I');
                s.extend_from_slice(&base.to_le_bytes());
                assert_eq!(BigInt::from(base), *loads_unwrap(&s).extract_long().unwrap().read().unwrap());

                if base == -1 { base = 0 } else { base >>= 1 }
            }
        }

        assert_eq!(
            BigInt::from(0x1032547698badcfe_i64),
            *loads_unwrap(b"I\xfe\xdc\xba\x98\x76\x54\x32\x10").extract_long().unwrap().read().unwrap()
        );
        assert_eq!(
            BigInt::from(-0x1032547698badcff_i64),
            *loads_unwrap(b"I\x01\x23\x45\x67\x89\xab\xcd\xef").extract_long().unwrap().read().unwrap()
        );
        assert_eq!(
            BigInt::from(0x7f6e5d4c3b2a1908_i64),
            *loads_unwrap(b"I\x08\x19\x2a\x3b\x4c\x5d\x6e\x7f").extract_long().unwrap().read().unwrap()
        );
        assert_eq!(
            BigInt::from(-0x7f6e5d4c3b2a1909_i64),
            *loads_unwrap(b"I\xf7\xe6\xd5\xc4\xb3\xa2\x91\x80").extract_long().unwrap().read().unwrap()
        );
    }

    #[test]
    fn test_bool() {
        assert!(loads_unwrap(b"T").extract_bool().unwrap());
        assert!(!loads_unwrap(b"F").extract_bool().unwrap());
    }

    #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
    #[test]
    fn test_floats() {
        assert_eq!((i64::MAX as f64) * 3.7e250, loads_unwrap(b"g\x11\x9f6\x98\xd2\xab\xe4w").extract_float().unwrap());
    }

    // Note: strings, unicode, dicts, sets, and code objects are covered by the
    // Python 2.7 byte fixtures in tests/fixtures.rs.

    #[test]
    fn test_bytes() {
        let b1 = loads_unwrap(b"\xf3\x00\x00\x00\x00").extract_bytes().unwrap();
        assert_eq!(b"", &b1.read().unwrap()[..]);
        let b2 = loads_unwrap(b"\xf3\x0c\x00\x00\x00Andr\xe8 Previn").extract_bytes().unwrap();
        assert_eq!(b"Andr\xe8 Previn", &b2.read().unwrap()[..]);
        let b3 = loads_unwrap(b"\xf3\x03\x00\x00\x00abc").extract_bytes().unwrap();
        assert_eq!(b"abc", &b3.read().unwrap()[..]);
        let b4 = loads_unwrap(&[b"\xf3\x10'\x00\x00" as &[u8], &[b' '; 10_000]].concat()).extract_bytes().unwrap();
        assert_eq!(b" ".repeat(10_000), &b4.read().unwrap()[..]);
    }

    #[test]
    fn test_exceptions() {
        loads_unwrap(b"S").extract_stop_iteration().unwrap();
    }

    // TODO: test_bytearray, test_memoryview, test_array

    #[test]
    fn test_patch_873224() {
        assert_match!(marshal_loads(b"0").unwrap_err(), errors::ErrorKind::IsNull);
        let f_err = marshal_loads(b"f").unwrap_err();
        match f_err {
            errors::ErrorKind::Io(io_err) => {
                assert_eq!(io_err.kind(), io::ErrorKind::UnexpectedEof);
            }
            _ => panic!(),
        }
        let int_err = marshal_loads(b"l\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 ").unwrap_err();
        match int_err {
            errors::ErrorKind::Io(io_err) => {
                assert_eq!(io_err.kind(), io::ErrorKind::UnexpectedEof);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_fuzz() {
        for i in 0..=u8::MAX {
            println!("{:?}", marshal_loads(&[i]));
        }
    }

    /// Warning: this has to be run on a release build to avoid a stack overflow.
    #[cfg(not(debug_assertions))]
    #[test]
    fn test_loads_recursion() {
        loads_unwrap(&[&b"(\x01\x00\x00\x00".repeat(100)[..], b"N"].concat());
        loads_unwrap(&[&b"[\x01\x00\x00\x00".repeat(100)[..], b"N"].concat());
        loads_unwrap(&[&b"{N".repeat(100)[..], b"N", &b"0".repeat(100)[..]].concat());
        loads_unwrap(&[&b">\x01\x00\x00\x00".repeat(100)[..], b"N"].concat());

        assert_match!(
            marshal_loads(&[&b"(\x01\x00\x00\x00".repeat(1048576)[..], b"N"].concat()).unwrap_err(),
            errors::ErrorKind::RecursionLimitExceeded
        );
        assert_match!(
            marshal_loads(&[&b"[\x01\x00\x00\x00".repeat(1048576)[..], b"N"].concat()).unwrap_err(),
            errors::ErrorKind::RecursionLimitExceeded
        );
        assert_match!(
            marshal_loads(&[&b"{N".repeat(1048576)[..], b"N", &b"0".repeat(1048576)[..]].concat()).unwrap_err(),
            errors::ErrorKind::RecursionLimitExceeded
        );
        assert_match!(
            marshal_loads(&[&b">\x01\x00\x00\x00".repeat(1048576)[..], b"N"].concat()).unwrap_err(),
            errors::ErrorKind::RecursionLimitExceeded
        );
    }

    #[test]
    fn test_invalid_longs() {
        assert_match!(
            marshal_loads(b"l\x02\x00\x00\x00\x00\x00\x00\x00").unwrap_err(),
            errors::ErrorKind::UnnormalizedLong
        );
    }
}
