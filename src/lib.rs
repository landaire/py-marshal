/// Ported from <https://github.com/python/cpython/blob/master/Python/marshal.c>
use bitflags::bitflags;
use num_bigint::BigInt;
use num_complex::Complex;
use num_derive::{FromPrimitive, ToPrimitive};
use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    hash::{Hash, Hasher},
    iter::FromIterator,
    rc::Rc,
};

#[derive(FromPrimitive, ToPrimitive, Debug, Copy, Clone)]
#[repr(u8)]
enum Type {
    Null               = b'0',
    None               = b'N',
    False              = b'F',
    True               = b'T',
    StopIter           = b'S',
    Ellipsis           = b'.',
    Int                = b'i',
    Int64              = b'I',
    Float              = b'f',
    BinaryFloat        = b'g',
    Complex            = b'x',
    BinaryComplex      = b'y',
    Long               = b'l',
    String             = b's',
    Interned           = b't',
    Ref                = b'r',
    Tuple              = b'(',
    List               = b'[',
    Dict               = b'{',
    Code               = b'c',
    Unicode            = b'u',
    Unknown            = b'?',
    Set                = b'<',
    FrozenSet          = b'>',
    Ascii              = b'a',
    AsciiInterned      = b'A',
    SmallTuple         = b')',
    ShortAscii         = b'z',
    ShortAsciiInterned = b'Z',
}
impl Type {
    const FLAG_REF: u8 = b'\x80';
}

struct Depth(Rc<()>);
impl Depth {
    const MAX: usize = 2000;

    #[must_use]
    pub fn new() -> Self {
        Self(Rc::new(()))
    }

    pub fn try_clone(&self) -> Option<Self> {
        if Rc::strong_count(&self.0) > Self::MAX {
            None
        } else {
            Some(Self(self.0.clone()))
        }
    }
}

bitflags! {
    pub struct CodeFlags: u32 {
        const OPTIMIZED                   = 0x1;
        const NEWLOCALS                   = 0x2;
        const VARARGS                     = 0x4;
        const VARKEYWORDS                 = 0x8;
        const NESTED                     = 0x10;
        const GENERATOR                  = 0x20;
        const NOFREE                     = 0x40;
        const COROUTINE                  = 0x80;
        const ITERABLE_COROUTINE        = 0x100;
        const ASYNC_GENERATOR           = 0x200;
        // TODO: old versions
        const GENERATOR_ALLOWED        = 0x1000;
        const FUTURE_DIVISION          = 0x2000;
        const FUTURE_ABSOLUTE_IMPORT   = 0x4000;
        const FUTURE_WITH_STATEMENT    = 0x8000;
        const FUTURE_PRINT_FUNCTION   = 0x10000;
        const FUTURE_UNICODE_LITERALS = 0x20000;
        const FUTURE_BARRY_AS_BDFL    = 0x40000;
        const FUTURE_GENERATOR_STOP   = 0x80000;
        #[allow(clippy::unreadable_literal)]
        const FUTURE_ANNOTATIONS     = 0x100000;
    }
}

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Obj {
    None,
    StopIteration,
    Ellipsis,
    Bool(     bool),
    Long(     Rc<BigInt>),
    Float(    f64),
    Complex(  Complex<f64>),
    Bytes(    Rc<Vec<u8>>),
    String(   Rc<String>),
    Tuple(    Rc<Vec<Obj>>),
    List(     Rc<Vec<Obj>>),
    Dict(     Rc<HashMap<ObjHashable, Obj>>),
    Set(      Rc<HashSet<ObjHashable>>),
    FrozenSet(Rc<HashSet<ObjHashable>>),
    Code {
        argcount:        u32,
        posonlyargcount: u32,
        kwonlyargcount:  u32,
        nlocals:         u32,
        stacksize:       u32,
        flags:           CodeFlags,
        code:            Rc<Vec<u8>>,
        consts:          Rc<Vec<Obj>>,
        names:           Rc<Vec<String>>,
        varnames:        Rc<Vec<String>>,
        freevars:        Rc<Vec<String>>,
        cellvars:        Rc<Vec<String>>,
        filename:        Rc<String>,
        name:            Rc<String>,
        firstlineno:     u32,
        lnotab:          Rc<Vec<u8>>,
    },
    // etc.
}
macro_rules! define_extract {
    ($extract_fn:ident($variant:ident) -> Rc<$ret:ty>) => {
        #[must_use]
        pub fn $extract_fn(&self) -> Option<Rc<$ret>> {
            if let Self::$variant(x) = self {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    };
    ($extract_fn:ident($variant:ident) -> $ret:ty) => {
        #[must_use]
        pub fn $extract_fn(&self) -> Option<$ret> {
            if let Self::$variant(x) = self {
                Some(*x)
            } else {
                None
            }
        }
    };
}
impl Obj {
    define_extract! { extract_bool  (Bool)   -> bool }
    define_extract! { extract_long  (Long)   -> Rc<BigInt> }
    define_extract! { extract_bytes (Bytes)  -> Rc<Vec<u8>> }
    define_extract! { extract_string(String) -> Rc<String> }
    define_extract! { extract_tuple (Tuple)  -> Rc<Vec<Self>> }
}

/// This is a f64 wrapper suitable for use as a key in a (Hash)Map, since NaNs compare equal to
/// each other, so it can implement Eq and Hash. `HashF64(-0.0) == HashF64(0.0)`.
#[derive(Clone, Debug)]
pub struct HashF64(f64);
impl PartialEq for HashF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
    }
}
impl Eq for HashF64 {}
impl Hash for HashF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.0.is_nan() {
            // Multiple NaN values exist
            state.write_u8(0);
        } else if self.0 == 0.0 {
            // 0.0 == -0.0
            state.write_u8(1);
        } else {
            state.write_u64(self.0.to_bits()); // This should be fine, since all the dupes should be accounted for.
        }
    }
}

#[derive(Debug)]
pub struct HashableHashSet<T>(HashSet<T>);
impl<T> Hash for HashableHashSet<T>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut xor: u64 = 0;
        let hasher = std::collections::hash_map::DefaultHasher::new();
        for value in &self.0 {
            let mut hasher_clone = hasher.clone();
            value.hash(&mut hasher_clone);
            xor ^= hasher_clone.finish();
        }
        state.write_u64(xor);
    }
}
impl<T> PartialEq for HashableHashSet<T>
where
    T: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> Eq for HashableHashSet<T> where T: Eq + Hash {}
impl<T> FromIterator<T> for HashableHashSet<T>
where
    T: Eq + Hash,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self(iter.into_iter().collect())
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum ObjHashable {
    None,
    StopIteration,
    Ellipsis,
    Bool(bool),
    Long(Rc<BigInt>),
    Float(HashF64),
    Complex(Complex<HashF64>),
    String(Rc<String>),
    Tuple(Rc<Vec<ObjHashable>>),
    FrozenSet(Rc<HashableHashSet<ObjHashable>>),
    // etc.
}

impl TryFrom<&Obj> for ObjHashable {
    type Error = Obj;

    fn try_from(orig: &Obj) -> Result<Self, Obj> {
        match orig {
            Obj::None => Ok(Self::None),
            Obj::StopIteration => Ok(Self::StopIteration),
            Obj::Ellipsis => Ok(Self::Ellipsis),
            Obj::Bool(x) => Ok(Self::Bool(*x)),
            Obj::Long(x) => Ok(Self::Long(Rc::clone(x))),
            Obj::Float(x) => Ok(Self::Float(HashF64(*x))),
            Obj::Complex(Complex { re, im }) => Ok(Self::Complex(Complex {
                re: HashF64(*re),
                im: HashF64(*im),
            })),
            Obj::String(x) => Ok(Self::String(Rc::clone(x))),
            Obj::Tuple(x) => Ok(Self::Tuple(Rc::new(
                x.iter()
                    .map(Self::try_from)
                    .collect::<Result<Vec<Self>, Obj>>()?,
            ))),
            Obj::FrozenSet(x) => Ok(Self::FrozenSet(Rc::new(
                x.iter().cloned().collect::<HashableHashSet<Self>>(),
            ))),
            x => Err(x.clone()),
        }
    }
}

mod utils {
    use num_bigint::{BigUint, Sign};
    use num_traits::Zero;
    use std::cmp::Ordering;

    /// Based on `_PyLong_AsByteArray` in <https://github.com/python/cpython/blob/master/Objects/longobject.c>
    #[allow(clippy::cast_possible_truncation)]
    pub fn biguint_from_pylong_digits(digits: &[u16]) -> BigUint {
        if digits.is_empty() {
            return BigUint::zero();
        };
        assert!(digits[digits.len() - 1] != 0);
        let mut accum: u64 = 0;
        let mut accumbits: u8 = 0;
        let mut p = Vec::<u32>::new();
        for (i, &thisdigit) in digits.iter().enumerate() {
            accum |= u64::from(thisdigit) << accumbits;
            accumbits += if i == digits.len() - 1 {
                16 - (thisdigit.leading_zeros() as u8)
            } else {
                15
            };

            // Modified to get u32s instead of u8s.
            while accumbits >= 32 {
                p.push(accum as u32);
                accumbits -= 32;
                accum >>= 32;
            }
        }
        assert!(accumbits < 32);
        if accumbits > 0 {
            p.push(accum as u32);
        }
        BigUint::new(p)
    }

    pub fn sign_of<T: Ord + Zero>(x: &T) -> Sign {
        match x.cmp(&T::zero()) {
            Ordering::Less => Sign::Minus,
            Ordering::Equal => Sign::NoSign,
            Ordering::Greater => Sign::Plus,
        }
    }

    #[cfg(test)]
    mod test {
        use super::biguint_from_pylong_digits;
        use num_bigint::BigUint;

        #[allow(clippy::inconsistent_digit_grouping)]
        #[test]
        fn test_biguint_from_pylong_digits() {
            assert_eq!(
                biguint_from_pylong_digits(&[
                    0b000_1101_1100_0100,
                    0b110_1101_0010_0100,
                    0b001_0000_1001_1101
                ]),
                BigUint::from(0b001_0000_1001_1101_110_1101_0010_0100_000_1101_1100_0100_u64)
            );
        }
    }
}

pub mod read {
    use crate::{utils, CodeFlags, Depth, Obj, ObjHashable, Type};
    use num_bigint::BigInt;
    use num_complex::Complex;
    use num_traits::{FromPrimitive, Zero};
    use std::{
        collections::{HashMap, HashSet},
        convert::TryFrom,
        io::{self, Read},
        num::ParseFloatError,
        rc::Rc,
        str::{FromStr, Utf8Error},
        string::FromUtf8Error,
    };

    #[allow(clippy::pub_enum_variant_names)]
    #[derive(Debug)]
    pub enum Error {
        IoError(io::Error),
        InvalidType(u8),
        RecursionLimitExceeded,
        DigitOutOfRange(u16),
        UnnormalizedLong,
        Utf8Error(Utf8Error),
        FromUtf8Error(FromUtf8Error),
        ParseFloatError(ParseFloatError),
        IsNull,
        Unhashable(Obj),
        TypeError,
        InvalidRef,
    }

    type Result<T> = std::result::Result<T, Error>;

    struct RFile<R: Read> {
        depth: Depth,
        readable: R,
        refs: Vec<Obj>,
    }

    macro_rules! define_r {
        ($ident:ident -> $ty:ty; $n:literal) => {
            fn $ident(p: &mut RFile<impl Read>) -> Result<$ty> {
                let mut buf: [u8; $n] = [0; $n];
                p.readable.read_exact(&mut buf).map_err(Error::IoError)?;
                Ok(<$ty>::from_le_bytes(buf))
            }
        };
    }

    define_r! { r_byte      -> u8 ; 1 }
    define_r! { r_short     -> u16; 2 }
    define_r! { r_long      -> u32; 4 }
    define_r! { r_long64    -> u64; 8 }
    define_r! { r_float_bin -> f64; 8 }

    fn r_bytes(n: usize, p: &mut RFile<impl Read>) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        buf.resize(n, 0);
        p.readable.read_exact(&mut buf).map_err(Error::IoError)?;
        Ok(buf)
    }

    fn r_string(n: usize, p: &mut RFile<impl Read>) -> Result<String> {
        let buf = r_bytes(n, p)?;
        Ok(String::from_utf8(buf).map_err(Error::FromUtf8Error)?)
    }

    fn r_float_str(p: &mut RFile<impl Read>) -> Result<f64> {
        let n = r_byte(p)?;
        let s = r_string(n as usize, p)?;
        Ok(f64::from_str(&s).map_err(Error::ParseFloatError)?)
    }

    // TODO: test
    /// May misbehave on 16-bit platforms.
    fn r_pylong(p: &mut RFile<impl Read>) -> Result<BigInt> {
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
                return Err(Error::DigitOutOfRange(d));
            }
            digits.push(d);
        }
        if digits[(size - 1) as usize] == 0 {
            return Err(Error::UnnormalizedLong);
        }
        Ok(BigInt::from_biguint(
            utils::sign_of(&n),
            utils::biguint_from_pylong_digits(&digits),
        ))
    }

    fn r_ref(o: &Option<Obj>, p: &mut RFile<impl Read>) {
        if let Some(x) = o {
            p.refs.push(x.clone());
        }
    }

    fn r_vec(n: usize, p: &mut RFile<impl Read>) -> Result<Vec<Obj>> {
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(r_object_not_null(p)?);
        }
        Ok(vec)
    }

    fn r_hashmap(p: &mut RFile<impl Read>) -> Result<HashMap<ObjHashable, Obj>> {
        let mut map = HashMap::new();
        loop {
            match r_object(p)? {
                None => break,
                Some(key) => match r_object(p)? {
                    None => break,
                    Some(value) => {
                        map.insert(
                            ObjHashable::try_from(&key).map_err(Error::Unhashable)?,
                            value,
                        );
                    } // TODO
                },
            }
        }
        Ok(map)
    }

    fn r_hashset(n: usize, p: &mut RFile<impl Read>) -> Result<HashSet<ObjHashable>> {
        let mut set = HashSet::new();
        for _ in 0..n {
            set.insert(ObjHashable::try_from(&r_object_not_null(p)?).map_err(Error::Unhashable)?);
        }
        Ok(set)
    }

    fn r_object(p: &mut RFile<impl Read>) -> Result<Option<Obj>> {
        let code: u8 = r_byte(p)?;
        let _depth_handle = p
            .depth
            .try_clone()
            .map_or(Err(Error::RecursionLimitExceeded), Ok)?;
        let (flag, type_) = {
            let flag: bool = (code & Type::FLAG_REF) != 0;
            let type_u8: u8 = code & !Type::FLAG_REF;
            let type_: Type =
                Type::from_u8(type_u8).map_or(Err(Error::InvalidType(type_u8)), Ok)?;
            (flag, type_)
        };
        #[rustfmt::skip]
        #[allow(clippy::cast_possible_wrap)]
        let retval = match type_ {
            Type::Null          => None,
            Type::None          => Some(Obj::None),
            Type::StopIter      => Some(Obj::StopIteration),
            Type::Ellipsis      => Some(Obj::Ellipsis),
            Type::False         => Some(Obj::Bool(false)),
            Type::True          => Some(Obj::Bool(true)),
            Type::Int           => Some(Obj::Long(Rc::new(BigInt::from(r_long(p)?   as i32)))),
            Type::Int64         => Some(Obj::Long(Rc::new(BigInt::from(r_long64(p)? as i64)))),
            Type::Long          => Some(Obj::Long(Rc::new(r_pylong(p)?))),
            Type::Float         => Some(Obj::Float(r_float_str(p)?)),
            Type::BinaryFloat   => Some(Obj::Float(r_float_bin(p)?)),
            Type::Complex       => Some(Obj::Complex(Complex {
                re: r_float_str(p)?,
                im: r_float_str(p)?,
            })),
            Type::BinaryComplex => Some(Obj::Complex(Complex {
                re: r_float_bin(p)?,
                im: r_float_bin(p)?,
            })),
            Type::String => {
                Some(Obj::Bytes( Rc::new(r_bytes( r_long(p)? as usize, p)?)))
            }
            Type::AsciiInterned | Type::Ascii | Type::Interned | Type::Unicode => {
                Some(Obj::String(Rc::new(r_string(r_long(p)? as usize, p)?)))
            }
            Type::ShortAsciiInterned | Type::ShortAscii => {
                Some(Obj::String(Rc::new(r_string(r_byte(p)? as usize, p)?)))
            }
            Type::SmallTuple => Some(Obj::Tuple(    Rc::new(r_vec(    r_byte(p)? as usize, p)?))),
            Type::Tuple      => Some(Obj::Tuple(    Rc::new(r_vec(    r_long(p)? as usize, p)?))),
            Type::List       => Some(Obj::List(     Rc::new(r_vec(    r_long(p)? as usize, p)?))),
            Type::Set        => Some(Obj::Set(      Rc::new(r_hashset(r_long(p)? as usize, p)?))),
            Type::FrozenSet  => Some(Obj::FrozenSet(Rc::new(r_hashset(r_long(p)? as usize, p)?))),
            Type::Dict       => Some(Obj::Dict(     Rc::new(r_hashmap(p)?))),
            Type::Code       => Some(Obj::Code {
                argcount:        r_long(p)?,
                posonlyargcount: r_long(p)?,
                kwonlyargcount:  r_long(p)?,
                nlocals:         r_long(p)?,
                stacksize:       r_long(p)?,
                flags:           CodeFlags::from_bits_truncate(r_long(p)?),
                code:            r_object_extract_bytes(p)?,
                consts:          r_object_extract_tuple(p)?,
                names:           r_object_extract_tuple_string(p)?,
                varnames:        r_object_extract_tuple_string(p)?,
                freevars:        r_object_extract_tuple_string(p)?,
                cellvars:        r_object_extract_tuple_string(p)?,
                filename:        r_object_extract_string(p)?,
                name:            r_object_extract_string(p)?,
                firstlineno:     r_long(p)?,
                lnotab:          r_object_extract_bytes(p)?,
            }),
            Type::Ref => {
                let n = r_long(p)? as usize;
                Some(p.refs.get(n).ok_or(Error::InvalidRef)?.clone())
            }
            Type::Unknown => return Err(Error::InvalidType(Type::Unknown as u8)),
        };
        match retval {
            None
            | Some(Obj::None)
            | Some(Obj::StopIteration)
            | Some(Obj::Ellipsis)
            | Some(Obj::Bool(_)) => {}
            Some(_) if flag => r_ref(&retval, p),
            Some(_) => {}
        };
        Ok(retval)
    }

    fn r_object_not_null(p: &mut RFile<impl Read>) -> Result<Obj> {
        Ok(r_object(p)?.ok_or(Error::IsNull)?)
    }
    fn r_object_extract_string(p: &mut RFile<impl Read>) -> Result<Rc<String>> {
        Ok(r_object_not_null(p)?
            .extract_string()
            .ok_or(Error::TypeError)?)
    }
    fn r_object_extract_bytes(p: &mut RFile<impl Read>) -> Result<Rc<Vec<u8>>> {
        Ok(r_object_not_null(p)?
            .extract_bytes()
            .ok_or(Error::TypeError)?)
    }
    fn r_object_extract_tuple(p: &mut RFile<impl Read>) -> Result<Rc<Vec<Obj>>> {
        Ok(r_object_not_null(p)?
            .extract_tuple()
            .ok_or(Error::TypeError)?)
    }
    fn r_object_extract_tuple_string(p: &mut RFile<impl Read>) -> Result<Rc<Vec<String>>> {
        Ok(Rc::new(
            r_object_extract_tuple(p)?
                .iter()
                .map(|x| {
                    x.extract_string()
                        .as_deref()
                        .cloned()
                        .ok_or(Error::TypeError)
                })
                .collect::<Result<Vec<String>>>()?,
        ))
    }

    fn read_object(p: &mut RFile<impl Read>) -> Result<Option<Obj>> {
        r_object(p)
    }

    /// # Errors
    /// See [`Error`].
    pub fn marshal_loads(bytes: &[u8]) -> Result<Option<Obj>> {
        let mut rf = RFile {
            depth: Depth::new(),
            readable: bytes,
            refs: Vec::<Obj>::new(),
        };
        read_object(&mut rf)
    }

    /// Ported from <https://github.com/python/cpython/blob/master/Lib/test/test_marshal.py>
    #[cfg(test)]
    mod test {
        use super::{marshal_loads, Obj};
        use num_bigint::BigInt;

        fn loads_unwrap(s: &[u8]) -> Obj {
            marshal_loads(s).unwrap().unwrap()
        }

        #[allow(clippy::unreadable_literal)]
        #[test]
        fn test_int64() {
            for mut base in [i64::MAX, i64::MIN, -i64::MAX, -(i64::MIN >> 1)]
                .iter()
                .copied()
            {
                while base != 0 {
                    let mut s = Vec::<u8>::new();
                    s.push(b'I');
                    s.extend_from_slice(&base.to_le_bytes());
                    assert_eq!(
                        BigInt::from(base),
                        *loads_unwrap(&s).extract_long().unwrap()
                    );

                    if base == -1 {
                        base = 0
                    } else {
                        base >>= 1
                    }
                }
            }

            assert_eq!(
                BigInt::from(0x1032547698badcfe_i64),
                *loads_unwrap(b"I\xfe\xdc\xba\x98\x76\x54\x32\x10")
                    .extract_long()
                    .unwrap()
            );
            assert_eq!(
                BigInt::from(-0x1032547698badcff_i64),
                *loads_unwrap(b"I\x01\x23\x45\x67\x89\xab\xcd\xef")
                    .extract_long()
                    .unwrap()
            );
            assert_eq!(
                BigInt::from(0x7f6e5d4c3b2a1908_i64),
                *loads_unwrap(b"I\x08\x19\x2a\x3b\x4c\x5d\x6e\x7f")
                    .extract_long()
                    .unwrap()
            );
            assert_eq!(
                BigInt::from(-0x7f6e5d4c3b2a1909_i64),
                *loads_unwrap(b"I\xf7\xe6\xd5\xc4\xb3\xa2\x91\x80")
                    .extract_long()
                    .unwrap()
            );
        }

        #[test]
        fn test_bool() {
            assert_eq!(true, loads_unwrap(b"T").extract_bool().unwrap());
            assert_eq!(false, loads_unwrap(b"F").extract_bool().unwrap());
        }
    }
}
