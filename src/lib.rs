// Ported from <https://github.com/python/cpython/blob/master/Python/marshal.c>
use bitflags::bitflags;
pub use bstr;
use bstr::{BStr, BString};
use num_bigint::BigInt;
use num_complex::Complex;
use num_derive::{FromPrimitive, ToPrimitive};
use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    fmt,
    hash::{Hash, Hasher},
    iter::FromIterator,
    sync::{Arc, RwLock},
};

pub mod read;

/// `Arc` = immutable
/// `ArcRwLock` = mutable
pub type ArcRwLock<T> = Arc<RwLock<T>>;

#[derive(FromPrimitive, ToPrimitive, Debug, Copy, Clone)]
#[repr(u8)]
pub enum Type {
    Null          = b'0',
    None          = b'N',
    False         = b'F',
    True          = b'T',
    StopIter      = b'S',
    Ellipsis      = b'.',
    Int           = b'i',
    Int64         = b'I',
    Float         = b'f',
    BinaryFloat   = b'g',
    Complex       = b'x',
    BinaryComplex = b'y',
    Long          = b'l',
    String        = b's',
    Interned      = b't',
    StringRef     = b'R',
    Tuple         = b'(',
    List          = b'[',
    Dict          = b'{',
    Code          = b'c',
    Unicode       = b'u',
    Unknown       = b'?',
    Set           = b'<',
    FrozenSet     = b'>',
    /// Not a real type
    Bytes         = 0x0,
}
impl Type {
    const FLAG_REF: u8 = b'\x80';
}

pub(crate) struct Depth(Arc<()>);
impl Depth {
    const MAX: usize = 900;

    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(()))
    }

    pub fn try_clone(&self) -> Option<Self> {
        if Arc::strong_count(&self.0) > Self::MAX {
            None
        } else {
            Some(Self(self.0.clone()))
        }
    }
}
impl fmt::Debug for Depth {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        f.debug_tuple("Depth")
            .field(&Arc::strong_count(&self.0))
            .finish()
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
pub struct Code {
    pub argcount:        u32,
    pub nlocals:         u32,
    pub stacksize:       u32,
    pub flags:           CodeFlags,
    pub code:            Arc<Vec<u8>>,
    pub consts:          Arc<Vec<Obj>>,
    pub names:           Vec<Arc<BString>>,
    pub varnames:        Vec<Arc<BString>>,
    pub freevars:        Vec<Arc<BString>>,
    pub cellvars:        Vec<Arc<BString>>,
    pub filename:        Arc<BString>,
    pub name:            Arc<BString>,
    pub firstlineno:     u32,
    pub lnotab:          Arc<Vec<u8>>,
}

#[rustfmt::skip]
#[derive(Clone)]
pub enum Obj {
    None,
    StopIteration,
    Ellipsis,
    Bool     (bool),
    Long     (Arc<BigInt>),
    Float    (f64),
    Complex  (Complex<f64>),
    Bytes    (Arc<Vec<u8>>),
    String   (Arc<BString>),
    Tuple    (Arc<Vec<Obj>>),
    List     (ArcRwLock<Vec<Obj>>),
    Dict     (ArcRwLock<HashMap<ObjHashable, Obj>>),
    Set      (ArcRwLock<HashSet<ObjHashable>>),
    FrozenSet(Arc<HashSet<ObjHashable>>),
    Code     (Arc<Code>),
    // etc.
}
macro_rules! define_extract {
    ($extract_fn:ident($variant:ident) -> ()) => {
        define_extract! { $extract_fn -> () { $variant => () } }
    };
    ($extract_fn:ident($variant:ident) -> Arc<$ret:ty>) => {
        define_extract! { $extract_fn -> Arc<$ret> { $variant(x) => x } }
    };
    ($extract_fn:ident($variant:ident) -> ArcRwLock<$ret:ty>) => {
        define_extract! { $extract_fn -> ArcRwLock<$ret> { $variant(x) => x } }
    };
    ($extract_fn:ident($variant:ident) -> $ret:ty) => {
        define_extract! { $extract_fn -> $ret { $variant(x) => x } }
    };
    ($extract_fn:ident -> $ret:ty { $variant:ident$(($($pat:pat),+))? => $expr:expr }) => {
        /// # Errors
        /// Returns a reference to self if extraction fails
        pub fn $extract_fn(self) -> Result<$ret, Self> {
            if let Self::$variant$(($($pat),+))? = self {
                Ok($expr)
            } else {
                Err(self)
            }
        }
    }
}
macro_rules! define_is {
    ($is_fn:ident($variant:ident$(($($pat:pat),+))?)) => {
        /// # Errors
        /// Returns a reference to self if extraction fails
        #[must_use]
        pub fn $is_fn(&self) -> bool {
            if let Self::$variant$(($($pat),+))? = self {
                true
            } else {
                false
            }
        }
    }
}
impl Obj {
    define_extract! { extract_none          (None)          -> ()                                    }
    define_extract! { extract_stop_iteration(StopIteration) -> ()                                    }
    define_extract! { extract_bool          (Bool)          -> bool                                  }
    define_extract! { extract_long          (Long)          -> Arc<BigInt>                           }
    define_extract! { extract_float         (Float)         -> f64                                   }
    define_extract! { extract_bytes         (String)        -> Arc<BString>                          }
    define_extract! { extract_string        (String)        -> Arc<BString>                          }
    define_extract! { extract_tuple         (Tuple)         -> Arc<Vec<Self>>                        }
    define_extract! { extract_list          (List)          -> ArcRwLock<Vec<Self>>                  }
    define_extract! { extract_dict          (Dict)          -> ArcRwLock<HashMap<ObjHashable, Self>> }
    define_extract! { extract_set           (Set)           -> ArcRwLock<HashSet<ObjHashable>>       }
    define_extract! { extract_frozenset     (FrozenSet)     -> Arc<HashSet<ObjHashable>>             }
    define_extract! { extract_code          (Code)          -> Arc<Code>                             }

    define_is! { is_none          (None)          }
    define_is! { is_stop_iteration(StopIteration) }
    define_is! { is_bool          (Bool(_))       }
    define_is! { is_long          (Long(_))       }
    define_is! { is_float         (Float(_))      }
    define_is! { is_bytes         (Bytes(_))      }
    define_is! { is_string        (String(_))     }
    define_is! { is_tuple         (Tuple(_))      }
    define_is! { is_list          (List(_))       }
    define_is! { is_dict          (Dict(_))       }
    define_is! { is_set           (Set(_))        }
    define_is! { is_frozenset     (FrozenSet(_))  }
    define_is! { is_code          (Code(_))       }
}
/// Should mostly match Python's repr
///
/// # Float, Complex
/// - Uses `float('...')` instead of `...` for nan, inf, and -inf.
/// - Uses Rust's float-to-decimal conversion.
///
/// # Bytes, String
/// - Always uses double-quotes
/// - Escapes both kinds of quotes
///
/// # Code
/// - Uses named arguments for readability
/// - lnotab is formatted as bytes(...) with a list of integers, instead of a bytes literal
impl fmt::Debug for Obj {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::StopIteration => write!(f, "StopIteration"),
            Self::Ellipsis => write!(f, "Ellipsis"),
            Self::Bool(true) => write!(f, "True"),
            Self::Bool(false) => write!(f, "False"),
            Self::Long(x) => write!(f, "{}", x),
            &Self::Float(x) => python_float_repr_full(f, x),
            &Self::Complex(x) => python_complex_repr(f, x),
            Self::Bytes(x) => python_bytes_repr(f, x),
            Self::String(x) => python_string_repr(f, x.as_ref().as_ref()),
            Self::Tuple(x) => python_tuple_repr(f, x),
            Self::List(x) => f.debug_list().entries(x.read().unwrap().iter()).finish(),
            Self::Dict(x) => f.debug_map().entries(x.read().unwrap().iter()).finish(),
            Self::Set(x) => f.debug_set().entries(x.read().unwrap().iter()).finish(),
            Self::FrozenSet(x) => python_frozenset_repr(f, x),
            Self::Code(x) => python_code_repr(f, x),
        }
    }
}

impl Obj {
    pub fn typ(&self) -> Type {
        match self {
            Self::None => Type::None,
            Self::StopIteration => Type::StopIter,
            Self::Ellipsis => Type::Ellipsis,
            Self::Bool(true) => Type::True,
            Self::Bool(false) => Type::False,
            Self::Long(x) => Type::Long,
            &Self::Float(x) => Type::Float,
            &Self::Complex(x) => Type::Complex,
            Self::Bytes(x) => Type::Bytes,
            Self::String(x) => Type::String,
            Self::Tuple(x) => Type::Tuple,
            Self::List(x) => Type::List,
            Self::Dict(x) => Type::Dict,
            Self::Set(x) => Type::Set,
            Self::FrozenSet(x) => Type::FrozenSet,
            Self::Code(x) => Type::Code,
        }
    }
}

fn python_float_repr_full(f: &mut fmt::Formatter, x: f64) -> fmt::Result {
    python_float_repr_core(f, x)?;
    if x.fract() == 0. {
        write!(f, ".0")?;
    };
    Ok(())
}
fn python_float_repr_core(f: &mut fmt::Formatter, x: f64) -> fmt::Result {
    if x.is_nan() {
        write!(f, "float('nan')")
    } else if x.is_infinite() {
        if x.is_sign_positive() {
            write!(f, "float('inf')")
        } else {
            write!(f, "-float('inf')")
        }
    } else {
        // properly handle -0.0
        if x.is_sign_negative() {
            write!(f, "-")?;
        }
        write!(f, "{}", x.abs())
    }
}
fn python_complex_repr(f: &mut fmt::Formatter, x: Complex<f64>) -> fmt::Result {
    if x.re == 0. && x.re.is_sign_positive() {
        python_float_repr_core(f, x.im)?;
        write!(f, "j")?;
    } else {
        write!(f, "(")?;
        python_float_repr_core(f, x.re)?;
        if x.im >= 0. || x.im.is_nan() {
            write!(f, "+")?;
        }
        python_float_repr_core(f, x.im)?;
        write!(f, "j)")?;
    };
    Ok(())
}
fn python_bytes_repr(f: &mut fmt::Formatter, x: &[u8]) -> fmt::Result {
    write!(f, "b\"")?;
    for &byte in x.iter() {
        match byte {
            b'\t' => write!(f, "\\t")?,
            b'\n' => write!(f, "\\n")?,
            b'\r' => write!(f, "\\r")?,
            b'\'' | b'"' | b'\\' => write!(f, "\\{}", char::from(byte))?,
            b' '..=b'~' => write!(f, "{}", char::from(byte))?,
            _ => write!(f, "\\x{:02x}", byte)?,
        }
    }
    write!(f, "\"")?;
    Ok(())
}
fn python_string_repr(f: &mut fmt::Formatter, x: &BStr) -> fmt::Result {
    let original = format!("{:?}", x);
    let mut last_end = 0;
    // Note: the behavior is arbitrary if there are improper escapes.
    for (start, _) in original.match_indices("\\u{") {
        f.write_str(&original[last_end..start])?;
        let len = original[start..].find('}').ok_or(fmt::Error)? + 1;
        let end = start + len;
        match len - 4 {
            0..=2 => write!(f, "\\x{:0>2}", &original[start + 3..end - 1])?,
            3..=4 => write!(f, "\\u{:0>4}", &original[start + 3..end - 1])?,
            5..=8 => write!(f, "\\U{:0>8}", &original[start + 3..end - 1])?,
            _ => panic!("Internal error: length of unicode escape = {} > 8", len),
        }
        last_end = end;
    }
    f.write_str(&original[last_end..])?;
    Ok(())
}
fn python_tuple_repr(f: &mut fmt::Formatter, x: &[Obj]) -> fmt::Result {
    if x.is_empty() {
        f.write_str("()") // Otherwise this would get formatted into an empty string
    } else {
        let mut debug_tuple = f.debug_tuple("");
        for o in x.iter() {
            debug_tuple.field(&o);
        }
        debug_tuple.finish()
    }
}
fn python_frozenset_repr(f: &mut fmt::Formatter, x: &HashSet<ObjHashable>) -> fmt::Result {
    f.write_str("frozenset(")?;
    if !x.is_empty() {
        f.debug_set().entries(x.iter()).finish()?;
    }
    f.write_str(")")?;
    Ok(())
}
fn python_code_repr(f: &mut fmt::Formatter, x: &Code) -> fmt::Result {
    write!(f, "code(argcount={:?}, nlocals={:?}, stacksize={:?}, flags={:?}, code={:?}, consts={:?}, names={:?}, varnames={:?}, freevars={:?}, cellvars={:?}, filename={:?}, name={:?}, firstlineno={:?}, lnotab=bytes({:?}))", x.argcount, x.nlocals, x.stacksize, x.flags, Obj::Bytes(Arc::clone(&x.code)), x.consts, x.names, x.varnames, x.freevars, x.cellvars, x.filename, x.name, x.firstlineno, &x.lnotab)
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

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum ObjHashable {
    None,
    StopIteration,
    Ellipsis,
    Bool(bool),
    Long(Arc<BigInt>),
    Float(HashF64),
    Complex(Complex<HashF64>),
    String(Arc<BString>),
    Tuple(Arc<Vec<ObjHashable>>),
    FrozenSet(Arc<HashableHashSet<ObjHashable>>),
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
            Obj::Long(x) => Ok(Self::Long(Arc::clone(x))),
            Obj::Float(x) => Ok(Self::Float(HashF64(*x))),
            Obj::Complex(Complex { re, im }) => Ok(Self::Complex(Complex {
                re: HashF64(*re),
                im: HashF64(*im),
            })),
            Obj::String(x) => Ok(Self::String(Arc::clone(x))),
            Obj::Tuple(x) => Ok(Self::Tuple(Arc::new(
                x.iter()
                    .map(Self::try_from)
                    .collect::<Result<Vec<Self>, Obj>>()?,
            ))),
            Obj::FrozenSet(x) => Ok(Self::FrozenSet(Arc::new(
                x.iter().cloned().collect::<HashableHashSet<Self>>(),
            ))),
            x => Err(x.clone()),
        }
    }
}
impl fmt::Debug for ObjHashable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::StopIteration => write!(f, "StopIteration"),
            Self::Ellipsis => write!(f, "Ellipsis"),
            Self::Bool(true) => write!(f, "True"),
            Self::Bool(false) => write!(f, "False"),
            Self::Long(x) => write!(f, "{}", x),
            Self::Float(x) => python_float_repr_full(f, x.0),
            Self::Complex(x) => python_complex_repr(
                f,
                Complex {
                    re: x.re.0,
                    im: x.im.0,
                },
            ),
            Self::String(x) => python_string_repr(f, x.as_ref().as_ref()),
            Self::Tuple(x) => python_tuple_hashable_repr(f, x),
            Self::FrozenSet(x) => python_frozenset_repr(f, &x.0),
        }
    }
}
fn python_tuple_hashable_repr(f: &mut fmt::Formatter, x: &[ObjHashable]) -> fmt::Result {
    if x.is_empty() {
        f.write_str("()") // Otherwise this would get formatted into an empty string
    } else {
        let mut debug_tuple = f.debug_tuple("");
        for o in x.iter() {
            debug_tuple.field(&o);
        }
        debug_tuple.finish()
    }
}

#[cfg(test)]
mod test {
    use super::{Code, CodeFlags, Obj, ObjHashable};
    use bstr::{BString, ByteSlice};
    use num_bigint::BigInt;
    use num_complex::Complex;
    use std::{
        collections::{HashMap, HashSet},
        sync::{Arc, RwLock},
    };

    #[test]
    fn test_debug_repr() {
        assert_eq!(format!("{:?}", Obj::None), "None");
        assert_eq!(format!("{:?}", Obj::StopIteration), "StopIteration");
        assert_eq!(format!("{:?}", Obj::Ellipsis), "Ellipsis");
        assert_eq!(format!("{:?}", Obj::Bool(true)), "True");
        assert_eq!(format!("{:?}", Obj::Bool(false)), "False");
        assert_eq!(
            format!("{:?}", Obj::Long(Arc::new(BigInt::from(-123)))),
            "-123"
        );
        assert_eq!(format!("{:?}", Obj::Tuple(Arc::new(vec![]))), "()");
        assert_eq!(
            format!("{:?}", Obj::Tuple(Arc::new(vec![Obj::Bool(true)]))),
            "(True,)"
        );
        assert_eq!(
            format!(
                "{:?}",
                Obj::Tuple(Arc::new(vec![Obj::Bool(true), Obj::None]))
            ),
            "(True, None)"
        );
        assert_eq!(
            format!(
                "{:?}",
                Obj::List(Arc::new(RwLock::new(vec![Obj::Bool(true)])))
            ),
            "[True]"
        );
        assert_eq!(
            format!(
                "{:?}",
                Obj::Dict(Arc::new(RwLock::new(
                    vec![(
                        ObjHashable::Bool(true),
                        Obj::Bytes(Arc::new(Vec::from(b"a" as &[u8])))
                    )]
                    .into_iter()
                    .collect::<HashMap<_, _>>()
                )))
            ),
            "{True: b\"a\"}"
        );
        assert_eq!(
            format!(
                "{:?}",
                Obj::Set(Arc::new(RwLock::new(
                    vec![ObjHashable::Bool(true)]
                        .into_iter()
                        .collect::<HashSet<_>>()
                )))
            ),
            "{True}"
        );
        assert_eq!(
            format!(
                "{:?}",
                Obj::FrozenSet(Arc::new(
                    vec![ObjHashable::Bool(true)]
                        .into_iter()
                        .collect::<HashSet<_>>()
                ))
            ),
            "frozenset({True})"
        );
        assert_eq!(format!("{:?}", Obj::Code(Arc::new(Code {
            argcount: 0,
            nlocals: 3,
            stacksize: 4,
            flags: CodeFlags::NESTED | CodeFlags::COROUTINE,
             code: Arc::new(Vec::from(b"abc" as &[u8])),
            consts: Arc::new(vec![Obj::Bool(true)]),
            names: vec![],
            varnames: vec![Arc::new(BString::from("a"))],
            freevars: vec![Arc::new(BString::from("b")), Arc::new(BString::from("c"))],
            cellvars: vec![Arc::new(BString::from("de"))],
            filename: Arc::new(BString::from("xyz.py")),
            name: Arc::new(BString::from("fgh")),
            firstlineno: 5,
            lnotab: Arc::new(vec![255, 0, 45, 127, 0, 73]),
        }))), "code(argcount=0, nlocals=3, stacksize=4, flags=NESTED | COROUTINE, code=b\"abc\", consts=[True], names=[], varnames=[\"a\"], freevars=[\"b\", \"c\"], cellvars=[\"de\"], filename=\"xyz.py\", name=\"fgh\", firstlineno=5, lnotab=bytes([255, 0, 45, 127, 0, 73]))");
    }

    #[test]
    fn test_float_debug_repr() {
        assert_eq!(format!("{:?}", Obj::Float(1.23)), "1.23");
        assert_eq!(format!("{:?}", Obj::Float(f64::NAN)), "float('nan')");
        assert_eq!(format!("{:?}", Obj::Float(f64::INFINITY)), "float('inf')");
        assert_eq!(format!("{:?}", Obj::Float(-f64::INFINITY)), "-float('inf')");
        assert_eq!(format!("{:?}", Obj::Float(0.0)), "0.0");
        assert_eq!(format!("{:?}", Obj::Float(-0.0)), "-0.0");
    }

    #[test]
    fn test_complex_debug_repr() {
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 2., im: 1. })),
            "(2+1j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 0., im: 1. })),
            "1j"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 2., im: 0. })),
            "(2+0j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 0., im: 0. })),
            "0j"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -2., im: 1. })),
            "(-2+1j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -2., im: 0. })),
            "(-2+0j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 2., im: -1. })),
            "(2-1j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 0., im: -1. })),
            "-1j"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -2., im: -1. })),
            "(-2-1j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: 0., im: -1. })),
            "-1j"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -2., im: 0. })),
            "(-2+0j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -0., im: 1. })),
            "(-0+1j)"
        );
        assert_eq!(
            format!("{:?}", Obj::Complex(Complex { re: -0., im: -1. })),
            "(-0-1j)"
        );
    }

    #[test]
    fn test_bytes_string_debug_repr() {
        assert_eq!(format!("{:?}", Obj::Bytes(Arc::new(Vec::from(
                            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe" as &[u8]
                            )))),
        "b\"\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\t\\n\\x0b\\x0c\\r\\x0e\\x0f\\x10\\x11\\x12\\x13\\x14\\x15\\x16\\x17\\x18\\x19\\x1a\\x1b\\x1c\\x1d\\x1e\\x1f !\\\"#$%&\\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\\x7f\\x80\\x81\\x82\\x83\\x84\\x85\\x86\\x87\\x88\\x89\\x8a\\x8b\\x8c\\x8d\\x8e\\x8f\\x90\\x91\\x92\\x93\\x94\\x95\\x96\\x97\\x98\\x99\\x9a\\x9b\\x9c\\x9d\\x9e\\x9f\\xa0\\xa1\\xa2\\xa3\\xa4\\xa5\\xa6\\xa7\\xa8\\xa9\\xaa\\xab\\xac\\xad\\xae\\xaf\\xb0\\xb1\\xb2\\xb3\\xb4\\xb5\\xb6\\xb7\\xb8\\xb9\\xba\\xbb\\xbc\\xbd\\xbe\\xbf\\xc0\\xc1\\xc2\\xc3\\xc4\\xc5\\xc6\\xc7\\xc8\\xc9\\xca\\xcb\\xcc\\xcd\\xce\\xcf\\xd0\\xd1\\xd2\\xd3\\xd4\\xd5\\xd6\\xd7\\xd8\\xd9\\xda\\xdb\\xdc\\xdd\\xde\\xdf\\xe0\\xe1\\xe2\\xe3\\xe4\\xe5\\xe6\\xe7\\xe8\\xe9\\xea\\xeb\\xec\\xed\\xee\\xef\\xf0\\xf1\\xf2\\xf3\\xf4\\xf5\\xf6\\xf7\\xf8\\xf9\\xfa\\xfb\\xfc\\xfd\\xfe\""
        );
        assert_eq!(format!("{:?}", Obj::String(Arc::new(BString::from(
                            "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f")))),
                            "\"\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\t\\n\\x0b\\x0c\\r\\x0e\\x0f\\x10\\x11\\x12\\x13\\x14\\x15\\x16\\x17\\x18\\x19\\x1a\\x1b\\x1c\\x1d\\x1e\\x1f !\\\"#$%&\\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\\x7f\"");
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
