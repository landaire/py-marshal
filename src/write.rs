//! Serializer for the Python 2.7 marshal format -- the inverse of [`crate::read`].
//!
//! Python's own `marshal.c` shares immutable objects through a reference table
//! (the `FLAG_REF` / `TYPE_REF` / `TYPE_STRINGREF` machinery). That table is a
//! *size* optimization, not a correctness requirement: `marshal.loads` reads a
//! stream that writes every object inline just fine. So this writer is
//! deliberately ref-free -- it emits each object in full. The result is a few
//! bytes larger than CPython's but loads back to an equal value, which is all we
//! need to round-trip deobfuscated code objects without depending on Python.
use crate::{Code, Obj, ObjHashable, Type};
use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};
use std::convert::TryFrom;
use std::io::{self, Write};

/// Serialize an object to a byte vector.
#[must_use]
pub fn marshal_dumps(obj: &Obj) -> Vec<u8> {
    let mut buf = Vec::new();
    // Writing into a `Vec` is infallible.
    w_object(obj, &mut buf).expect("writing to an in-memory buffer cannot fail");
    buf
}

/// Serialize an object into the given writer.
///
/// # Errors
/// Propagates any I/O error from the underlying writer.
pub fn marshal_dump(obj: &Obj, w: &mut impl Write) -> io::Result<()> {
    w_object(obj, w)
}

fn w_type(t: Type, w: &mut impl Write) -> io::Result<()> {
    w.write_all(&[t as u8])
}

/// 4-byte little-endian length/value, matching the reader's `r_long`.
fn w_long(n: u32, w: &mut impl Write) -> io::Result<()> {
    w.write_all(&n.to_le_bytes())
}

/// 2-byte little-endian, matching the reader's `r_short`.
fn w_short(n: u16, w: &mut impl Write) -> io::Result<()> {
    w.write_all(&n.to_le_bytes())
}

/// Write a raw byte string body: 4-byte length followed by the bytes.
fn w_byte_block(data: &[u8], w: &mut impl Write) -> io::Result<()> {
    w_long(data.len() as u32, w)?;
    w.write_all(data)
}

/// Write a `TYPE_STRING` object. Python 2.7 has no distinct bytes type, so byte
/// strings, text, and code/lnotab blobs all serialize the same way.
fn w_string(data: &[u8], w: &mut impl Write) -> io::Result<()> {
    w_type(Type::String, w)?;
    w_byte_block(data, w)
}

/// Encode an arbitrary-precision integer as `TYPE_LONG`: a signed digit count
/// (negative when the value is negative) followed by base-2^15 digits, exactly
/// what `read::r_pylong` decodes.
fn w_pylong(value: &BigInt, w: &mut impl Write) -> io::Result<()> {
    w_type(Type::Long, w)?;

    let sign = value.sign();
    let mut mag: BigUint = value.magnitude().clone();
    let mask = BigUint::from(0x7fffu16);
    let mut digits: Vec<u16> = Vec::new();
    while !mag.is_zero() {
        // Low 15 bits of the remaining magnitude.
        let digit = (&mag & &mask).to_u16().expect("15-bit masked value always fits a u16");
        digits.push(digit);
        mag >>= 15u32;
    }

    let count = digits.len() as i32;
    let signed_count = if let num_bigint::Sign::Minus = sign { -count } else { count };
    w_long(signed_count as u32, w)?;
    for digit in digits {
        w_short(digit, w)?;
    }
    Ok(())
}

/// Write an integer, preferring the compact `TYPE_INT` form when it fits in an
/// `i32` (the common case) and falling back to `TYPE_LONG` otherwise.
fn w_int(value: &BigInt, w: &mut impl Write) -> io::Result<()> {
    if let Some(small) = value.to_i32() {
        w_type(Type::Int, w)?;
        w_long(small as u32, w)
    } else {
        w_pylong(value, w)
    }
}

/// Write a tuple of byte strings (used for code object name lists).
fn w_string_tuple(items: &[std::sync::Arc<bstr::BString>], w: &mut impl Write) -> io::Result<()> {
    w_type(Type::Tuple, w)?;
    w_long(items.len() as u32, w)?;
    for item in items {
        w_string(item.as_ref().as_ref(), w)?;
    }
    Ok(())
}

fn w_code(code: &Code, w: &mut impl Write) -> io::Result<()> {
    w_type(Type::Code, w)?;
    w_long(code.argcount, w)?;
    w_long(code.nlocals, w)?;
    w_long(code.stacksize, w)?;
    w_long(code.flags.bits(), w)?;
    // The reader extracts these via `r_object_extract_bytes` / `_tuple` /
    // `_string`, so each must be a full object, not a bare length-prefixed blob.
    w_string(code.code.as_ref(), w)?;

    w_type(Type::Tuple, w)?;
    w_long(code.consts.len() as u32, w)?;
    for konst in code.consts.iter() {
        w_object(konst, w)?;
    }

    w_string_tuple(&code.names, w)?;
    w_string_tuple(&code.varnames, w)?;
    w_string_tuple(&code.freevars, w)?;
    w_string_tuple(&code.cellvars, w)?;

    w_string(code.filename.as_ref().as_ref(), w)?;
    w_string(code.name.as_ref().as_ref(), w)?;
    w_long(code.firstlineno, w)?;
    w_string(code.lnotab.as_ref(), w)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn w_object(obj: &Obj, w: &mut impl Write) -> io::Result<()> {
    match obj {
        Obj::None => w_type(Type::None, w),
        Obj::StopIteration => w_type(Type::StopIter, w),
        Obj::Ellipsis => w_type(Type::Ellipsis, w),
        Obj::Bool(true) => w_type(Type::True, w),
        Obj::Bool(false) => w_type(Type::False, w),
        Obj::Long(x) => w_int(&x.read().unwrap(), w),
        Obj::Float(x) => {
            w_type(Type::BinaryFloat, w)?;
            w.write_all(&x.to_le_bytes())
        }
        Obj::Complex(x) => {
            w_type(Type::BinaryComplex, w)?;
            w.write_all(&x.re.to_le_bytes())?;
            w.write_all(&x.im.to_le_bytes())
        }
        // Python 2.7 marshal has no bytes type distinct from str.
        Obj::Bytes(x) => w_string(x.read().unwrap().as_slice(), w),
        Obj::String(x) => w_string(x.read().unwrap().as_ref(), w),
        Obj::Tuple(x) => {
            let items = x.read().unwrap();
            w_type(Type::Tuple, w)?;
            w_long(items.len() as u32, w)?;
            for item in items.iter() {
                w_object(item, w)?;
            }
            Ok(())
        }
        Obj::List(x) => {
            let items = x.read().unwrap();
            w_type(Type::List, w)?;
            w_long(items.len() as u32, w)?;
            for item in items.iter() {
                w_object(item, w)?;
            }
            Ok(())
        }
        Obj::Dict(x) => {
            let map = x.read().unwrap();
            w_type(Type::Dict, w)?;
            for (key, value) in map.iter() {
                let key_obj = Obj::try_from(key).expect("hashable key converts back to an object");
                w_object(&key_obj, w)?;
                w_object(value, w)?;
            }
            // A null object terminates the dict stream.
            w_type(Type::Null, w)
        }
        Obj::Set(x) => w_hashset(Type::Set, &x.read().unwrap(), w),
        Obj::FrozenSet(x) => w_hashset(Type::FrozenSet, &x.read().unwrap(), w),
        Obj::Code(x) => w_code(&x.read().unwrap(), w),
    }
}

fn w_hashset(typ: Type, set: &std::collections::HashSet<ObjHashable>, w: &mut impl Write) -> io::Result<()> {
    w_type(typ, w)?;
    w_long(set.len() as u32, w)?;
    for element in set.iter() {
        let element_obj = Obj::try_from(element).expect("hashable element converts back to an object");
        w_object(&element_obj, w)?;
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::marshal_dumps;
    use crate::read::marshal_loads;
    use crate::{Code, CodeFlags, Obj, ObjHashable};
    use bstr::BString;
    use num_bigint::BigInt;
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, RwLock};

    /// Round-trip an object through write -> read and return the re-read value.
    fn roundtrip(obj: &Obj) -> Obj {
        marshal_loads(&marshal_dumps(obj)).unwrap()
    }

    /// Round-trip and assert the Debug reprs match. Debug renders by value (not
    /// pointer), so this is a strong structural equality check -- but it is
    /// order-sensitive, so it must not be used for dicts/sets.
    fn assert_roundtrips(obj: &Obj) {
        let back = roundtrip(obj);
        assert_eq!(format!("{:?}", obj), format!("{:?}", back));
    }

    #[test]
    fn scalars() {
        assert_roundtrips(&Obj::None);
        assert_roundtrips(&Obj::StopIteration);
        assert_roundtrips(&Obj::Ellipsis);
        assert_roundtrips(&Obj::Bool(true));
        assert_roundtrips(&Obj::Bool(false));
    }

    #[test]
    fn small_and_negative_ints() {
        for v in [0_i64, 1, -1, 42, -42, i32::MAX as i64, i32::MIN as i64] {
            assert_roundtrips(&Obj::Long(Arc::new(RwLock::new(BigInt::from(v)))));
        }
    }

    #[test]
    fn big_ints_use_pylong() {
        // Values outside i32 force the TYPE_LONG (base-2^15 digit) path.
        let cases = [
            BigInt::parse_bytes(b"85070591730234615847396907784232501249", 10).unwrap(),
            -BigInt::parse_bytes(b"85070591730234615847396907784232501249", 10).unwrap(),
            BigInt::from(1) << 4096_u32,
            -(BigInt::from(1) << 4096_u32),
            BigInt::from(i64::MAX),
            BigInt::from(i64::MIN),
        ];
        for v in cases {
            let obj = Obj::Long(Arc::new(RwLock::new(v.clone())));
            let back = roundtrip(&obj);
            assert_eq!(*back.extract_long().unwrap().read().unwrap(), v);
        }
    }

    #[test]
    fn floats() {
        for v in [0.0_f64, -0.0, 1.5, -1.5, 3.7e250, f64::INFINITY, -f64::INFINITY] {
            let back = roundtrip(&Obj::Float(v));
            assert_eq!(back.extract_float().unwrap(), v);
        }
        // NaN does not equal itself; check via Debug repr instead.
        assert_roundtrips(&Obj::Float(f64::NAN));
    }

    #[test]
    fn strings_and_bytes() {
        assert_roundtrips(&Obj::String(Arc::new(RwLock::new(BString::from("")))));
        assert_roundtrips(&Obj::String(Arc::new(RwLock::new(BString::from("Andr\u{e8} Previn")))));
        assert_roundtrips(&Obj::String(Arc::new(RwLock::new(BString::from(" ".repeat(10_000))))));
        // Arbitrary (non-UTF-8) bytes must survive too.
        let raw: Vec<u8> = (0..=255u8).collect();
        assert_roundtrips(&Obj::String(Arc::new(RwLock::new(BString::from(raw)))));
    }

    #[test]
    fn tuples_and_lists() {
        let inner = vec![Obj::Bool(true), Obj::None, Obj::Long(Arc::new(RwLock::new(BigInt::from(7))))];
        assert_roundtrips(&Obj::Tuple(Arc::new(RwLock::new(inner.clone()))));
        assert_roundtrips(&Obj::List(Arc::new(RwLock::new(inner))));
        assert_roundtrips(&Obj::Tuple(Arc::new(RwLock::new(vec![]))));
    }

    #[test]
    fn nested_collections() {
        let obj = Obj::Tuple(Arc::new(RwLock::new(vec![
            Obj::List(Arc::new(RwLock::new(vec![Obj::Bool(false)]))),
            Obj::Tuple(Arc::new(RwLock::new(vec![Obj::None, Obj::Ellipsis]))),
            Obj::String(Arc::new(RwLock::new(BString::from("deep")))),
        ])));
        assert_roundtrips(&obj);
    }

    #[test]
    fn dicts_by_value() {
        let mut map = HashMap::new();
        map.insert(
            ObjHashable::String(Arc::new(BString::from("a"))),
            Obj::Long(Arc::new(RwLock::new(BigInt::from(1)))),
        );
        map.insert(ObjHashable::String(Arc::new(BString::from("b"))), Obj::Bool(true));
        let obj = Obj::Dict(Arc::new(RwLock::new(map)));
        let back = roundtrip(&obj);
        let back = back.extract_dict().unwrap();
        let back = back.read().unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(
            *back[&ObjHashable::String(Arc::new(BString::from("a")))].clone().extract_long().unwrap().read().unwrap(),
            BigInt::from(1)
        );
        assert!(back[&ObjHashable::String(Arc::new(BString::from("b")))].clone().extract_bool().unwrap());
    }

    #[test]
    fn sets_by_value() {
        let mut set = HashSet::new();
        set.insert(ObjHashable::String(Arc::new(BString::from("x"))));
        set.insert(ObjHashable::String(Arc::new(BString::from("y"))));
        for obj in [Obj::Set(Arc::new(RwLock::new(set.clone()))), Obj::FrozenSet(Arc::new(RwLock::new(set.clone())))] {
            let back = roundtrip(&obj);
            let len = match &back {
                Obj::Set(s) => s.read().unwrap().len(),
                Obj::FrozenSet(s) => s.read().unwrap().len(),
                _ => panic!("expected a set"),
            };
            assert_eq!(len, 2);
        }
    }

    fn sample_code() -> Code {
        Code {
            argcount: 1,
            nlocals: 2,
            stacksize: 5,
            flags: CodeFlags::NOFREE | CodeFlags::NEWLOCALS | CodeFlags::OPTIMIZED,
            code: Arc::new(Vec::from(&b"d\x00\x00S"[..])),
            consts: Arc::new(vec![
                Obj::None,
                Obj::Long(Arc::new(RwLock::new(BigInt::from(99)))),
                Obj::String(Arc::new(RwLock::new(BString::from("konst")))),
            ]),
            names: vec![Arc::new(BString::from("marshal")), Arc::new(BString::from("loads"))],
            varnames: vec![Arc::new(BString::from("self")), Arc::new(BString::from("new"))],
            freevars: vec![],
            cellvars: vec![],
            filename: Arc::new(BString::from("<string>")),
            name: Arc::new(BString::from("test_fn")),
            firstlineno: 3,
            lnotab: Arc::new(vec![0, 1, 16, 1]),
        }
    }

    #[test]
    fn code_object_fields() {
        let obj = Obj::Code(Arc::new(RwLock::new(sample_code())));
        let back = roundtrip(&obj);
        let back = back.extract_code().unwrap();
        let back = back.read().unwrap();
        let original = sample_code();
        assert_eq!(back.argcount, original.argcount);
        assert_eq!(back.nlocals, original.nlocals);
        assert_eq!(back.stacksize, original.stacksize);
        assert_eq!(back.flags, original.flags);
        assert_eq!(back.code, original.code);
        assert_eq!(back.consts.len(), original.consts.len());
        assert_eq!(back.firstlineno, original.firstlineno);
        assert_eq!(back.lnotab, original.lnotab);
        assert_eq!(*back.filename, *original.filename);
        assert_eq!(*back.name, *original.name);
        assert!(back.names.iter().map(|n| n.to_string()).eq(original.names.iter().map(|n| n.to_string())));
        assert!(back.varnames.iter().map(|n| n.to_string()).eq(original.varnames.iter().map(|n| n.to_string())));
    }

    #[test]
    fn nested_code_in_consts() {
        // A module whose const tuple holds a nested function code object.
        let inner = Obj::Code(Arc::new(RwLock::new(sample_code())));
        let mut module = sample_code();
        module.consts = Arc::new(vec![Obj::None, inner]);
        let obj = Obj::Code(Arc::new(RwLock::new(module)));
        let back = roundtrip(&obj);
        let back = back.extract_code().unwrap();
        let back = back.read().unwrap();
        assert_eq!(back.consts.len(), 2);
        let nested = back.consts[1].clone().extract_code().unwrap();
        assert_eq!(*nested.read().unwrap().name, "test_fn");
    }

    /// The encoding is deterministic and stable across a full write -> read ->
    /// write cycle: serializing the re-read object reproduces the exact bytes.
    /// This exercises the writer end to end (a nested module code object) and
    /// confirms the reader consumes our output without depending on any stale
    /// external vector.
    #[test]
    fn write_read_write_is_byte_stable() {
        let inner = Obj::Code(Arc::new(RwLock::new(sample_code())));
        let mut module = sample_code();
        module.consts = Arc::new(vec![
            Obj::None,
            Obj::Long(Arc::new(RwLock::new(BigInt::from(1) << 200))),
            Obj::String(Arc::new(RwLock::new(BString::from("module")))),
            inner,
        ]);
        let obj = Obj::Code(Arc::new(RwLock::new(module)));

        let first = marshal_dumps(&obj);
        let reread = marshal_loads(&first).unwrap();
        let second = marshal_dumps(&reread);
        assert_eq!(first, second, "re-serializing a re-read object must be byte-identical");
    }
}
