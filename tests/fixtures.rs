//! Parses marshal fixtures produced by real Python 2.7 (see
//! `scripts/gen_fixtures.py`) and snapshots the parsed result. The .bin files
//! are checked in, so this runs without any Python interpreter.
use py27_marshal::read::marshal_loads;

/// Load a fixture, parse it, and snapshot its Debug repr under the fixture name.
macro_rules! fixture_snapshot {
    ($name:ident) => {
        #[test]
        fn $name() {
            let bytes = include_bytes!(concat!("fixtures/", stringify!($name), ".bin"));
            let obj = marshal_loads(bytes).expect("fixture parses");
            insta::assert_debug_snapshot!(obj);
        }
    };
}

fixture_snapshot!(none);
fixture_snapshot!(bool_true);
fixture_snapshot!(bool_false);
fixture_snapshot!(int);
fixture_snapshot!(int_neg);
fixture_snapshot!(int_big);
fixture_snapshot!(int_big_neg);
fixture_snapshot!(float);
fixture_snapshot!(float_neg);
fixture_snapshot!(complex);
fixture_snapshot!(str_bytes);
fixture_snapshot!(unicode);
fixture_snapshot!(unicode_empty);
fixture_snapshot!(tuple);
fixture_snapshot!(tuple_empty);
fixture_snapshot!(list);
fixture_snapshot!(nested);
fixture_snapshot!(code);
fixture_snapshot!(dict_tuple_key);

/// Dicts and sets hash-order their entries, so snapshotting the repr would be
/// flaky. Assert structurally instead.
#[test]
fn dict() {
    use bstr::BString;
    use num_bigint::BigInt;
    use py27_marshal::ObjHashable;
    use std::sync::Arc;

    let bytes = include_bytes!("fixtures/dict.bin");
    let dict = marshal_loads(bytes).unwrap().extract_dict().unwrap();
    let dict = dict.read().unwrap();
    assert_eq!(dict.len(), 2);
    for key in ["a", "b"] {
        let v = dict[&ObjHashable::String(Arc::new(BString::from(key)))].clone().extract_long().unwrap();
        let expected = if key == "a" { 1 } else { 2 };
        assert_eq!(*v.read().unwrap(), BigInt::from(expected));
    }
}

#[test]
fn set() {
    let bytes = include_bytes!("fixtures/set.bin");
    let set = marshal_loads(bytes).unwrap().extract_set().unwrap();
    assert_eq!(set.read().unwrap().len(), 3);
}

#[test]
fn frozenset() {
    let bytes = include_bytes!("fixtures/frozenset.bin");
    let set = marshal_loads(bytes).unwrap().extract_frozenset().unwrap();
    assert_eq!(set.read().unwrap().len(), 2);
}

/// Two code objects sharing interned strings: the second resolves its names
/// through the reader's reference table.
#[test]
fn code_tuple() {
    let bytes = include_bytes!("fixtures/code_tuple.bin");
    let tuple = marshal_loads(bytes).unwrap().extract_tuple().unwrap();
    let tuple = tuple.read().unwrap();
    assert_eq!(tuple.len(), 2);
    for o in tuple.iter() {
        let code = o.clone().extract_code().unwrap();
        assert_eq!(*code.read().unwrap().name, "<module>");
    }
}
