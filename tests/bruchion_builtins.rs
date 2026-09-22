//! The CJC-language entry points for the Bruchion switch and the fused loss-and-gradient:
//! `bruchion_kernels(on)`, `bruchion_kernels_enabled()`, `mse_loss_grad(pred, target)`.
//! Held to the Rust APIs they wrap, bit for bit, and run through both executors from
//! CJC source. This test crate builds cjc-runtime without the feature, so the switch
//! records its state and routes nothing; the feature build's parity is cjc-runtime's.

use cjc_runtime::builtins::dispatch_builtin;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn run(mir: bool, body: &str) -> Result<Vec<String>, String> {
    let src = format!("fn main() {{\n{body}\n}}\n");
    let (program, diags) = cjc_parser::parse_source(&src);
    assert!(!diags.has_errors(), "parse errors:\n{:#?}\nsource:\n{src}", diags.diagnostics);
    if mir {
        cjc_mir_exec::run_program_with_executor(&program, 42)
            .map(|(_, exec)| exec.output)
            .map_err(|e| format!("{e:?}"))
    } else {
        let mut interp = cjc_eval::Interpreter::new(42);
        interp.exec(&program).map(|_| interp.output).map_err(|e| format!("{e:?}"))
    }
}

#[test]
fn the_switch_builtins_wrap_the_runtime_policy() {
    cjc_runtime::runtime_policy::set_bruchion_kernels(false);
    let prev = dispatch_builtin("bruchion_kernels", &[Value::Bool(true)]).unwrap().unwrap();
    assert!(matches!(prev, Value::Bool(false)), "returns the previous state");
    assert!(cjc_runtime::runtime_policy::get().bruchion_kernels, "the switch is on");
    let on = dispatch_builtin("bruchion_kernels_enabled", &[]).unwrap().unwrap();
    assert!(matches!(on, Value::Bool(b) if b == cjc_runtime::bruchion::dispatch::enabled()));
    let prev = dispatch_builtin("bruchion_kernels", &[Value::Bool(false)]).unwrap().unwrap();
    assert!(matches!(prev, Value::Bool(true)));
    assert!(!cjc_runtime::runtime_policy::get().bruchion_kernels);
    assert!(matches!(dispatch_builtin("bruchion_kernels_enabled", &[]).unwrap().unwrap(), Value::Bool(false)));
    assert!(dispatch_builtin("bruchion_kernels", &[Value::Float(1.0)]).is_err(), "a Float is refused");
    assert!(dispatch_builtin("bruchion_kernels", &[]).is_err());
    assert!(dispatch_builtin("bruchion_kernels_enabled", &[Value::Bool(true)]).is_err());
}

#[test]
fn the_mse_loss_grad_builtin_is_the_rust_function_bit_for_bit() {
    let n = 1001;
    let mut rng = cjc_repro::Rng::seeded(7);
    let p: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let t: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let mut want_grad = vec![0.0; n];
    let want_loss = cjc_runtime::ml::mse_loss_grad(&p, &t, &mut want_grad).unwrap();
    let pt = Tensor::from_vec(p.clone(), &[n]).unwrap();
    let tt = Tensor::from_vec(t.clone(), &[n]).unwrap();
    let got = dispatch_builtin("mse_loss_grad", &[Value::Tensor(pt), Value::Tensor(tt)]).unwrap().unwrap();
    let Value::Tuple(parts) = got else { panic!("a tuple, got {got:?}") };
    assert_eq!(parts.len(), 2);
    let Value::Float(loss) = parts[0] else { panic!("a Float first") };
    let Value::Tensor(grad) = &parts[1] else { panic!("a Tensor second") };
    assert_eq!(loss.to_bits(), want_loss.to_bits());
    assert_eq!(grad.shape(), &[n]);
    let gb: Vec<u64> = grad.to_vec().iter().map(|v| v.to_bits()).collect();
    let wb: Vec<u64> = want_grad.iter().map(|v| v.to_bits()).collect();
    assert_eq!(gb, wb);
    // The witness pins the definition (n = 3, d = 2.9): h + h, not (2 d) / 3.
    let p3 = Tensor::from_vec(vec![2.9, 0.2, -0.4], &[3]).unwrap();
    let t3 = Tensor::from_vec(vec![0.0; 3], &[3]).unwrap();
    let got = dispatch_builtin("mse_loss_grad", &[Value::Tensor(p3), Value::Tensor(t3)]).unwrap().unwrap();
    let Value::Tuple(parts) = got else { panic!() };
    let Value::Tensor(grad) = &parts[1] else { panic!() };
    let h = (1.0f64 / 3.0) * 2.9;
    assert_eq!(grad.to_vec()[0].to_bits(), (h + h).to_bits());
    assert_ne!(grad.to_vec()[0].to_bits(), ((2.0f64 * 2.9) / 3.0).to_bits());
    // Shape and arity errors.
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert!(dispatch_builtin("mse_loss_grad", &[Value::Tensor(a.clone()), Value::Tensor(b)]).is_err());
    assert!(dispatch_builtin("mse_loss_grad", &[Value::Tensor(a)]).is_err());
}

#[test]
fn the_builtins_run_from_cjc_source_in_both_executors() {
    let body = r#"
        let p: Tensor = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
        let t: Tensor = Tensor.from_vec([0.0, 0.0, 0.0], [3]);
        let r: Any = mse_loss_grad(p, t);
        print(r.0);
        print(r.1);
        let was: Bool = bruchion_kernels(true);
        print(was);
        print(bruchion_kernels_enabled());
        bruchion_kernels(false);
    "#;
    let e = run(false, body).unwrap_or_else(|err| panic!("eval: {err}"));
    let m = run(true, body).unwrap_or_else(|err| panic!("mir: {err}"));
    assert_eq!(e, m, "AST and MIR executors must print the same");
    cjc_runtime::runtime_policy::set_bruchion_kernels(false);
    let joined = e.join("\n");
    assert!(joined.contains("4.666666666666667"), "the loss (1 + 4 + 9) / 3:\n{joined}");
    assert!(joined.contains("0.6666666666666666"), "grad[0] = 2/3:\n{joined}");
    assert!(joined.contains("false"), "the switch was off:\n{joined}");
}
