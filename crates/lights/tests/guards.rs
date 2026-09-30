use lights::error::LightsError;
use lights::guard::{Budget, unreachable};

#[test]
fn unreachable_panics_with_site() {
    let caught = std::panic::catch_unwind(|| unreachable("test-site", "test-detail"));
    let payload = match caught {
        Ok(_) => panic!("guard::unreachable returned instead of panicking"),
        Err(p) => p,
    };
    let text = match payload.downcast_ref::<String>() {
        Some(s) => s.clone(),
        None => match payload.downcast_ref::<&str>() {
            Some(s) => (*s).to_string(),
            None => panic!("panic payload was not a string"),
        },
    };
    assert!(text.contains("test-site"), "message missing site: {text}");
    assert!(text.contains("test-detail"), "message missing detail: {text}");
}

#[test]
fn budget_exhausts_at_limit_not_before() {
    let mut b = Budget::new("s", 3);
    assert!(b.tick().is_ok());
    assert!(b.tick().is_ok());
    assert!(b.tick().is_ok());
    assert_eq!(b.used(), 3);
    assert_eq!(b.limit(), 3);
    match b.tick() {
        Err(LightsError::BudgetExhausted { site, limit }) => {
            assert_eq!(site, "s");
            assert_eq!(limit, 3);
        }
        other => panic!("expected BudgetExhausted, got {other:?}"),
    }
    assert_eq!(b.used(), 3);
    b.reset();
    assert_eq!(b.used(), 0);
    assert!(b.tick().is_ok());
}

#[test]
fn budget_tick_or_die_panics_past_the_limit() {
    let mut b = Budget::new("die-site", 2);
    b.tick_or_die();
    b.tick_or_die();
    let caught = std::panic::catch_unwind(move || {
        let mut local = Budget::new("die-site", 2);
        local.tick_or_die();
        local.tick_or_die();
        local.tick_or_die();
    });
    assert!(caught.is_err(), "tick_or_die did not fail past the limit");
}
