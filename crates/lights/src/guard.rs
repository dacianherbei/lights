use crate::error::{LightsError, Result};

#[cold]
#[inline(never)]
pub fn unreachable(site: &'static str, detail: &str) -> ! {
    panic!("lights: unreachable branch at {site}: {detail}");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    site: &'static str,
    limit: u64,
    used: u64,
}

impl Budget {
    pub fn new(site: &'static str, limit: u64) -> Self {
        Self {
            site,
            limit,
            used: 0,
        }
    }

    pub fn tick(&mut self) -> Result<()> {
        if self.used >= self.limit {
            return Err(LightsError::BudgetExhausted {
                site: self.site,
                limit: self.limit,
            });
        }
        self.used += 1;
        Ok(())
    }

    pub fn tick_or_die(&mut self) {
        match self.tick() {
            Ok(()) => (),
            Err(e) => unreachable("Budget::tick_or_die", &format!("{e}")),
        }
    }

    pub fn site(&self) -> &'static str {
        self.site
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }
}
