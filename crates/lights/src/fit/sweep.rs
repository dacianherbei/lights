use crate::error::Result;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Concurrency {
    Serial,
    Parallel,
}

impl Concurrency {
    pub const ALL: [Concurrency; 2] = [Concurrency::Serial, Concurrency::Parallel];

    pub fn name(self) -> &'static str {
        match self {
            Concurrency::Serial => "serial",
            Concurrency::Parallel => "parallel",
        }
    }

    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        match text {
            "serial" => Ok(Concurrency::Serial),
            "parallel" => Ok(Concurrency::Parallel),
            other => Err(format!("--sweep takes serial or parallel, not {other}")),
        }
    }
}

pub fn map<J, T, F>(concurrency: Concurrency, jobs: &[J], work: F) -> Result<Vec<T>>
where
    J: Sync,
    T: Send,
    F: Fn(&J) -> Result<T> + Sync + Send,
{
    let outcomes: Vec<Result<T>> = match concurrency {
        Concurrency::Serial => jobs.iter().map(&work).collect(),
        Concurrency::Parallel => jobs.par_iter().map(&work).collect(),
    };
    let mut out = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        out.push(outcome?);
    }
    Ok(out)
}
