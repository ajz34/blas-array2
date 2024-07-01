pub type AnyError = Box<dyn std::error::Error>;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BLASError(pub String);

impl std::error::Error for BLASError {}

impl BLASError {
    pub fn assert(cond: bool, s: String) -> Result<(), BLASError> {
        match cond {
            true => Ok(()),
            false => Err(BLASError(s)),
        }
    }
}

impl std::fmt::Display for BLASError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}