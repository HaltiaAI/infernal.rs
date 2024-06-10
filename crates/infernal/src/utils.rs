use core::fmt;

/// Representing a unit of measure for computer storage (bytes, KiB, MiB, ...). Useful for printing
/// out large byte values.
pub struct Bytes(pub usize);

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut value = self.0 as f64;
        let mut unit = "B";
        for u in &["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"] {
            if value.abs() < 1024.0 {
                unit = u;
                break;
            }
            value /= 1024.0;
        }
        if unit == "B" {
            write!(f, "{} bytes", self.0)
        } else {
            write!(f, "{} bytes ({:.2} {})", self.0, value, unit)
        }
    }
}

impl fmt::Debug for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}
